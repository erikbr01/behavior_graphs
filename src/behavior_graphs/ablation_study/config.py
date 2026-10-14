"""One dataset description, loaded from YAML, that every stage of the study reads.

Before this, each stage carried its own copy of the paths and each analysis
script hardcoded ``abc_bottles_gpu``. Adding a second dataset meant editing
twenty files, and the one place the duplication actually bit was the policy's
training set: the study's ``splits.json`` and the policy's ``episode_ids`` were
maintained separately, the latter went stale, and 76 of the 87 supposedly
held-out bottles episodes ended up in policy training.

So a dataset is one file. Paths, the split rule, which representations to sweep
and how to score are all read from it, and nothing downstream is allowed to
guess.

Two split forms are supported deliberately:

``fractions``
    Seeded train/val/test over the annotated episodes. What new datasets use.

``file``
    An existing ``splits.json`` with its own key names. The bottles sweep was fit
    and scored under the older nested ``val_frac``/``inner_test_frac`` scheme, so
    its cached results are only reproducible against that file. Re-splitting
    bottles at 60/20/20 is a re-run of the whole grid, not a config edit.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "ablation_study"


@dataclasses.dataclass(frozen=True)
class SplitSpec:
    """How to divide episodes, either generated or read from a file."""

    seed: int = 0
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    annotated_only: bool = True
    #: Read an existing splits.json instead of generating. Mutually exclusive
    #: with the fractions above, which are then ignored.
    file: str | None = None
    #: For a legacy file: which key is the fit set, the selection set, the
    #: reported set. Named here rather than assumed so old and new layouts can
    #: coexist without a translation layer scattered through the code.
    keys: dict[str, str] = dataclasses.field(
        default_factory=lambda: {"train": "train", "val": "val", "test": "test"}
    )

    def __post_init__(self) -> None:
        if self.file is not None:
            return
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total:g}")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("every split fraction must be positive")


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    """What to score and which number decides."""

    gt_set: str = "task"
    #: IoU 0.75, not 0.50. At 0.50 a count-matched uniform predictor scores
    #: 0.13-0.53 depending on the task, so most of the criterion's range lies
    #: below a trivial baseline -- and the best real cells sat at 0.50-0.53 on
    #: earbuds, i.e. inside that dead zone. `probes/metric_validity.py` scores
    #: a ladder of deliberately degraded ground truths and finds F1@0.50
    #: ranking uniform *above* ground truth whose boundaries are jittered by
    #: ten windows, while at 0.75 every degraded rung stays above it on two of
    #: three tasks. 0.75 is also what WGO-Bench publishes against, so the
    #: numbers are externally comparable: their fixed-length baseline scores
    #: 0.070 and their best method 0.306.
    #:
    #: Switching this re-ranks: the winning cell moves on 28 of 48 (task, arm)
    #: pairs and the representation ranking changes on all eight tasks, though
    #: the top arm per task does not move. `probes/rerank_075.py` has the diff.
    criterion: str = "segmental_f1@0.75"
    iou_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    min_segments: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    #: Only used by the boundary metric; the segmental metric is IoU-based and
    #: needs no tolerance, which is why it survives a change of dataset tempo.
    tolerance: float | None = None
    #: Reported, not enforced. The old justification for that -- "raw segmental
    #: F1 is single-peaked in density and so needs no band" -- was never true:
    #: mean F1 rises monotonically with density across the observed range
    #: (0.239 to 0.514 on earbuds) because only 6 of 111 cells sit above
    #: density 1.0, so the over-dense half of the band is simply unsampled. At
    #: IoU 0.75 over-segmentation is punished hard enough (a 3x-too-dense
    #: degraded ground truth falls to 0.127) that the band matters less, but
    #: the claim above should not be relied on.
    density_band: tuple[float, float] = (0.8, 1.5)
    #: Measured across folds; differences below this are not resolvable.
    noise_floor: float = 0.027
    #: Smallest K worth reporting on this task, or 0 for "every K the grid ran".
    #:
    #: Not a filter for tidiness. A partition into fewer clusters than the
    #: ground truth has segments per episode cannot represent that structure
    #: however good the representation is, so those cells answer a different
    #: question -- and pooling them is not harmless: on the cooking collection
    #: (10.9 segments an episode) including K = 4, 6, 8 made PCA beat UMAP on
    #: 67% of matched pairs, and dropping them turned that into 56% with a mean
    #: difference of +0.004. The reduction ordering reversed on an unusable
    #: region of the grid.
    #:
    #: Per task because the right floor is a property of the task's own ground
    #: truth: whiteboard averages 10.1 segments an episode and chess 39.3, so
    #: one global floor is either too high for one or too low for the other.
    #: Reported in every caption that applies it, rather than applied silently.
    k_floor: int = 0


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: str
    root: pathlib.Path
    sweep_dir: pathlib.Path
    raw: pathlib.Path | None = None
    rate: int = 10
    representations: tuple[str, ...] = ("state", "rgrl_dit", "rgrl_cond", "rgrl_enc")
    grid: str | None = None
    #: The dataset key run_sweep names its subdirectory after. Defaults to
    #: `name`, but bottles' existing 5,176 jobs already live under `bottles174`,
    #: and a new key would put later passes in a sibling directory the reports
    #: would not find.
    sweep_key: str | None = None
    #: Per-representation grid override, e.g. {dinov3: ablation_dinov3}. A frozen
    #: vision backbone is 3,072-d against the policy taps' 512, which makes the
    #: full window cross I/O-bound rather than compute-bound, so it runs a
    #: subgrid. Mapped here rather than special-cased in the runner.
    grids: dict[str, str] = dataclasses.field(default_factory=dict)
    policy_run: pathlib.Path | None = None
    policy_zarr: pathlib.Path | None = None
    #: Whether the training store holds only the annotated episodes. Separate
    #: from ``split.annotated_only``, which scopes *evaluation*, because the two
    #: genuinely differ: bottles' store is full, so excluding its held-out
    #: episodes still leaves 1,070 to train on, while legos' is annotated-only
    #: and the same exclusion leaves 607. `rgrl_train_ids` applies one exclusion
    #: rule to both; this is the knob that makes them differ, so it is stated
    #: rather than inferred. Deriving it from ``split.annotated_only`` would
    #: rebuild the bottles store at 174 episodes instead of 1,192.
    zarr_annotated_only: bool = True
    split: SplitSpec = dataclasses.field(default_factory=SplitSpec)
    evaluation: EvalSpec = dataclasses.field(default_factory=EvalSpec)

    @property
    def rescored(self) -> pathlib.Path:
        return self.sweep_dir / "rescored.json"

    @property
    def splits_path(self) -> pathlib.Path:
        """Where the resolved split lives, generated or legacy."""
        if self.split.file:
            p = pathlib.Path(self.split.file)
            return p if p.is_absolute() else self.root / p
        return self.root / "splits.json"

    @property
    def annotations(self) -> pathlib.Path:
        return self.root / "annotations" / "segments.json"

    def grid_for(self, layer: str) -> str:
        """Which grid config a representation sweeps over."""
        return self.grids.get(layer) or self.grid or "ablation_bottles"

    @property
    def key(self) -> str:
        """Dataset key for run_sweep, which becomes the sweep subdirectory."""
        return self.sweep_key or self.name

    @property
    def sweep_fold(self) -> str:
        """Which ``split.fold`` the pipeline should use when sweeping.

        ``resolve_split`` maps ``val`` to the file's (train, val) keys and
        ``inner`` to (train_inner, test_inner). A generated three-way split fits
        on ``train`` and selects on ``val``, leaving ``test`` for the report, so
        it wants ``val``; the legacy bottles file keeps its reported half under
        ``val`` and selects on ``test_inner``, so it wants ``inner``. Derived
        rather than configured, because getting it wrong silently selects
        hyperparameters on the reported set.
        """
        return "inner" if self.split.file else "val"


def _as_path(v: Any) -> pathlib.Path | None:
    return None if v is None else pathlib.Path(str(v)).expanduser()


def load(name: str | pathlib.Path) -> DatasetConfig:
    """Load ``configs/ablation_study/<name>.yaml``, or a path to one."""
    p = pathlib.Path(name)
    if not p.suffix:
        p = CONFIG_DIR / f"{name}.yaml"
    if not p.exists():
        avail = sorted(q.stem for q in CONFIG_DIR.glob("*.yaml"))
        raise SystemExit(f"no study config {name!r}; available: {', '.join(avail) or 'none'}")
    raw = yaml.safe_load(p.read_text()) or {}

    split = SplitSpec(**(raw.pop("split", None) or {}))
    ev = raw.pop("evaluation", None) or {}
    for k in ("iou_thresholds", "min_segments", "density_band"):
        if k in ev and ev[k] is not None:
            ev[k] = tuple(ev[k])
    evaluation = EvalSpec(**ev)

    reps = raw.pop("representations", None)
    return DatasetConfig(
        name=raw.pop("name", p.stem),
        task=raw.pop("task"),
        root=_as_path(raw.pop("root")),
        sweep_dir=_as_path(raw.pop("sweep_dir")),
        raw=_as_path(raw.pop("raw", None)),
        rate=int(raw.pop("rate", 10)),
        representations=tuple(reps) if reps else DatasetConfig.representations,
        grid=raw.pop("grid", None),
        sweep_key=raw.pop("sweep_key", None),
        grids=dict(raw.pop("grids", None) or {}),
        policy_run=_as_path(raw.pop("policy_run", None)),
        policy_zarr=_as_path(raw.pop("policy_zarr", None)),
        zarr_annotated_only=bool(raw.pop("zarr_annotated_only", True)),
        split=split,
        evaluation=evaluation,
    )


def available() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
