"""Demo-source adapter: robo_dataset stores first-class, robomimic hdf5 fallback.

Demo scoring/curation needs per-demo feature arrays from whatever holds the
demonstrations. The first-class source is a robo_dataset store directory
(what our converters and ``scripts/datasets/build_salvage_datasets.py`` produce); the
robomimic hdf5 reader is kept ONLY for external robomimic downloads that were
never converted — it is also the only source of the ``mask/`` quality groups
that drive oracle demo weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DemoSet:
    """Demos in a fixed order: names, per-demo (T, D) features, mask groups.

    ``mask_groups`` maps a robomimic ``mask/`` group name (e.g.
    ``"okay_better"``) to the demo names it contains; it is empty for sources
    without mask annotations (robo_dataset stores).
    """

    names: list[str]
    features: list[np.ndarray]
    mask_groups: dict[str, list[str]] = field(default_factory=dict)


def load_demo_source(path: Path | str, obs_keys: list[str]) -> DemoSet:
    """Load per-demo features from a robo_dataset store or a robomimic hdf5.

    A directory is opened as a robo_dataset store (format auto-detected) and
    ``obs_keys`` name store keys, concatenated per episode. An ``.hdf5``/
    ``.h5`` file falls back to the robomimic layout (``data/<demo>/obs/<key>``
    in numeric demo order, plus ``mask/`` quality groups).
    """
    path = Path(path)
    if path.is_dir():
        return _load_store_demos(path, obs_keys)
    if path.suffix in (".hdf5", ".h5"):
        return _load_robomimic_hdf5_demos(path, obs_keys)
    raise ValueError(
        f"demo source {path} is neither a robo_dataset store directory nor a robomimic hdf5 file"
    )


def _load_store_demos(path: Path, obs_keys: list[str]) -> DemoSet:
    from robo_dataset.store_factory import open_store

    store = open_store(str(path))
    missing = [k for k in obs_keys if k not in store.keys]
    if missing:
        raise KeyError(f"store {path} lacks keys {missing}; available: {sorted(store.keys)}")
    names, features = [], []
    for i in range(store.n_episodes):
        n = store.episode_length(i)
        names.append(f"episode_{i}")
        features.append(
            np.concatenate(
                [np.asarray(store.get_frames(i, k, 0, n), dtype=np.float32) for k in obs_keys],
                axis=1,
            )
        )
    return DemoSet(names=names, features=features)


def _load_robomimic_hdf5_demos(path: Path, obs_keys: list[str]) -> DemoSet:
    """hdf5 fallback for external robomimic downloads (numeric demo order)."""
    import h5py

    names, features, groups = [], [], {}
    with h5py.File(path, "r") as f:
        for demo in sorted(f["data"].keys(), key=lambda d: int(d.split("_")[-1])):
            missing = [k for k in obs_keys if k not in f["data"][demo]["obs"]]
            if missing:
                raise KeyError(
                    f"{demo} lacks obs keys {missing}; "
                    f"available: {sorted(f['data'][demo]['obs'].keys())}"
                )
            names.append(demo)
            features.append(
                np.concatenate(
                    [np.asarray(f["data"][demo]["obs"][k], dtype=np.float32) for k in obs_keys],
                    axis=1,
                )
            )
        if "mask" in f:
            groups = {grp: [raw.decode() for raw in f["mask"][grp]] for grp in f["mask"]}
    return DemoSet(names=names, features=features, mask_groups=groups)
