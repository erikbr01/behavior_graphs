"""LIBERO-PRO environment adapter (perturbation-robust LIBERO evaluation).

LIBERO-PRO (Zxy-MLlab/LIBERO-PRO; fork erikbr01/libero_pro) extends LIBERO
with perturbed task suites probing generalization beyond memorization. It
registers them in the standard ``libero.libero.benchmark`` registry as
``<base>_<perturbation>`` (e.g. ``libero_spatial_env``) with base suites
``libero_{spatial,object,goal,10}`` and perturbations:

==========  =====================================================
``object``  object appearance/color/scale replacement
``swap``    position perturbation (objects relocated/swapped)
``lan``     semantic paraphrase of the language instruction
``task``    task redefinition (new goal; not combinable)
``env``     environment replacement (e.g. main table -> kitchen)
``temp``    combined perturbations generated from a YAML config
==========  =====================================================

Because the registry API is identical to LIBERO's, this adapter reuses the
whole openpi eval pathway from :class:`LiberoAdapter` and only switches the
entry point to ``examples/libero/eval_save_with_embeddings_pro.py`` (added to
the openpi fork), which resolves the per-episode step budget for perturbed
suite names and validates that the suite's bddl/init files are present.

Setup, inside the openpi submodule's uv env (LIBERO-PRO ships the same
``libero`` package name and replaces the bundled LIBERO)::

    cd third_party/openpi
    uv pip install -e /path/to/LIBERO-PRO

then download the perturbed bddl/init files from
https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro into
``LIBERO-PRO/libero/libero/{bddl_files,init_files}/`` (or generate them with
LIBERO-PRO's ``perturbation.create_env``).

Caveats
-------
* This wiring is verified against the LIBERO-PRO and openpi sources but has
  not been executed end-to-end (no GPU budget for eval runs).
* :meth:`training_data` / :meth:`train_policy` are inherited unchanged:
  LIBERO-PRO is an evaluation-only benchmark for policies trained on
  standard LIBERO.
"""

from __future__ import annotations

from typing import Any

from behavior_graphs.core.env import Policy, RolloutSet
from behavior_graphs.envs.libero import LiberoAdapter

PERTURBATIONS = ("object", "swap", "lan", "task", "env", "temp")


class LiberoProAdapter(LiberoAdapter):
    """LIBERO-PRO perturbed suites via the openpi submodule."""

    name = "libero_pro"

    _EVAL_SCRIPT = "examples/libero/eval_save_with_embeddings_pro.py"

    def __init__(
        self,
        task_suite: str = "libero_spatial",
        perturbation: str | None = "env",
        max_steps: int | None = None,
        checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero",
        config_name: str = "pi05_libero",
        repo_root: str | None = None,
        submodule: str = "third_party/openpi",
        num_trials_per_task: int = 20,
        replan_steps: int = 5,
        resize_size: int = 224,
        demo_data_dir: str | None = None,
        gpu: int | None = None,
    ) -> None:
        """
        Args:
            task_suite: base LIBERO suite (libero_spatial | libero_object |
                libero_goal | libero_10). The evaluated suite is
                ``<task_suite>_<perturbation>``.
            perturbation: one of ``object | swap | lan | task | env | temp``,
                or ``None`` to use ``task_suite`` verbatim as a full
                LIBERO-PRO suite name (e.g. ``libero_goal_swap``).
            max_steps: per-episode env step budget override; default is the
                base suite's budget (resolved by the eval script).
            (remaining args as in :class:`LiberoAdapter`)
        """
        if perturbation is None:
            suite = task_suite
        else:
            if perturbation not in PERTURBATIONS:
                raise ValueError(
                    f"unknown perturbation {perturbation!r}; expected one of "
                    f"{PERTURBATIONS} or None"
                )
            suite = f"{task_suite}_{perturbation}"
        super().__init__(
            task_suite=suite,
            checkpoint=checkpoint,
            config_name=config_name,
            repo_root=repo_root,
            submodule=submodule,
            num_trials_per_task=num_trials_per_task,
            replan_steps=replan_steps,
            resize_size=resize_size,
            demo_data_dir=demo_data_dir,
            gpu=gpu,
        )
        self.perturbation = perturbation
        self.max_steps = max_steps

    def collect_rollouts(
        self,
        policy: Policy,
        output_dir,
        num_episodes: int,
        seed: int = 0,
        **kwargs: Any,
    ) -> RolloutSet:
        """Roll out on the perturbed suite (see :class:`LiberoAdapter`)."""
        if self.max_steps is not None:
            kwargs.setdefault("max_steps", self.max_steps)
        return super().collect_rollouts(policy, output_dir, num_episodes, seed=seed, **kwargs)
