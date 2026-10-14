"""Environment adapters.

Adapters are imported lazily so that importing :mod:`behavior_graphs.envs`
never requires simulator or policy-learning dependencies.
"""

from typing import TYPE_CHECKING

_ADAPTERS = {
    "RobomimicAdapter": "behavior_graphs.envs.robomimic",
    "MimicgenAdapter": "behavior_graphs.envs.mimicgen",
    "DroidAdapter": "behavior_graphs.envs.droid",
    "LiberoAdapter": "behavior_graphs.envs.libero",
    "LiberoProAdapter": "behavior_graphs.envs.libero_pro",
    "VlaFoundryAdapter": "behavior_graphs.envs.vla_foundry",
    # rgrl_env.py defines RgrlAdapter but it was never registered here, so
    # `from behavior_graphs.envs import RgrlAdapter` raised ImportError while its
    # sibling WarpRMAdapter resolved.
    "RgrlAdapter": "behavior_graphs.envs.rgrl_env",
    "DemoSet": "behavior_graphs.envs.demo_source",
    "load_demo_source": "behavior_graphs.envs.demo_source",
    "TrajectoryDirectory": "behavior_graphs.envs.trajectory_dir",
    "RolloutDirectory": "behavior_graphs.envs.trajectory_dir",
    "write_trajectory_dir": "behavior_graphs.envs.trajectory_dir",
    "write_rollout_dir": "behavior_graphs.envs.trajectory_dir",
}

__all__ = list(_ADAPTERS)


def __getattr__(name: str):
    if name in _ADAPTERS:
        import importlib

        module = importlib.import_module(_ADAPTERS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from behavior_graphs.envs.demo_source import DemoSet as DemoSet
    from behavior_graphs.envs.demo_source import load_demo_source as load_demo_source
    from behavior_graphs.envs.droid import DroidAdapter as DroidAdapter
    from behavior_graphs.envs.libero import LiberoAdapter as LiberoAdapter
    from behavior_graphs.envs.libero_pro import LiberoProAdapter as LiberoProAdapter
    from behavior_graphs.envs.mimicgen import MimicgenAdapter as MimicgenAdapter
    from behavior_graphs.envs.robomimic import RobomimicAdapter as RobomimicAdapter
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory as RolloutDirectory
    from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory as TrajectoryDirectory
    from behavior_graphs.envs.trajectory_dir import write_rollout_dir as write_rollout_dir
    from behavior_graphs.envs.trajectory_dir import write_trajectory_dir as write_trajectory_dir
    from behavior_graphs.envs.vla_foundry import VlaFoundryAdapter as VlaFoundryAdapter
