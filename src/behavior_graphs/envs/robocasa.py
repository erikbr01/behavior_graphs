"""RoboCasa environment adapter (diffusion-policy backed)."""

from __future__ import annotations

from behavior_graphs.envs.diffusion_policy_env import DiffusionPolicyEnvAdapter


class RobocasaAdapter(DiffusionPolicyEnvAdapter):
    """RoboCasa kitchen tasks trained via diffusion_policy.

    ``task`` selects a config dir inside the submodule, e.g.
    ``"low_dim/robocasa_layout_lowdim/diffusion_policy_cnn"`` or
    ``"image/robocasa_lerobot_atomic/diffusion_policy_cnn"``.

    RoboCasa needs its own simulator stack; point ``python_path`` at an
    environment with robocasa installed if the submodule's default venv
    does not include it.
    """

    name = "robocasa"

    # The robocasa runner builds its env from env_name + recorded env_kwargs;
    # only the dataset nodes take a path.
    dataset_override_keys = ("task.dataset.dataset_path", "task.dataset_path")
