"""Robomimic environment adapter (diffusion-policy backed)."""

from __future__ import annotations

from behavior_graphs.envs.diffusion_policy_env import DiffusionPolicyEnvAdapter


class RobomimicAdapter(DiffusionPolicyEnvAdapter):
    """Robomimic tasks (lift, square, can, ...) trained via diffusion_policy.

    ``task`` selects a config dir inside the submodule, e.g.
    ``"low_dim/lift_mh/diffusion_policy_cnn"`` or
    ``"image/square_mh/diffusion_policy_cnn"``.
    """

    name = "robomimic"
