"""MimicGen environment adapter (diffusion-policy backed)."""

from __future__ import annotations

from behavior_graphs.envs.diffusion_policy_env import DiffusionPolicyEnvAdapter


class MimicgenAdapter(DiffusionPolicyEnvAdapter):
    """MimicGen-generated task variants (square_d0/d1, coffee, threading, ...).

    ``task`` selects a config dir inside the submodule, e.g.
    ``"low_dim/square_mimicgen_lowdim/diffusion_policy_cnn"``.
    """

    name = "mimicgen"
