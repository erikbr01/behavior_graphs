"""Sweep grid-value generators, instantiated via hydra ``_target_``.

A sweep spec gives each hyperparameter a list of values to try. Three forms
are equivalent, in increasing brevity → expressiveness:

- a scalar (``stride: 2``) — a fixed value, one-element grid;
- an explicit list (``width: [1, 3, 5, 8]``);
- a generator — a ``_target_`` config that *returns a list* when
  instantiated, e.g.::

      n_clusters: {_target_: behavior_graphs.pipeline.grid.irange, _args_: [4, 20, 2]}
      min_dist:   {_target_: behavior_graphs.pipeline.grid.linspace, _args_: [0.0, 0.5, 3]}
      lr:         {_target_: behavior_graphs.pipeline.grid.logspace, _args_: [1.0e-4, 1.0e-1, 4]}

Using ``_target_`` keeps grids on the same instantiation mechanism as every
pipeline stage (``hydra.utils.instantiate``) instead of a bespoke mini-DSL.
"""

from __future__ import annotations


def irange(start: int, stop: int, step: int = 1) -> list[int]:
    """Inclusive integer range: ``irange(4, 20, 2) == [4, 6, ..., 20]``."""
    if step <= 0:
        raise ValueError(f"irange step must be positive, got {step}")
    return list(range(int(start), int(stop) + 1, int(step)))


def linspace(start: float, stop: float, num: int) -> list[float]:
    """``num`` evenly spaced floats from ``start`` to ``stop`` (inclusive)."""
    import numpy as np

    return [float(f"{x:.10g}") for x in np.linspace(float(start), float(stop), int(num))]


def logspace(start: float, stop: float, num: int) -> list[float]:
    """``num`` geometrically spaced floats from ``start`` to ``stop`` (inclusive)."""
    import numpy as np

    return [float(f"{x:.10g}") for x in np.geomspace(float(start), float(stop), int(num))]
