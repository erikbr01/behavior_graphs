"""Shared feature normalization for the dimensionality-reduction stages."""

from __future__ import annotations

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def make_normalizer(method: str) -> TransformerMixin | None:
    """Return an unfitted sklearn scaler for ``method``, or ``None``.

    Args:
        method: ``"none"`` (no scaling), ``"standard"``, ``"minmax"``, or
            ``"robust"``.

    Returns:
        A fresh scaler instance to be fit by the caller, or ``None`` when
        ``method == "none"``.
    """
    if method == "none":
        return None
    if method not in _SCALERS:
        raise ValueError(
            f"normalize must be one of ['none', {', '.join(map(repr, _SCALERS))}], got {method!r}"
        )
    return _SCALERS[method]()
