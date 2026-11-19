"""Shared strict controls for existing CPU and native batch fitters."""

import numpy as np


def validate_fixed(fixed: np.ndarray | None, nparams: int) -> np.ndarray:
    """Return a shared boolean mask, rejecting malformed or nonbinary entries."""
    if fixed is None:
        return np.zeros(nparams, bool)
    mask = np.asarray(fixed)
    if (
        mask.shape != (nparams,)
        or mask.dtype.kind not in "biu"
        or not np.all((mask == 0) | (mask == 1))
    ):
        raise ValueError("fixed_params must be a binary mask of shape (nparams,)")
    return mask.astype(bool)


def validate_bounds(
    bounds: tuple | None, starts: np.ndarray, fixed: np.ndarray, dtype: type = np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast numeric bounds and validate feasibility before and after casting.

    Parameters
    ----------
    bounds : tuple or None
        Lower and upper arrays, independently shaped (nparams,) or (nfits,nparams).
    starts : ndarray
        Finite initial parameters, shaped (nfits,nparams).
    fixed : ndarray
        Shared boolean mask. Equal bounds require a fixed feasible parameter.
    dtype : type
        Optimizer precision; float32 conversion may not overflow or collapse bounds.

    Returns
    -------
    lower, upper, initial : ndarray
        Independent contiguous arrays in optimizer precision.
    """
    initial = np.asarray(starts, dtype=float)
    if initial.ndim != 2 or not np.all(np.isfinite(initial)):
        raise ValueError("initial parameters must be a finite two-dimensional array")
    if bounds is None:
        lower = np.full(initial.shape, -np.inf)
        upper = np.full(initial.shape, np.inf)
    else:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("bounds must be a numeric (lower, upper) pair")
        sides = []
        for side in bounds:
            values = np.asarray(side)
            if values.dtype.kind not in "fiu" or values.shape not in (
                (initial.shape[1],),
                initial.shape,
            ):
                raise ValueError(
                    "each numeric bound side must have shape (nparams,) or (nfits,nparams)"
                )
            sides.append(np.broadcast_to(values, initial.shape).astype(float, copy=True))
        lower, upper = sides
    _check_bounds(lower, upper, initial, fixed)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = [np.ascontiguousarray(a, dtype=dtype) for a in (lower, upper, initial)]
    for original, cast in zip((lower, upper, initial), converted):
        if np.any(np.isfinite(original) & ~np.isfinite(cast)):
            raise ValueError("float32 conversion overflow in fitting controls")
    if np.any((lower < upper) & (converted[0] == converted[1])):
        raise ValueError("float32 conversion collapsed distinct bounds")
    _check_bounds(*converted, fixed)
    return tuple(converted)


def _check_bounds(
    lower: np.ndarray, upper: np.ndarray, starts: np.ndarray, fixed: np.ndarray
) -> None:
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("bounds cannot contain NaNs")
    if np.any(np.isposinf(lower)) or np.any(np.isneginf(upper)):
        raise ValueError("illegal infinity sign in bounds")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    if np.any((lower == upper) & ~fixed):
        raise ValueError("equal bounds require fixed parameters")
    if np.any(starts < lower) or np.any(starts > upper):
        raise ValueError("initial parameters, including fixed values, must be feasible")
