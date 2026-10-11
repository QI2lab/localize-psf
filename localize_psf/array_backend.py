"""Select NumPy or CuPy without importing CUDA for CPU-only consumers."""

import numpy as np


def get_array_module(*arrays):
    """Return CuPy for device arrays and NumPy for host arrays or scalars.

    Parameters
    ----------
    arrays : array-like
        Inputs whose existing memory location determines the backend.

    Returns
    -------
    module
        NumPy or CuPy; arrays are never transferred by this helper.
    """
    if any(hasattr(array, "__cuda_array_interface__") for array in arrays):
        import cupy

        return cupy
    return np
