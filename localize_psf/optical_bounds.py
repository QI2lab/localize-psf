"""Optics-derived Gaussian width priors using the existing PSF convention."""

import numpy as np

from localize_psf.fit_psf import na2sxy


def gaussian_width_bounds(
    numerical_aperture: float,
    wavelength_um: float,
    refractive_index: float,
    spacing_zyx_um,
    *,
    relative_sigma_range: tuple[float, float],
    sampling_floor_pixels: float,
    use_gpu: bool = True,
) -> dict:
    """Derive optical sigmas and explicit sampling-aware bounds on CUDA.

    Lateral sigma reuses ``na2sxy`` (Airy-to-Gaussian convention). Axial sigma
    uses sqrt(6)/pi * n * wavelength / NA**2, as documented by the existing
    Gaussian PSF model. Wavelength and spacings are physical micrometres.
    The bound policy is [max(low_factor*sigma, floor_pixels*spacing),
    high_factor*sigma]. Factors must bracket the nominal width. The sampling
    floor is an explicit fitting policy, not an inferred pixel blur or measured
    resolution. Unsupported sampling fails rather than widening the bounds.

    These are ideal optical priors, including for deconvolved data; they do not
    calibrate an OPM PSF, aberrations, interpolation or localization covariance.
    ``use_gpu=False`` is reserved for independent CPU references.
    """
    if use_gpu:
        try:
            import cupy as xp
        except ImportError as error:
            raise RuntimeError("optical width bounds require CuPy/CUDA") from error
    else:
        xp = np
    inputs = xp.asarray(
        [numerical_aperture, wavelength_um, refractive_index, sampling_floor_pixels],
        dtype=xp.float64,
    )
    if inputs.shape != (4,) or not bool(xp.all(xp.isfinite(inputs) & (inputs > 0))):
        raise ValueError("optical parameters and sampling floor must be finite and positive")
    na, wavelength, index, floor = inputs
    if bool(na > index):
        raise ValueError("numerical aperture cannot exceed the immersion refractive index")
    spacing = xp.asarray(spacing_zyx_um, dtype=xp.float64)
    if spacing.shape != (3,) or not bool(xp.all(xp.isfinite(spacing) & (spacing > 0))):
        raise ValueError("stored ZYX spacings must be finite positive micrometres")
    factors = xp.asarray(relative_sigma_range, dtype=xp.float64)
    if (
        factors.shape != (2,)
        or not bool(xp.all(xp.isfinite(factors)))
        or not bool(
            (0 < factors[0]) & (factors[0] <= 1) & (factors[1] >= 1) & (factors[0] < factors[1])
        )
    ):
        raise ValueError("relative sigma range must be positive, ordered and include one")
    lateral = na2sxy(na, wavelength)
    axial = xp.sqrt(xp.asarray(6.0)) / xp.pi * index * wavelength / na**2
    nominal = xp.stack((axial, lateral, lateral))
    lower = xp.maximum(factors[0] * nominal, floor * spacing)
    upper = factors[1] * nominal
    if not bool(xp.all(xp.isfinite(nominal) & xp.isfinite(upper) & (nominal > 0))):
        raise ValueError("optical width calculation overflowed or underflowed")
    if bool(xp.any(lower > nominal)):
        raise ValueError("stored sampling floor exceeds the nominal optical width")
    return {
        "nominal_sigma_zyx_um": nominal,
        "lower_sigma_zyx_um": lower,
        "upper_sigma_zyx_um": upper,
    }
