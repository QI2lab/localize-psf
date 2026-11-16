"""Compatibility entry point for the packaged U-FISH localization workflow."""

from localize_psf.workflow import (
    LocalizationResult,
    app,
    cli,
    localize_ufish_prior,
    plot_calibration,
    save_localizations,
    save_trackpy,
    to_trackpy_dataframe,
)

__all__ = [
    "LocalizationResult",
    "app",
    "cli",
    "localize_ufish_prior",
    "plot_calibration",
    "save_localizations",
    "save_trackpy",
    "to_trackpy_dataframe",
]

if __name__ == "__main__":
    app()
