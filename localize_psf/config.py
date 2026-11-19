"""Validated, versioned JSON settings shared by previews and the batch handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Positive = Annotated[float, Field(gt=0)]


class Parameter(BaseModel):
    """A bound/initial value in µm or counts; center values are prior-relative offsets.

    None selects the production estimate or ROI limit. A fixed parameter requires
    an explicit initial value. In 2D, cz and sz are fixed by the plane adapter.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    lower: float | None = None
    upper: float | None = None
    initial: float | None = None
    fixed: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Reject unusable bounds before dispatch to the production validator."""
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper or (self.lower == self.upper and not self.fixed):
                raise ValueError("Bounds must be ordered; equal limits require a fixed parameter")
        if self.fixed and self.initial is None:
            raise ValueError("A fixed parameter requires an initial value")
        if self.initial is not None:
            if self.lower is not None and self.initial < self.lower:
                raise ValueError("Initial value is below its lower bound")
            if self.upper is not None and self.initial > self.upper:
                raise ValueError("Initial value is above its upper bound")
        return self


class PreviewSettings(BaseModel):
    """Selection in the source layer, distinct from full-dataset batch scope."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    axes: str = "zyx"
    timepoint: int = Field(default=0, ge=0)
    channel: int = Field(default=0, ge=0)
    z_plane: int = Field(default=0, ge=0)
    crop_zyx_px: tuple[int, int, int, int, int, int] = (0, 41, 0, 128, 0, 128)

    @model_validator(mode="after")
    def validate_axes_and_crop(self) -> Self:
        """Require an explicit axis mapping and positive, half-open crop extents."""
        if not {"y", "x"} <= set(self.axes) <= set("tczyx") or len(set(self.axes)) != len(
            self.axes
        ):
            raise ValueError("Axes must be unique t/c/z/y/x labels including y and x")
        for start, stop in zip(self.crop_zyx_px[::2], self.crop_zyx_px[1::2], strict=True):
            if start < 0 or stop <= start:
                raise ValueError("Crop stops must exceed nonnegative starts")
        return self


class FitFilters(BaseModel):
    """Post-fit acceptance shared by previews and batch exports.

    Amplitude limits are inclusive, in original image counts; null is unbounded.
    Status selectors accept all fits or require the upstream pass/fail flag.
    Covariance that was not computed is unavailable, rather than a failed check.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    amplitude_min: float | None = Field(default=None, ge=0)
    amplitude_max: float | None = Field(default=None, ge=0)
    covariance: Literal["all", "pass", "fail"] = "all"
    convergence: Literal["all", "pass", "fail"] = "all"

    @model_validator(mode="after")
    def validate_amplitude_range(self) -> Self:
        """Require an ordered inclusive interval when both limits are supplied."""
        if self.amplitude_min is not None and self.amplitude_max is not None:
            if self.amplitude_min > self.amplitude_max:
                raise ValueError("Minimum amplitude must not exceed maximum amplitude")
        return self


# Published Spotiflow 0.6 model names, grouped by native spatial dimensions.
SPOTIFLOW_MODELS: dict[str, tuple[str, ...]] = {
    "2d": ("fluo_live", "general", "hybiss", "synth_complex"),
    "3d": ("smfish_3d", "synth_3d"),
}


class SpotiflowSettings(BaseModel):
    """Native prediction options; distances/tiles use input pixels.

    A null model selects fluo_live (2D) or smfish_3d (3D); custom models are
    trained model directories. Null probability/tile size use model defaults
    and automatic tiling. Subpixel offsets use stereographic flow.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    model: str | None = Field(default=None, min_length=1)
    prob_thresh: float | None = Field(default=None, ge=0, le=1)
    min_distance: int = Field(default=1, ge=1)
    subpix: bool = True
    max_tile_size: int | None = Field(default=None, ge=16)


class LocalizationConfig(BaseModel):
    """Batch metadata plus explicit detector, fitter, provenance, and preview settings."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    schema_version: Literal[1] = 1
    input_uri: str = ""
    source_layer: str = ""
    fit_model: Literal["2d", "3d"] = "3d"
    predictor: Literal["ufish", "spotiflow"] = "ufish"
    spotiflow: SpotiflowSettings = Field(default_factory=SpotiflowSettings)
    blend_3d: bool = True
    ufish_median_filter: bool = Field(
        default=True,
        description="Apply a size-3 spatial GPU median filter before prediction (ZYX in 3D, YX in 2D), independently per timepoint/channel. Fits use original counts.",
    )
    intensity_threshold: float = Field(default=0.1, ge=0)
    collapse_close_priors: bool = Field(
        default=True,
        description="Collapse both priors and final fitted centers within one nominal optical Gaussian FWHM axially and radially in XY, using the existing nearby-peak keep-one policy.",
    )
    exclude_edge_centroids: bool = Field(
        default=True,
        description="Exclude centroids near array faces and zero-padded data-footprint edges (YX in 2D).",
    )
    edge_margin_px: int = Field(default=1, ge=1, description="Excluded voxel layers at each face.")
    ufish_weights: str = "finetune_models/v1.0.1-suntag_model.pth"
    inference_batch_size: int = Field(default=32, ge=1)
    fit_roi_size_um: tuple[Positive, Positive, Positive] = (1.75, 0.65, 0.65)
    voxel_size_zyx_um: tuple[Positive, Positive, Positive] = (0.2, 0.1, 0.1)
    origin_zyx_um: tuple[float, float, float] = (0, 0, 0)
    estimator: Literal["MLE", "LSE"] = "MLE"
    max_number_iterations: int = Field(default=200, ge=1)
    tolerance: Positive = 1e-6
    roi_batch_size: int = Field(default=32, ge=1)
    return_covariance: bool = False
    fit_filters: FitFilters = Field(default_factory=FitFilters)
    numerical_aperture: Positive = 1.35
    wavelength_um: Positive = 0.515
    refractive_index: Positive = 1.4
    relative_sigma_range: tuple[Positive, Positive] = (0.5, 2)
    batch_axes: bool = True
    timepoint: int = Field(default=0, ge=0)
    channel: int = Field(default=0, ge=0)
    preview: PreviewSettings = Field(default_factory=PreviewSettings)
    fit_parameters: dict[str, Parameter] = Field(
        default_factory=lambda: {
            "amp": Parameter(lower=0),
            "cx": Parameter(),
            "cy": Parameter(),
            "cz": Parameter(),
            "sxy": Parameter(lower=0.03, upper=1, initial=0.15),
            "sz": Parameter(lower=0.05, upper=2, initial=0.35),
            "bg": Parameter(lower=0),
        }
    )
    provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> int:
        """JSON booleans and floats are not integer schema versions."""
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        """Keep the native model contract and physical parameter domains explicit."""
        if set(self.fit_parameters) != {"amp", "cx", "cy", "cz", "sxy", "sz", "bg"}:
            raise ValueError("Fit parameters must match the upstream Gaussian model")
        for name in ("amp", "bg", "sxy", "sz"):
            lower = self.fit_parameters[name].lower
            if lower is None or lower < 0 or (name in {"sxy", "sz"} and lower == 0):
                raise ValueError(f"{name} requires a physical lower bound")
        if not 0 < self.relative_sigma_range[0] <= 1 <= self.relative_sigma_range[1]:
            raise ValueError("Optical sigma factors must bracket one")
        if self.relative_sigma_range[0] >= self.relative_sigma_range[1]:
            raise ValueError("Optical sigma factors must be ordered")
        if self.numerical_aperture > self.refractive_index:
            raise ValueError("NA cannot exceed the immersion refractive index")
        if self.fit_filters.covariance != "all" and not self.return_covariance:
            raise ValueError("Covariance filtering requires return_covariance=True")
        return self

    def save(self, path: str | Path) -> None:
        """Write the current version 1 configuration as strict UTF-8 JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file. Optional bounds are null, never IEEE infinities.
        """
        Path(path).write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load strict version 1 JSON without coercing unsupported settings.

        Parameters
        ----------
        path : str or pathlib.Path
            Previously exported UTF-8 configuration.

        Returns
        -------
        LocalizationConfig
        Validated version 1 settings; invalid fields raise pydantic.ValidationError.
        """
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
