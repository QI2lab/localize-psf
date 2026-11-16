"""Validated, versioned JSON settings shared by previews and the batch handoff."""

from __future__ import annotations

import shlex
import sys
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


class LocalizationConfig(BaseModel):
    """Batch metadata plus explicit detector, fitter, provenance, and preview settings."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    schema_version: Literal[1] = 1
    input_uri: str = ""
    source_layer: str = ""
    fit_model: Literal["2d", "3d"] = "3d"
    blend_3d: bool = True
    intensity_threshold: float = Field(default=0.2, ge=0)
    ufish_weights: str = "finetune_models/v1.0.1-simfish_model.pth"
    inference_batch_size: int = Field(default=32, ge=1)
    fit_roi_size_um: tuple[Positive, Positive, Positive] = (1.75, 0.65, 0.65)
    voxel_size_zyx_um: tuple[Positive, Positive, Positive] = (0.2, 0.1, 0.1)
    origin_zyx_um: tuple[float, float, float] = (0, 0, 0)
    estimator: Literal["MLE", "LSE"] = "MLE"
    max_number_iterations: int = Field(default=200, ge=1)
    tolerance: Positive = 1e-6
    roi_batch_size: int = Field(default=32, ge=1)
    return_covariance: bool = False
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

    def save_batch_file(
        self, path: str | Path, output_dir: str | Path, *, interpreter: str = sys.executable
    ) -> tuple[Path, str]:
        """Save current JSON and a runnable terminal file without starting a process.

        Parameters
        ----------
        path : str or pathlib.Path
            JSON destination; a sibling ``<stem>.batch.sh`` or ``.batch.ps1`` is written.
        output_dir : str or pathlib.Path
            Future result directory; it is not created by this method.
        interpreter : str, optional
            Python executable from the caller's installed uv CUDA environment.

        Returns
        -------
        script : pathlib.Path
            Absolute batch filename. POSIX scripts are executable by their owner.
        command : str
            Shell/PowerShell command with quoted paths, also suitable for a clipboard.

        Notes
        -----
        Relative paths resolve against the caller's current working directory.
        The command uses this configuration's input URI and never manages a subprocess.
        """
        if not self.input_uri:
            raise ValueError("A batch file requires an input URI")
        path = Path(path).resolve()
        self.save(path)
        source = self.input_uri if "://" in self.input_uri else str(Path(self.input_uri).resolve())
        arguments = [
            interpreter,
            "-m",
            "localize_psf.batch",
            source,
            "--metadata",
            str(path),
            "--output-dir",
            str(Path(output_dir).resolve()),
        ]
        windows = sys.platform == "win32"
        command = (
            "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
            if windows
            else shlex.join(arguments)
        )
        script = path.with_name(path.stem + ".batch" + (".ps1" if windows else ".sh"))
        script.write_text(
            (command + "\nexit $LASTEXITCODE\n")
            if windows
            else "#!/bin/sh\nexec " + command + "\n",
            encoding="utf-8-sig" if windows else "utf-8",
        )
        if not windows:
            script.chmod(script.stat().st_mode | 0o100)
        return script, command
