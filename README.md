# localize-psf

Tools for localizing fluorescent objects, fitting point-spread functions, and
related image analysis. The package provides shared tools for
[mcSIM](https://github.com/QI2lab/mcSIM) and [OPM](https://github.com/QI2lab/OPM).

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python ≥3.12. In an active Python
environment, clone the repository and install:

```bash
git clone https://github.com/QI2lab/localize-psf.git
cd localize-psf
uv pip install .
```

For GPU fitting and U-FISH prediction, install the optional dependencies:

```bash
uv pip install ".[gpu,example]"
```

GPU use requires an NVIDIA GPU and driver on Linux x86-64 or Windows x64.
On Windows, use an x64 Visual Studio Developer terminal for installation.

## Usage

Run batch localization on an OME-Zarr image group or OME-TIFF using a JSON
configuration:

```bash
python -m localize_psf.batch image.ome.zarr/0 --metadata localization.json --output-dir results
```

Start with [localization.json](examples/localization.json), or configure settings
interactively with [napari-localize-psf](https://github.com/QI2lab/napari-localize-psf).
Localizations are saved as Parquet files in the output directory.

Examples of spot localization, fitting, and PSF calibration are available in
[examples](examples).
