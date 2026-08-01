<div align="center">
<img src="https://raw.githubusercontent.com/CyrilJl/mapflow/main/_static/logo.svg" alt="mapflow logo" width="200" height="200">

# mapflow

[![PyPI version](https://badge.fury.io/py/mapflow.svg)](https://pypi.org/project/mapflow/)
[![Conda version](https://anaconda.org/conda-forge/mapflow/badges/version.svg)](https://anaconda.org/conda-forge/mapflow)
[![CI](https://github.com/CyrilJl/mapflow/actions/workflows/CI.yaml/badge.svg)](https://github.com/CyrilJl/mapflow/actions/workflows/CI.yaml)
[![Documentation Status](https://readthedocs.org/projects/mapflow/badge/?version=latest)](https://mapflow.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
</div>

`mapflow` creates geographic plots and video animations directly from
[`xarray.DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html) objects. It detects common
coordinate names, understands CRS metadata, includes world borders, and streams lazily backed animation frames to
keep memory use bounded.

## Installation

Install the Python package from PyPI:

```bash
python -m pip install mapflow
```

Or from conda-forge, which also installs FFmpeg:

```bash
conda install -c conda-forge mapflow
```

Creating animations requires the `ffmpeg` executable on `PATH`. Static plots do not require FFmpeg.

## Quick start

```python
import xarray as xr

from mapflow import animate, plot_da

ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
temperature = ds["t2m"]

plot_da(temperature.isel(time=0))
animate(temperature.isel(time=slice(120)), "temperature.mp4", video_width=1280)
```

<img src="https://raw.githubusercontent.com/CyrilJl/mapflow/main/_static/plot_da.png" alt="Example mapflow plot" width="500">

The [documentation](https://mapflow.readthedocs.io) covers static plots, scalar animations, vector-field quiver
plots, CRS handling, color normalization, and the reusable `PlotModel` and `Animation` classes.

## Development

```bash
git clone https://github.com/CyrilJl/mapflow.git
cd mapflow
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution and release checks. Changes are documented in
[CHANGELOG.md](CHANGELOG.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
