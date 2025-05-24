<div align="center">
<img src="_static/logo.svg" alt="mapflow logo" width="200" height="200">

# mapflow

[![PyPI version](https://badge.fury.io/py/mapflow.svg)](https://badge.fury.io/py/mapflow)
[![Conda version](https://anaconda.org/conda-forge/mapflow/badges/version.svg)](https://anaconda.org/conda-forge/mapflow)
[![Run Pytest](https://github.com/CyrilJl/mapflow/actions/workflows/pytest.yaml/badge.svg)](https://github.com/CyrilJl/mapflow/actions/workflows/pytest.yaml)
</div>

``mapflow`` transforms 3D ``xr.DataArray`` in video files in one code line. It relies on ``matplotlib`` and ``ffmpeg``. Make sure ``ffmpeg`` is installed on your system.

## Installation

```bash
pip install mapflow
```

Or:

```bash
conda install -c conda-forge -y mapflow
```

## Simple usage

```python
import xarray as xr
from mapflow import animate

ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4')
```

https://github.com/user-attachments/assets/a6e136eb-5c35-4236-9472-0845a4804740

## Quick documentation

``animate`` is the core function of ``mapflow``.
It creates an animation from an xarray DataArray. This function prepares data from an xarray DataArray (e.g., handling geographic coordinates, extracting time information for titles) and then uses the `Animation` class to generate and save the animation.

### `animate` function arguments

* **`da`** (`xr.DataArray`):
    Input DataArray with at least time, x, and y dimensions.

* **`path`** (`str`):
    Output path for the video file.

* **`time_name`** (`str`, optional):
    Name of the time coordinate in `da`. Defaults to `"time"`.

* **`x_name`** (`str`, optional):
    Name of the x-coordinate (e.g., longitude) in `da`. Defaults to `"longitude"`.

* **`y_name`** (`str`, optional):
    Name of the y-coordinate (e.g., latitude) in `da`. Defaults to `"latitude"`.

* **`crs`** (`int | str | pyproj.CRS`, optional):
    Coordinate Reference System of the data. Defaults to `4326` (WGS84).

    ```python
    # Examples
    crs_epsg_code = 4326 
    crs_proj_string = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
    # from pyproj import CRS
    # crs_object = CRS.from_epsg(3035)
    ```

* **`borders`** (`gpd.GeoDataFrame | gpd.GeoSeries | None`, optional):
    Custom borders to use for plotting. If `None`, defaults to world borders. Defaults to `None`.

    ```python
    import geopandas as gpd
    # Example: Load a shapefile for custom borders
    custom_borders = gpd.read_file("path/to/your/custom_borders.shp")
    ```

* **`cmap`** (`str`, optional):
    Colormap for the plot. Defaults to `"jet"`.
    Refer to Matplotlib documentation for available colormaps: <https://matplotlib.org/stable/gallery/color/colormap_reference.html>

* **`norm`** (`matplotlib.colors.Normalize`, optional):
    Custom normalization object. Defaults to `None`.

    ```python
    import matplotlib.colors as mcolors
    # Example: Using a custom normalization
    custom_norm = mcolors.PowerNorm(gamma=0.5)
    ```

* **`log`** (`bool`, optional):
    Use logarithmic color scale. Defaults to `False`.

* **`qmin`** (`float`, optional):
    Minimum quantile for color normalization. Defaults to `0.01`. Used if `vmin` is not set.

* **`qmax`** (`float`, optional):
    Maximum quantile for color normalization. Defaults to `99.9`. Used if `vmax` is not set.

* **`vmin`** (`float`, optional):
    Minimum value for color normalization. Overrides `qmin`. Defaults to `None`.

* **`vmax`** (`float`, optional):
    Maximum value for color normalization. Overrides `qmax`. Defaults to `None`.

* **`time_format`** (`str`, optional):
    Strftime format for time in titles. Defaults to `"%Y-%m-%dT%H"`.
    Refer to Python's `strftime` documentation for formatting options: <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>

    ```python
    # Example: Display only Year-Month-Day
    # time_format_example = "%Y-%m-%d"
    # Example: Display Hour:Minute AM/PM
    # time_format_example = "%I:%M %p" 
    ```

* **`upsample_ratio`** (`int`, optional):
    Factor to upsample data temporally for smoother animations. Defaults to `4`. A value of `1` means no upsampling.

* **`fps`** (`int`, optional):
    Frames per second for the video. Defaults to `24`.

* **`n_jobs`** (`int`, optional):
    Number of parallel jobs for frame generation. Defaults to `None` (auto-determined, typically 2/3 of CPU cores). Set to `1` for no parallel processing.

* **`verbose`** (`int`, optional):
    Verbosity level for the Animation class. If `> 0`, progress bars will be shown. Defaults to `0`.

## Future developments

* More customization options
* Better RAM management (generating upsampled frames on the fly instead of computing them all at once)
* Documentation
