# mapflow

[![Run Pytest](https://github.com/CyrilJl/mapflow/actions/workflows/pytest.yaml/badge.svg)](https://github.com/CyrilJl/mapflow/actions/workflows/pytest.yaml)

``mapflow`` transforms 3D ``xr.DataArray`` in video files in one code line. It relies on ``matplotlib`` and ``ffmpeg``.

## Usage

```python
import xarray as xr
from mapflow import animate

ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4')
```
https://raw.githubusercontent.com/CyrilJl/mapflow/main/_static/animation.mp4
