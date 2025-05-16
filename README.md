# xrviz

[![Run Pytest](https://github.com/CyrilJl/xrviz/actions/workflows/pytest.yaml/badge.svg)](https://github.com/CyrilJl/xrviz/actions/workflows/pytest.yaml)

``xrviz`` transforms 3D ``xr.DataArray`` in video files in one code line. It relies on ``matplotlib`` and ``ffmpeg``.

## Usage

```python
import xarray as xr
from xrviz import animate

ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4')
```

https://github.com/user-attachments/assets/5e9c75a0-f0e0-4b0d-9420-292f8703d1c7
