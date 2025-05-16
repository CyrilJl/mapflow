# xrviz

[![Run Pytest](https://github.com/CyrilJl/xrviz/actions/workflows/pytest.yaml/badge.svg)](https://github.com/CyrilJl/xrviz/actions/workflows/pytest.yaml)

``xrviz`` transforms 3D ``xr.DataArray`` in video files in one code line. It relies on ``matplotlib`` and ``ffmpeg``.

## Usage

```python
import xarray as xr
from xrviz import animate

ds = xr.tutorial.open_dataset("air_temperature")
animate(da=ds['air'].isel(time=slice(120)), path='animation.mp4')
```
