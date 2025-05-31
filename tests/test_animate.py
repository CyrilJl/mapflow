import os
from tempfile import TemporaryDirectory

import geopandas as gpd
import pytest
import xarray as xr
from shapely.geometry import box

from mapflow import animate


@pytest.fixture
def air_data():
    ds = xr.tutorial.open_dataset("air_temperature")
    return ds["air"].isel(time=slice(0, 48))


def test_animate(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_log(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_log.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            log=True,
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_vmin_vmax(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_vmin_vmax.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            vmin=250,  # Example value
            vmax=300,  # Example value
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_qmin_qmax(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_qmin_qmax.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            qmin=10,  # Example value
            qmax=90,  # Example value
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_cmap(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_cmap.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            cmap="viridis",  # Example colormap
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_upsample_ratio(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_upsample_ratio.mp4"
        # Use fewer frames for upsampling test to keep it fast
        animate(
            air_data.isel(time=slice(0, 10)),
            path=path,
            x_name="lon",
            y_name="lat",
            upsample_ratio=10,  # Example ratio
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_fps(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_fps.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            fps=10,  # Example fps
            verbose=True,
        )
        assert os.path.exists(path)


def test_animate_borders(air_data):
    borders = gpd.GeoSeries([box(-2, 42, 8, 50)], crs=4326)
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation_fps.mp4"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            borders=borders,
            verbose=True,
        )
        assert os.path.exists(path)

def test_animate_mkv(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation.mkv"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            verbose=True,
        )
        assert os.path.exists(path)

def test_animate_mov(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation.mov"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            verbose=True,
        )
        assert os.path.exists(path)

def test_animate_avi(air_data):
    with TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_animation.avi"
        animate(
            da=air_data,
            path=path,
            x_name="lon",
            y_name="lat",
            verbose=True,
        )
        assert os.path.exists(path)
