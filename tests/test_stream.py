import matplotlib.pyplot as plt
import pytest
import xarray as xr

from mapflow import plot_da_stream, animate_stream


@pytest.fixture
def air_temperature_gradient_data() -> xr.Dataset:
    return xr.tutorial.load_dataset("air_temperature_gradient")


def test_plot_da_stream(air_temperature_gradient_data):
    u = air_temperature_gradient_data["dTdx"].isel(time=0)
    v = air_temperature_gradient_data["dTdx"].isel(time=0)
    plot_da_stream(u, v, show=False)
    plt.close()


def test_animate_stream(air_temperature_gradient_data):
    u = air_temperature_gradient_data["dTdx"].isel(time=slice(2))
    v = air_temperature_gradient_data["dTdx"].isel(time=slice(2))
    animate_stream(u, v, path="test_animation.mp4")
