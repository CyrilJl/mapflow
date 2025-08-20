import matplotlib.pyplot as plt
import pytest
import xarray as xr

from mapflow import plot_da, plot_da_quiver


@pytest.fixture
def air_data() -> xr.DataArray:
    ds = xr.tutorial.open_dataset("air_temperature")
    return ds["air"].isel(time=slice(0, 48))


@pytest.fixture
def air_temperature_gradient_data() -> xr.Dataset:
    return xr.tutorial.load_dataset("air_temperature_gradient")


def test_plot_da(air_data):
    plot_da(da=air_data.isel(time=0), show=False)
    plt.close()


def test_plot_da_quiver(air_temperature_gradient_data):
    ds = air_temperature_gradient_data.isel(time=0)
    plot_da_quiver(ds["dTdx"], ds["dTdy"], show=False)
    plt.close()


def test_plot_da_quiver_subsample(air_temperature_gradient_data):
    ds = air_temperature_gradient_data.isel(time=0)
    plot_da_quiver(ds["dTdx"], ds["dTdy"], subsample=5, show=False)
    plt.close()


def test_plot_da_quiver_arrows_kwgs(air_temperature_gradient_data):
    ds = air_temperature_gradient_data.isel(time=0)
    plot_da_quiver(ds["dTdx"], ds["dTdy"], arrows_kwgs={"color": "blue"}, show=False)
    plt.close()
