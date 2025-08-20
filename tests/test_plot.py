import matplotlib.pyplot as plt
import pytest
import xarray as xr

from mapflow import plot_da, plot_da_quiver


@pytest.fixture
def air_data() -> xr.DataArray:
    ds = xr.tutorial.open_dataset("air_temperature")
    return ds["air"].isel(time=slice(0, 48))


def test_plot_da(air_data):
    plot_da(da=air_data.isel(time=0), show=False)
    plt.close()


def test_plot_da_quiver(air_data):
    u = air_data.isel(time=0)
    v = air_data.isel(time=0)
    plot_da_quiver(u, v, show=False)
    plt.close()
    plot_da_quiver(u, v, subsample=5, show=False)
    plt.close()
