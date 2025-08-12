import matplotlib.pyplot as plt
import pytest
import xarray as xr

from mapflow import plot_da


@pytest.fixture
def air_data() -> xr.DataArray:
    ds = xr.tutorial.open_dataset("air_temperature")
    return ds["air"].isel(time=slice(0, 48))


def test_plot_da(air_data):
    plot_da(da=air_data.isel(time=0), show=False)
    plt.close()
