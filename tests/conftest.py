import matplotlib
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

matplotlib.use("Agg")


@pytest.fixture
def air_data() -> xr.DataArray:
    """Small deterministic temperature field used without network access."""
    rng = np.random.default_rng(42)
    time = np.arange("2020-01-01", "2020-01-09", dtype="datetime64[D]")
    lat = np.linspace(20, 60, 12)
    lon = np.linspace(-20, 20, 16)
    data = 275 + 8 * rng.random((time.size, lat.size, lon.size))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="air",
        attrs={"units": "K"},
    )


@pytest.fixture
def air_temperature_gradient_data(air_data: xr.DataArray) -> xr.Dataset:
    """Vector field with rioxarray-compatible CRS metadata."""
    dtdy, dtdx = np.gradient(air_data.values, axis=(1, 2))
    dataset = xr.Dataset(
        {
            "dTdx": (air_data.dims, dtdx),
            "dTdy": (air_data.dims, dtdy),
        },
        coords=air_data.coords,
    )
    spatial_ref = xr.DataArray(0, attrs={"crs_wkt": CRS.from_epsg(4326).to_wkt()})
    return dataset.assign_coords(spatial_ref=spatial_ref)
