import subprocess
import warnings
import xarray as xr
from pyproj import CRS

X_NAME_CANDIDATES = ("x", "lon", "longitude")
Y_NAME_CANDIDATES = ("y", "lat", "latitude")
TIME_NAME_CANDIDATES = ("time", "t", "times")


def check_ffmpeg():
    """Checks if ffmpeg is available on the system and outputs a warning if not."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn("ffmpeg is not found. Some functionalities might be limited.")


def guess_coord_name(da_coords, candidates, provided_name, coord_type_for_error) -> str:
    """Guesses the coordinate name if not provided.

    Iterates through da_coords, compares lowercased names with candidates.

    Args:
        da_coords (list): The coordinates from a DataArray.
        candidates (tuple): Plausible names for the coordinate.
        provided_name (str or None): The name given by the user, if any.
        coord_type_for_error (str): The type of coordinate being searched for, used for error messages.

    Returns:
        str: The identified coordinate name.

    Raises:
        ValueError: If the coordinate name cannot be guessed.
    """
    if provided_name is not None:
        return provided_name

    for coord_name_key in da_coords:
        # Convert coord_name_key to string before lower() in case it's not already a string
        coord_name_str = str(coord_name_key).lower()
        if coord_name_str in candidates:
            return str(coord_name_key)  # Return original case name

    raise ValueError(
        f"Could not automatically detect {coord_type_for_error}-coordinate. "
        f"Please specify '{coord_type_for_error}_name' from available coordinates: {list(da_coords.keys())}. "
        f"Tried to guess from candidates: {candidates}."
    )


def process_crs(da, crs):
    """Processes the Coordinate Reference System (CRS).

    If a CRS is not provided, it attempts to extract it from the DataArray's
    `spatial_ref` coordinate. Defaults to EPSG:4326 if not found.

    Args:
        da (xr.DataArray): The input DataArray.
        crs (int | str | CRS | None): The user-provided CRS.

    Returns:
        CRS: A pyproj.CRS object.
    """
    if crs is None:
        if "spatial_ref" in da.coords:
            crs = da.spatial_ref.attrs.get("crs_wkt", 4326)
        else:
            crs = 4326
    return CRS.from_user_input(crs)


def check_da(da, time_name, x_name, y_name, crs):
    """Validates and preprocesses the input DataArray.

    This function performs several checks and modifications:
    - Ensures the input is an xarray.DataArray.
    - Verifies that the specified time, x, and y coordinates exist.
    - Processes the CRS.
    - Wraps longitudes to the -180 to 180 range for geographic CRS.
    - Sorts the DataArray by its coordinates.
    - Ensures the DataArray is 3-dimensional and transposes it to (`time`, `y`, `x`).

    Args:
        da (xr.DataArray): The DataArray to check.
        time_name (str): The name of the time coordinate.
        x_name (str): The name of the x-coordinate.
        y_name (str): The name of the y-coordinate.
        crs (int | str | CRS | None): The user-provided CRS.

    Returns:
        tuple[xr.DataArray, CRS]: A tuple containing the processed DataArray and the CRS object.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray, got {type(da)}")
    for dim in (x_name, y_name, time_name):
        if dim not in da.coords:
            raise ValueError(f"Dimension '{dim}' not found in DataArray coordinates: {da.dims}")
    crs_ = process_crs(da, crs)
    if crs_.is_geographic:
        da[x_name] = xr.where(da[x_name] > 180, da[x_name] - 360, da[x_name])

    # For non-rectilinear grids (2D coordinates), sorting by spatial dimensions is not possible.
    if da[x_name].ndim == 1 and da[y_name].ndim == 1:
        da = da.sortby(x_name).sortby(y_name)

    da = da.sortby(time_name).squeeze()

    if da.ndim != 3:
        raise ValueError(
            f"DataArray must have 3 dimensions ({time_name}, {y_name}, {x_name}), got {da.ndim} dimensions."
        )

    # Ensure time is the first dimension
    if da[x_name].ndim == 1 and da[y_name].ndim == 1:
        da = da.transpose(time_name, y_name, x_name)
    elif list(da.dims)[0] != time_name:
        current_dims = list(da.dims)
        current_dims.remove(time_name)
        new_order = [time_name] + current_dims
        da = da.transpose(*new_order)
    return da, crs_
