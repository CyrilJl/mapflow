from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from ._classic import Animation, PlotModel
from ._misc import (
    TIME_NAME_CANDIDATES,
    X_NAME_CANDIDATES,
    Y_NAME_CANDIDATES,
    check_da,
    guess_coord_name,
    process_crs,
)


def plot_da_quiver(
    u,
    v,
    x_name=None,
    y_name=None,
    crs=4326,
    subsample: int = 1,
    show=True,
    arrows_kwgs: dict = None,
    **kwargs,
):
    """
    Plots a quiver plot from two xarray DataArrays, representing the U and V
    components of a vector field.

    The magnitude of the vector field is represented by a color mesh, and the
    direction is shown with quiver arrows.

    Args:
        u (xr.DataArray): DataArray for the U-component of the vector field.
        v (xr.DataArray): DataArray for the V-component of the vector field.
        x_name (str, optional): Name of the x-coordinate dimension.
            If None, will attempt to guess.
        y_name (str, optional): Name of the y-coordinate dimension.
            If None, will attempt to guess.
        crs: Coordinate Reference System. Can be:
            - EPSG code (e.g., 4326 for WGS84)
            - PROJ string
            - pyproj.CRS object
            - If the DataArray has a 'crs' attribute, that will be used by default
        subsample (int, optional): The subsampling factor for the quiver arrows.
            For example, a value of 10 will plot one arrow for every 10 grid points.
            Defaults to 1.
        show: Whether to display the plot
        arrows_kwgs (dict, optional): Additional keyword arguments passed to
            `matplotlib.pyplot.quiver`. Defaults to None.
        **kwargs: Additional arguments passed to PlotModel.__call__(), including:
            - figsize: Tuple (width, height) in inches
            - qmin/qmax: Quantile ranges for color scaling (0-100)
            - vmin/vmax: Explicit value ranges for color scaling
            - log: Whether to use logarithmic color scale
            - cmap: Colormap name
            - norm: Custom normalization
            - shading: Color shading method
            - shrink: Colorbar shrink factor
            - label: Colorbar label
            - title: Plot title

    Example:
        .. code-block:: python

            import xarray as xr
            from mapflow import plot_da_quiver

            ds = xr.tutorial.load_dataset("air_temperature_gradient").isel(time=0)
            plot_da_quiver(u=ds["dTdx"], v=ds["dTdy"], subsample=4)

    See Also:
        PlotModel: The underlying plotting class used by this function.
    """
    actual_x_name = guess_coord_name(u.coords, X_NAME_CANDIDATES, x_name, "x")
    actual_y_name = guess_coord_name(u.coords, Y_NAME_CANDIDATES, y_name, "y")

    if u[actual_x_name].ndim == 1 and u[actual_y_name].ndim == 1:
        u = u.sortby(actual_x_name).sortby(actual_y_name)
        v = v.sortby(actual_x_name).sortby(actual_y_name)

    crs_ = process_crs(u, crs)
    if crs_.is_geographic:
        u[actual_x_name] = xr.where(u[actual_x_name] > 180, u[actual_x_name] - 360, u[actual_x_name])
        v[actual_x_name] = xr.where(v[actual_x_name] > 180, v[actual_x_name] - 360, v[actual_x_name])

    magnitude = np.sqrt(u**2 + v**2)
    p = PlotModel(x=u[actual_x_name].values, y=u[actual_y_name].values, crs=crs_)
    data = p._process_data(magnitude.values)
    p(data, show=False, **kwargs)

    if subsample > 1:
        u_subsampled = u.isel(
            {actual_y_name: slice(None, None, subsample), actual_x_name: slice(None, None, subsample)}
        )
        v_subsampled = v.isel(
            {actual_y_name: slice(None, None, subsample), actual_x_name: slice(None, None, subsample)}
        )
        x = u_subsampled[actual_x_name].values
        y = u_subsampled[actual_y_name].values
        u_subsampled = u_subsampled.values
        v_subsampled = v_subsampled.values
    else:
        x = u[actual_x_name].values
        y = u[actual_y_name].values
        u_subsampled = u.values
        v_subsampled = v.values

    if u[actual_x_name].ndim == 1:
        x, y = np.meshgrid(x, y)

    if arrows_kwgs is None:
        arrows_kwgs = {}
    plt.quiver(x, y, u_subsampled, v_subsampled, **arrows_kwgs)
    if show:
        plt.show()


class QuiverAnimation(Animation):
    def quiver(
        self,
        u,
        v,
        path,
        subsample: int = 1,
        figsize: tuple = None,
        title=None,
        fps: int = 24,
        upsample_ratio: int = 2,
        cmap="jet",
        qmin=0.01,
        qmax=99.9,
        vmin=None,
        vmax=None,
        norm=None,
        log=False,
        label=None,
        dpi=180,
        n_jobs=None,
        timeout="auto",
        **kwargs,
    ):
        magnitude = np.sqrt(u**2 + v**2)
        norm = self.plot._norm(
            magnitude.values,
            vmin=vmin,
            vmax=vmax,
            qmin=qmin,
            qmax=qmax,
            norm=norm,
            log=log,
        )
        self._animate(
            data=(u, v),
            path=path,
            frame_generator=self._generate_quiver_frame,
            figsize=figsize,
            title=title,
            fps=fps,
            upsample_ratio=upsample_ratio,
            cmap=cmap,
            norm=norm,
            label=label,
            dpi=dpi,
            n_jobs=n_jobs,
            timeout=timeout,
            subsample=subsample,
            **kwargs,
        )

    def _animate(
        self,
        data,
        path,
        frame_generator,
        figsize: tuple = None,
        title=None,
        fps: int = 24,
        upsample_ratio: int = 2,
        cmap="jet",
        norm=None,
        label=None,
        dpi=180,
        n_jobs=None,
        timeout="auto",
        **kwargs,
    ):
        titles = self._process_title(title, upsample_ratio)

        u, v = data
        u_upsampled = self.upsample(u.values, ratio=upsample_ratio)
        v_upsampled = self.upsample(v.values, ratio=upsample_ratio)
        data = (u_upsampled, v_upsampled)
        data_len = len(u_upsampled)

        with TemporaryDirectory() as tempdir:
            frame_paths = [Path(tempdir) / f"frame_{k:08d}.png" for k in range(data_len)]
            args = []
            for k in range(data_len):
                frame_data = (data[0][k], data[1][k])
                arg_tuple = (
                    frame_data,
                    frame_paths[k],
                    figsize,
                    titles[k] if titles and k < len(titles) else None,
                    cmap,
                    norm,
                    label,
                    dpi,
                    kwargs,
                )
                args.append(arg_tuple)

            n_jobs = int(2 / 3 * cpu_count()) if n_jobs is None else n_jobs
            with Pool(processes=n_jobs) as pool:
                list(
                    tqdm(
                        pool.imap(frame_generator, args),
                        total=data_len,
                        disable=(not self.verbose),
                        desc="Frames generation",
                        leave=False,
                    )
                )

            timeout = max(20, 0.1 * data_len) if timeout == "auto" else timeout
            self._create_video(tempdir, path, fps, timeout=timeout)

    def _generate_quiver_frame(self, args):
        """Generates a quiver frame and saves it as a PNG."""
        (u_frame, v_frame), frame_path, figsize, title, cmap, norm, label, dpi, kwargs = args
        x_name = kwargs.get("x_name")
        y_name = kwargs.get("y_name")
        coords = {y_name: self.plot.y, x_name: self.plot.x}
        dims = (y_name, x_name)
        u_da = xr.DataArray(u_frame, coords=coords, dims=dims)
        v_da = xr.DataArray(v_frame, coords=coords, dims=dims)
        magnitude = np.sqrt(u_frame**2 + v_frame**2)
        self.plot(
            data=magnitude,
            figsize=figsize,
            title=title,
            show=False,
            cmap=cmap,
            norm=norm,
            label=label,
        )
        subsample = kwargs.get("subsample", 1)
        if subsample > 1:
            u_sub = u_da.isel({y_name: slice(None, None, subsample), x_name: slice(None, None, subsample)})
            v_sub = v_da.isel({y_name: slice(None, None, subsample), x_name: slice(None, None, subsample)})
            x = u_sub[x_name].values
            y = u_sub[y_name].values
            u_sub = u_sub.values
            v_sub = v_sub.values
        else:
            x = u_da[x_name].values
            y = u_da[y_name].values
            u_sub = u_da.values
            v_sub = v_da.values

        if u_da[x_name].ndim == 1:
            x, y = np.meshgrid(x, y)

        arrows_kwgs = kwargs.get("arrows_kwgs")
        if arrows_kwgs is None:
            arrows_kwgs = {}
        plt.quiver(x, y, u_sub, v_sub, **arrows_kwgs)
        plt.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.clf()
        plt.close()


def animate_quiver(
    u: xr.DataArray,
    v: xr.DataArray,
    path: str,
    time_name: str = None,
    x_name: str = None,
    y_name: str = None,
    crs=None,
    field_name: str = None,
    borders: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    verbose: int = 0,
    subsample: int = 1,
    arrows_kwgs: dict = None,
    **kwargs,
):
    """
    Creates a quiver animation from two xarray DataArrays.

    Args:
        u (xr.DataArray): Input DataArray for the U-component with at least time, x, and y dimensions.
        v (xr.DataArray): Input DataArray for the V-component with at least time, x, and y dimensions.
        path (str): Output path for the video file. Supported formats are avi, mov and mp4.
        time_name (str, optional): Name of the time coordinate in `da`. If None,
            it's guessed from ['time', 't', 'times']. Defaults to None.
        x_name (str, optional): Name of the x-coordinate (e.g., longitude) in `da`.
            If None, it's guessed from ['x', 'lon', 'longitude']. Defaults to None.
        y_name (str, optional): Name of the y-coordinate (e.g., latitude) in `da`.
            If None, it's guessed from ['y', 'lat', 'latitude']. Defaults to None.
        crs (int | str | CRS, optional): Coordinate Reference System of the data.
            Defaults to 4326 (WGS84).
        borders (gpd.GeoDataFrame | gpd.GeoSeries | None, optional):
            Custom borders to use for plotting. If None, defaults to
            world borders. Defaults to None.
        verbose (int, optional): Verbosity level for the Animation class.
            Defaults to 0.
        subsample (int, optional): The subsampling factor for the quiver arrows.
            For example, a value of 10 will plot one arrow for every 10 grid points.
            Defaults to 1.
        arrows_kwgs (dict, optional): Additional keyword arguments passed to
            `matplotlib.pyplot.quiver`. Defaults to None.
        **kwargs: Additional keyword arguments passed to the Animation class, including:
            - cmap (str): Colormap for the plot. Defaults to "jet".
            - norm (matplotlib.colors.Normalize): Custom normalization object.
            - log (bool): Use logarithmic color scale. Defaults to False.
            - qmin (float): Minimum quantile for color normalization. Defaults to 0.01.
            - qmax (float): Maximum quantile for color normalization. Defaults to 99.9.
            - vmin (float): Minimum value for color normalization. Overrides qmin.
            - vmax (float): Maximum value for color normalization. Overrides qmax.
            - time_format (str): Strftime format for time in titles. Defaults to "%Y-%m-%dT%H".
            - upsample_ratio (int): Factor to upsample data temporally. Defaults to 4.
            - fps (int): Frames per second for the video. Defaults to 24.
            - n_jobs (int): Number of parallel jobs for frame generation.
            - dpi (int): Dots per inch for the saved frames. Defaults to 180.
            - timeout (str | int): Timeout for video creation. Defaults to 'auto'.

    Example:
        .. code-block:: python

            import xarray as xr
            from mapflow import animate_quiver

            ds = xr.tutorial.load_dataset("air_temperature_gradient")
            animate_quiver(u=ds["dTdx"], v=ds["dTdy"], path='animation.mkv', subsample=3)
    """
    actual_time_name = guess_coord_name(u.coords, TIME_NAME_CANDIDATES, time_name, "time")
    actual_x_name = guess_coord_name(u.coords, X_NAME_CANDIDATES, x_name, "x")
    actual_y_name = guess_coord_name(u.coords, Y_NAME_CANDIDATES, y_name, "y")

    u, crs_ = check_da(u, actual_time_name, actual_x_name, actual_y_name, crs)
    v, _ = check_da(v, actual_time_name, actual_x_name, actual_y_name, crs)

    animation = QuiverAnimation(
        x=u[actual_x_name].values,
        y=u[actual_y_name].values,
        crs=crs_,
        verbose=verbose,
        borders=borders,
    )
    output_path = Path(path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    unit = u.attrs.get("unit", None) or u.attrs.get("units", None)
    time_format = kwargs.get("time_format", "%Y-%m-%dT%H")
    time = u[actual_time_name].dt.strftime(time_format).values
    if field_name is None:
        titles = [f"{t}" for t in time]
    else:
        titles = [f"{field_name} · {t}" for t in time]

    quiver_kwargs = kwargs.copy()
    quiver_kwargs["x_name"] = actual_x_name
    quiver_kwargs["y_name"] = actual_y_name

    animation.quiver(
        u=u,
        v=v,
        path=output_path,
        title=titles,
        label=unit,
        subsample=subsample,
        arrows_kwgs=arrows_kwgs,
        **quiver_kwargs,
    )
