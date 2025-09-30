from copy import copy
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from tqdm.auto import tqdm

from ._misc import (
    TIME_NAME_CANDIDATES,
    X_NAME_CANDIDATES,
    Y_NAME_CANDIDATES,
    check_da,
    guess_coord_name,
    process_crs,
)


class PlotModel:
    """A class for plotting 2D data with geographic borders. Useful for multiple
    plots of the same geographic domain, as it pre-computes geographic borders.

    Args:
        x (np.ndarray): Array of x-coordinates (e.g., longitudes).
        y (np.ndarray): Array of y-coordinates (e.g., latitudes).
        crs (int | str | CRS, optional): Coordinate Reference System.
            Defaults to 4326 (WGS84).
        borders (gpd.GeoDataFrame | gpd.GeoSeries | None): Custom borders to use.
            If None, defaults to world borders from a packaged GeoPackage.

    .. code-block:: python

        import xarray as xr
        from mapflow import PlotModel

        ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
        da = ds["t2m"].isel(time=0)

        p = PlotModel(x=da.longitude, y=da.latitude)
        p(da)

    """

    def __init__(self, x, y, crs=4326, borders=None):
        self.x = np.asarray_chkfinite(x)
        self.y = np.asarray_chkfinite(y)
        if self.x.ndim != self.y.ndim:
            raise ValueError("x and y must have the same dimensionality (both 1D or both 2D)")

        self.crs = CRS.from_user_input(crs)
        if self.crs.is_geographic:
            self.aspect = 1 / np.cos((self.y.mean() * np.pi / 180))
        else:
            self.aspect = 1
        if self.x.ndim == 1:
            self.dx = abs(self.x[1] - self.x[0])
            self.dy = abs(self.y[1] - self.y[0])
        else:
            self.dx = np.diff(self.x, axis=1).max()
            self.dy = np.diff(self.y, axis=0).max()
        bbox = (
            self.x.min() - 10 * self.dx,
            self.y.min() - 10 * self.dy,
            self.x.max() + 10 * self.dx,
            self.y.max() + 10 * self.dy,
        )

        if borders is None:
            borders_ = gpd.read_file(Path(__file__).parent / "_static" / "world.gpkg")
        elif isinstance(borders, (gpd.GeoDataFrame, gpd.GeoSeries)):
            borders_ = borders
        else:
            raise TypeError("borders must be a geopandas GeoDataFrame, GeoSeries, or None.")
        borders_ = borders_.to_crs(self.crs).clip(bbox)
        self.borders = self._shp_to_lines(borders_)

    @staticmethod
    def _shp_to_lines(gdf):
        lines = []
        for geom in gdf.geometry.values:
            if isinstance(geom, Polygon):
                lines.append(geom.exterior.coords)
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    lines.append(poly.exterior.coords)
            elif isinstance(geom, LineString):
                lines.append(geom.coords)
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    lines.append(line.coords)
        return LineCollection(lines, linewidth=0.5, edgecolor="k")

    @staticmethod
    def _log_norm(data, vmin, vmax, qmin, qmax):
        """Generates a logarithmic normalization."""
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return Normalize(vmin=1e-1, vmax=1e0)

        vmin = np.nanpercentile(positive_data, q=qmin) if vmin is None else vmin
        vmax = np.nanpercentile(positive_data, q=qmax) if vmax is None else vmax

        if vmin <= 0 or vmax <= 0:
            raise ValueError(f"Normalization range for log scale must be positive. Got vmin={vmin}, vmax={vmax}")
        return LogNorm(vmin=vmin, vmax=vmax)

    @staticmethod
    def _norm(data, vmin, vmax, qmin, qmax, norm, log, diff=False):
        """Generates a normalization based on the specified parameters.

        Args:
            data (array-like): Data to normalize.
            vmin (float): Minimum value for normalization.
            vmax (float): Maximum value for normalization.
            qmin (float): Minimum quantile for normalization (0-100).
            qmax (float): Maximum quantile for normalization (0-100).
            norm (matplotlib.colors.Normalize): Custom normalization object.
            log (bool): Indicates if a logarithmic scale should be used.
            diff (bool): Indicates if a divergent colormap should be used.

        Returns:
            matplotlib.colors.Normalize: Normalization object.

        Raises:
            ValueError: If qmin/qmax are not between 0-100 or if log=True with no positive values.
        """
        # Validate quantile ranges
        if not (0 <= qmin <= 100):
            raise ValueError(f"qmin must be between 0 and 100, got {qmin}")
        if not (0 <= qmax <= 100):
            raise ValueError(f"qmax must be between 0 and 100, got {qmax}")
        if qmin >= qmax:
            raise ValueError(f"qmin must be less than qmax, got {qmin} and {qmax}")

        if norm is not None:
            return norm

        if diff:
            if vmax is None:
                vmax = np.nanpercentile(np.abs(data), q=qmax)
            vmin = -vmax
            return Normalize(vmin=vmin, vmax=vmax)

        if log:
            return PlotModel._log_norm(data, vmin, vmax, qmin, qmax)

        vmin = np.nanpercentile(data, q=qmin) if vmin is None else vmin
        vmax = np.nanpercentile(data, q=qmax) if vmax is None else vmax
        return Normalize(vmin=vmin, vmax=vmax)

    def _process_data(self, data):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array.")

        if self.x.ndim == 1:
            if data.shape[0] != self.y.size or data.shape[1] != self.x.size:
                raise ValueError("Data shape does not match x and y dimensions.")
        else:
            if data.shape != self.x.shape:
                raise ValueError("Data shape does not match x and y dimensions.")
        return data

    def __call__(
        self,
        data,
        figsize=None,
        qmin=0.01,
        qmax=99.9,
        vmin=None,
        vmax=None,
        log=False,
        diff=False,
        cmap="jet",
        norm=None,
        shading="nearest",
        shrink=0.4,
        label=None,
        title=None,
        show=True,
    ):
        """
        Plots a 2D data array using imshow or pcolormesh.

        This method handles the actual plotting of a single frame. It applies
        normalization, colormaps, adds a colorbar, overlays borders, sets the
        aspect ratio, title, and optionally displays the plot.

        Args:
            data (np.ndarray): 2D array of data to plot.
            figsize (tuple[float, float], optional): Figure size (width, height)
                in inches. Defaults to None (matplotlib's default).
            qmin (float, optional): Minimum quantile for color normalization if
                vmin is not set. Defaults to 0.01.
            qmax (float, optional): Maximum quantile for color normalization if
                vmax is not set. Defaults to 99.9.
            vmin (float, optional): Minimum value for color normalization.
                Overrides qmin. Defaults to None.
            vmax (float, optional): Maximum value for color normalization.
                Overrides qmax. Defaults to None.
            log (bool, optional): Whether to use a logarithmic color scale.
                Defaults to False.
            diff (bool, optional): Whether to use a divergent colormap.
                Defaults to False.
            cmap (str, optional): Colormap to use. Defaults to "jet".
            norm (matplotlib.colors.Normalize, optional): Custom normalization object.
                Overrides vmin, vmax, qmin, qmax, log. Defaults to None.
            shading (str, optional): Shading method for pcolormesh.
                Defaults to "nearest".
            shrink (float, optional): Factor by which to shrink the colorbar.
                Defaults to 0.4.
            label (str, optional): Label for the colorbar. Defaults to None.
            title (str, optional): Title for the plot. Defaults to None.
            show (bool, optional): Whether to display the plot using `plt.show()`.
                Defaults to True.
        """
        if diff:
            cmap = "bwr"
        data = self._process_data(data)
        norm = self._norm(data, vmin, vmax, qmin, qmax, norm, log=log, diff=diff)
        plt.figure(figsize=figsize)
        if (self.x.ndim == 1) and (self.y.ndim == 1):
            plt.imshow(
                X=data,
                cmap=cmap,
                norm=norm,
                origin="lower",
                extent=(
                    self.x.min() - self.dx / 2,
                    self.x.max() + self.dx / 2,
                    self.y.min() - self.dy / 2,
                    self.y.max() + self.dy / 2,
                ),
                interpolation=shading,
            )
        else:
            plt.pcolormesh(
                self.x,
                self.y,
                data,
                cmap=cmap,
                norm=norm,
                shading=shading,
                rasterized=True,
            )
        plt.colorbar(shrink=shrink, label=label)
        plt.xlim(self.x.min() - self.dx / 2, self.x.max() + self.dx / 2)
        plt.ylim(self.y.min() - self.dy / 2, self.y.max() + self.dy / 2)
        plt.gca().add_collection(copy(self.borders))
        plt.gca().set_aspect(self.aspect)
        plt.title(title)
        plt.gca().axis("off")
        plt.tight_layout()
        if show:
            plt.show()


def plot_da(da: xr.DataArray, x_name=None, y_name=None, crs=None, borders=None, diff=False, **kwargs):
    """Convenience function for quick plotting of an xarray DataArray using PlotModel.

    This is a simplified wrapper around the `PlotModel` class that handles:
    - Automatic coordinate detection
    - CRS processing
    - Data sorting and longitude wrapping (for geographic CRS)
    - Single-call plotting

    For better performance when making multiple plots of the same geographic domain,
    consider using `PlotModel` directly, which pre-computes geographic borders and
    can be reused for multiple plots.

    Args:
        da (xr.DataArray): xarray DataArray with 2D data to plot. Must have appropriate coordinates.
        x_name (str, optional): Name of the x-coordinate dimension. If None, will attempt to guess from `["x", "lon", "longitude"]`.
        y_name (str, optional): Name of the y-coordinate dimension. If None, will attempt to guess from `["y", "lat", "latitude"]`.
        crs (int | str | CRS, optional): Coordinate Reference System. Can be an EPSG code, a PROJ string, or a pyproj.CRS object.
            If the DataArray has a 'crs' attribute, that will be used by default. Defaults to 4326 (WGS84).
        borders (gpd.GeoDataFrame | gpd.GeoSeries | None): Custom borders to use.
            If None, defaults to world borders from a packaged GeoPackage.
        diff (bool, optional): Whether to use a divergent colormap. Defaults to False.
        **kwargs: Additional arguments passed to `PlotModel.__call__`, including:
            - `figsize` (tuple, optional): Figure size (width, height) in inches.
            - `qmin`/`qmax` (float, optional): Quantile ranges for color scaling (0-100).
            - `vmin`/`vmax` (float, optional): Explicit value ranges for color scaling.
            - `log` (bool, optional): Whether to use a logarithmic color scale.
            - `cmap` (str, optional): Colormap name.
            - `norm` (matplotlib.colors.Normalize, optional): Custom normalization object.
            - `shading` (str, optional): Color shading method.
            - `shrink` (float, optional): Colorbar shrink factor.
            - `label` (str, optional): Colorbar label.
            - `title` (str, optional): Plot title.
            - `show` (bool, optional): Whether to display the plot.

    Example:
        .. code-block:: python

            import xarray as xr
            from mapflow import plot_da

            ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
            plot_da(da=ds['t2m'].isel(time=0))

    See Also:
        :class:`PlotModel`: The underlying plotting class used by this function.
    """
    actual_x_name = guess_coord_name(da.coords, X_NAME_CANDIDATES, x_name, "x")
    actual_y_name = guess_coord_name(da.coords, Y_NAME_CANDIDATES, y_name, "y")

    if da[actual_x_name].ndim == 1 and da[actual_y_name].ndim == 1:
        da = da.sortby(actual_x_name).sortby(actual_y_name)
    crs_ = process_crs(da, crs)
    if crs_.is_geographic:
        da[actual_x_name] = xr.where(da[actual_x_name] > 180, da[actual_x_name] - 360, da[actual_x_name])
        da = da.sortby(actual_x_name)

    p = PlotModel(
        x=da[actual_x_name].values,
        y=da[actual_y_name].values,
        crs=crs_,
        borders=borders,
    )
    data = p._process_data(da.values)
    p(data, diff=diff, **kwargs)


class Animation:
    """A class for creating animations from 3D data with geographic borders.

    This class is useful for creating multiple animations of the same geographic
    domain, as it pre-computes geographic borders.

    Args:
        x (np.ndarray): Array of x-coordinates (e.g., longitudes).
        y (np.ndarray): Array of y-coordinates (e.g., latitudes).
        crs (int | str | CRS, optional): Coordinate Reference System.
            Defaults to 4326 (WGS84).
        verbose (int, optional): Verbosity level. If > 0, progress bars
            will be shown. Defaults to 0.
        borders (gpd.GeoDataFrame | gpd.GeoSeries | None, optional):
            Custom borders to use for plotting. If None, defaults to
            world borders. Defaults to None.

    .. code-block:: python

        import xarray as xr
        from mapflow import Animation

        ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
        da = ds["t2m"].isel(time=slice(120))

        animation = Animation(x=da.longitude, y=da.latitude, verbose=1)
        animation(da, "animation.mp4")

    """

    def __init__(self, x, y, crs=4326, verbose=0, borders=None):
        self.plot = PlotModel(x=x, y=y, crs=crs, borders=borders)
        self.verbose = verbose

    @staticmethod
    def upsample(data, ratio=5):
        if ratio == 1:
            return data
        else:
            nt, ny, nx = data.shape
            ret = np.empty((ratio * (nt - 1) + 1, ny, nx), dtype=data.dtype)
            ret[::ratio] = data
            delta = np.diff(data, axis=0)
            for k in range(1, ratio):
                ret[k::ratio] = ret[::ratio][:-1] + k * delta / ratio
            return ret

    @staticmethod
    def _process_title(title, upsample_ratio):
        if title is None:
            return
        if isinstance(title, str):
            return [title] * upsample_ratio
        elif isinstance(title, (list, tuple)):
            return np.repeat(title, upsample_ratio).tolist()
        else:
            raise ValueError("Title must be a string or a list of strings.")

    def __call__(
        self,
        data,
        path,
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
        diff=False,
        label=None,
        dpi=180,
        n_jobs=None,
        timeout="auto",
    ):
        """Generates an animation from a sequence of 2D data arrays.

        The method processes the input data, optionally upsamples it for smoother
        transitions, generates individual frames in parallel, and then compiles
        these frames into a video file using FFmpeg.

        Args:
            data (np.ndarray): A 3D numpy array where the first dimension is time
                (or frame sequence) and the next two are spatial (y, x).
            path (str | Path): The output path for the generated video file.
                Supported formats are avi, mov and mp4.
            figsize (tuple[float, float], optional): Figure size (width, height)
                in inches. Defaults to None (matplotlib's default).
            title (str | list[str], optional): Title for the plot. If a string,
                it's used for all frames. If a list, each element corresponds to a
                frame's title (before upsampling). Defaults to None.
            fps (int, optional): Frames per second for the output video.
                Defaults to 24.
            upsample_ratio (int, optional): Factor by which to upsample the data
                along the time axis for smoother animations. Defaults to 2.
            cmap (str, optional): Colormap to use for the plot. Defaults to "jet".
            qmin (float, optional): Minimum quantile for color normalization.
                Defaults to 0.01.
            qmax (float, optional): Maximum quantile for color normalization.
                Defaults to 99.9.
            vmin (float, optional): Minimum value for color normalization. Overrides qmin.
            vmax (float, optional): Maximum value for color normalization. Overrides qmax.
            norm (matplotlib.colors.Normalize, optional): Custom normalization object.
            log (bool, optional): Whether to use a logarithmic color scale. Defaults to False.
            diff (bool, optional): Whether to use a divergent colormap. Defaults to False.
            label (str, optional): Label for the colorbar. Defaults to None.
            dpi (int, optional): Dots per inch for the saved frames. Defaults to 180.
            n_jobs (int, optional): Number of parallel jobs for frame generation.
                Defaults to 2/3 of CPU cores.
            timeout (int | str, optional): Timeout for the ffmpeg command in seconds.
                Defaults to "auto", which sets the timeout to `max(20, 0.1 * data_len)`.
        """
        if diff:
            cmap = "bwr"
        norm = self.plot._norm(data, vmin, vmax, qmin, qmax, norm, log, diff)
        self._animate(
            data=data,
            path=path,
            frame_generator=self._generate_frame,
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
            diff=diff,
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
        diff=False,
    ):
        titles = self._process_title(title, upsample_ratio)
        data = self.upsample(data, ratio=upsample_ratio)
        data_len = len(data)

        with TemporaryDirectory() as tempdir:
            frame_paths = [Path(tempdir) / f"frame_{k:08d}.png" for k in range(data_len)]
            args = []
            for k in range(data_len):
                frame_data = data[k]
                arg_tuple = (
                    frame_data,
                    frame_paths[k],
                    figsize,
                    titles[k] if titles and k < len(titles) else None,
                    cmap,
                    norm,
                    label,
                    dpi,
                    {"diff": diff},
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

    def _generate_frame(self, args):
        """Generates a frame and saves it as a PNG."""
        data_frame, frame_path, figsize, title, cmap, norm, label, dpi, kwargs = args
        self.plot(
            data=data_frame,
            figsize=figsize,
            title=title,
            show=False,
            cmap=cmap,
            norm=norm,
            label=label,
            **kwargs,
        )
        plt.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.clf()
        plt.close()

    @staticmethod
    def _build_ffmpeg_cmd(tempdir, path, fps):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in (".avi", ".mkv", ".mov", ".mp4"):
            raise ValueError("Output format must be either .avi, .mkv, .mov or .mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "image2",
            "-framerate",
            str(fps),
            "-i",
            str(Path(tempdir) / "frame_%08d.png"),
        ]
        if suffix in (".mkv", ".mov", ".mp4"):
            cmd.extend(["-vcodec", "libx265", "-crf", "22"])
        elif suffix == ".avi":
            cmd.extend(["-vcodec", "mpeg4", "-q:v", "5"])
        cmd.append(str(path))
        return cmd

    @staticmethod
    def _create_video(tempdir, path, fps, timeout):
        cmd = Animation._build_ffmpeg_cmd(tempdir, path, fps)
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during video creation: {e}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Standard output: {e.stdout}")
            print(f"Standard error: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            print(f"Video creation timed out after {timeout} seconds")
            raise


def animate(
    da: xr.DataArray,
    path: str,
    time_name: str = None,
    x_name: str = None,
    y_name: str = None,
    crs=None,
    borders: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    verbose: int = 0,
    diff=False,
    **kwargs,
):
    """Creates an animation from an xarray DataArray.

    This function prepares data from an xarray DataArray (e.g., handling
    geographic coordinates, extracting time information for titles) and
    then uses the `Animation` class to generate and save the animation.

    Args:
        da (xr.DataArray): Input DataArray with at least time, x, and y dimensions.
        path (str): Output path for the video file. Supported formats are avi, mov
            and mp4.
        time_name (str, optional): Name of the time coordinate in `da`. If None,
            it's guessed from `["time", "t", "times"]`. Defaults to None.
        x_name (str, optional): Name of the x-coordinate (e.g., longitude) in `da`.
            If None, it's guessed from `["x", "lon", "longitude"]`. Defaults to None.
        y_name (str, optional): Name of the y-coordinate (e.g., latitude) in `da`.
            If None, it's guessed from `["y", "lat", "latitude"]`. Defaults to None.
        crs (int | str | CRS, optional): Coordinate Reference System of the data.
            Defaults to 4326 (WGS84).
        borders (gpd.GeoDataFrame | gpd.GeoSeries | None, optional):
            Custom borders to use for plotting. If None, defaults to
            world borders. Defaults to None.
        verbose (int, optional): Verbosity level for the Animation class.
            Defaults to 0.
        **kwargs: Additional keyword arguments passed to the `Animation` class, including:
            - `cmap` (str, optional): Colormap for the plot.
            - `norm` (matplotlib.colors.Normalize, optional): Custom normalization object.
            - `log` (bool, optional): Use logarithmic color scale.
            - `diff` (bool, optional): Whether to use a divergent colormap.
            - `qmin` (float, optional): Minimum quantile for color normalization.
            - `qmax` (float, optional): Maximum quantile for color normalization.
            - `vmin` (float, optional): Minimum value for color normalization.
            - `vmax` (float, optional): Maximum value for color normalization.
            - `time_format` (str, optional): Strftime format for time in titles.
            - `upsample_ratio` (int, optional): Factor to upsample data temporally.
            - `fps` (int, optional): Frames per second for the video.
            - `n_jobs` (int, optional): Number of parallel jobs for frame generation.
            - `dpi` (int, optional): Dots per inch for the saved frames.
            - `timeout` (str | int, optional): Timeout for video creation.


    .. code-block:: python

        import xarray as xr
        from mapflow import animate

        ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
        animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4')

    See Also:
        :class:`Animation`: The underlying animation class used by this function.
    """
    actual_time_name = guess_coord_name(da.coords, TIME_NAME_CANDIDATES, time_name, "time")
    actual_x_name = guess_coord_name(da.coords, X_NAME_CANDIDATES, x_name, "x")
    actual_y_name = guess_coord_name(da.coords, Y_NAME_CANDIDATES, y_name, "y")

    da, crs_ = check_da(da, actual_time_name, actual_x_name, actual_y_name, crs)

    animation = Animation(
        x=da[actual_x_name].values,
        y=da[actual_y_name].values,
        crs=crs_,
        verbose=verbose,
        borders=borders,
    )
    output_path = Path(path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    unit = da.attrs.get("unit", None) or da.attrs.get("units", None)
    time_format = kwargs.get("time_format", "%Y-%m-%dT%H")
    time = da[actual_time_name].dt.strftime(time_format).values
    field = da.name or da.attrs.get("long_name")
    titles = [f"{field} · {t}" for t in time]
    animation(
        data=da.values,
        path=output_path,
        title=titles,
        label=unit,
        diff=diff,
        **kwargs,
    )
