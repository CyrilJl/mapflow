import subprocess
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyproj import CRS
from tqdm.auto import tqdm

from ._plot import PlotModel


class Animation:
    def __init__(self, x, y, crs=4326, verbose=0, borders=None):
        """
        Initializes the Animation class.

        Sets up the plotting model with the given coordinates, CRS, and borders.

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
        """
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
        label=None,
        dpi=180,
        n_jobs=None,
        timeout="auto",
    ):
        """
        Generates an animation from a sequence of 2D data arrays.

        The method processes the input data, optionally upsamples it for smoother
        transitions, generates individual frames in parallel, and then compiles
        these frames into a video file using FFmpeg.

        Args:
            data (np.ndarray): A 3D numpy array where the first dimension is time
                (or frame sequence) and the next two are spatial (y, x).
            path (str | Path): The output path for the generated video file.
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
            label (str, optional): Label for the colorbar. Defaults to None.
            dpi (int, optional): Dots per inch for the saved frames. Defaults to 180.
            n_jobs (int, optional): Number of parallel jobs for frame generation.
                Defaults to 2/3 of CPU cores.
        """
        norm = self.plot._norm(data, vmin, vmax, qmin, qmax, norm, log)
        titles = self._process_title(title, upsample_ratio)
        data = self.upsample(data, ratio=upsample_ratio)

        with TemporaryDirectory() as tempdir:
            frame_paths = [Path(tempdir) / f"frame_{k:08d}.png" for k in range(len(data))]
            args = [
                (
                    data[k],
                    frame_paths[k],
                    figsize,
                    titles[k],
                    cmap,
                    norm,
                    label,
                    dpi,
                )
                for k in range(len(data))
            ]

            # Generate frames in parallel
            n_jobs = int(2 / 3 * cpu_count()) if n_jobs is None else n_jobs
            with Pool(processes=n_jobs) as pool:
                list(
                    tqdm(
                        pool.imap(self._generate_frame, args),
                        total=len(data),
                        disable=(not self.verbose),
                        desc="Frames generation",
                        leave=False,
                    )
                )

            timeout = min(10, 0.1 * len(data)) if timeout == "auto" else timeout
            # ffmpeg command to create the video
            self._create_video(tempdir, path, fps, timeout=timeout)

    def _generate_frame(self, args):
        """Generates a frame and saves it as a PNG."""
        data_frame, frame_path, figsize, title, cmap, norm, label, dpi = args
        self.plot(
            data=data_frame,
            figsize=figsize,
            title=title,
            show=False,
            cmap=cmap,
            norm=norm,
            label=label,
        )
        plt.savefig(frame_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        plt.clf()
        plt.close()

    def _create_video(self, tempdir, path, fps, timeout):
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "image2",
            "-framerate",
            str(fps),
            "-i",
            Path(tempdir) / "frame_%08d.png",
            "-vcodec",
            "libx264",
            "-crf",
            "22",
            path,
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,  # Captures the output as text
                stdout=subprocess.PIPE,  # Captures standard output
                stderr=subprocess.PIPE,  # Captures standard error
                timeout=timeout,
            )
            print(result.stdout)  # Optional: display standard output
        except subprocess.CalledProcessError as e:
            print(f"Error during command execution: {e}")
            print(f"Standard output: {e.stdout}")
            print(f"Standard error: {e.stderr}")
            raise  # Propagate the error if necessary


def check_da(da: xr.DataArray, time_name, x_name, y_name, crs):
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray, got {type(da)}")
    for dim in (x_name, y_name, time_name):
        if dim not in da.dims:
            raise ValueError(f"Dimension '{dim}' not found in DataArray dimensions: {da.dims}")
    crs_ = CRS.from_user_input(crs)
    if crs_.is_geographic:
        da[x_name] = xr.where(da[x_name] > 180, da[x_name] - 360, da[x_name])
        da = da.sortby(x_name)
    return da, crs_


def animate(
    da: xr.DataArray,
    path: str,
    time_name="time",
    x_name="longitude",
    y_name="latitude",
    crs=4326,
    borders=None,
    cmap="jet",
    norm=None,
    log=False,
    qmin=0.01,
    qmax=99.9,
    vmin=None,
    vmax=None,
    time_format="%Y-%m-%dT%H",
    upsample_ratio=4,
    fps=24,
    n_jobs=None,
    verbose=0,
):
    """
    Creates an animation from an xarray DataArray.

    This function prepares data from an xarray DataArray (e.g., handling
    geographic coordinates, extracting time information for titles) and
    then uses the `Animation` class to generate and save the animation.

    Args:
        da (xr.DataArray): Input DataArray with at least time, x, and y dimensions.
        path (str): Output path for the video file.
        time_name (str, optional): Name of the time coordinate in `da`.
            Defaults to "time".
        x_name (str, optional): Name of the x-coordinate (e.g., longitude) in `da`.
            Defaults to "longitude".
        y_name (str, optional): Name of the y-coordinate (e.g., latitude) in `da`.
            Defaults to "latitude".
        crs (int | str | CRS, optional): Coordinate Reference System of the data.
            Defaults to 4326 (WGS84).
        borders (gpd.GeoDataFrame | gpd.GeoSeries | None, optional):
            Custom borders to use for plotting. If None, defaults to
            world borders. Defaults to None.
        cmap (str, optional): Colormap for the plot. Defaults to "jet".
        norm (matplotlib.colors.Normalize, optional): Custom normalization object.
        log (bool, optional): Use logarithmic color scale. Defaults to False.
        qmin (float, optional): Minimum quantile for color normalization.
            Defaults to 0.01.
        qmax (float, optional): Maximum quantile for color normalization.
            Defaults to 99.9.
        vmin (float, optional): Minimum value for color normalization. Overrides qmin.
        vmax (float, optional): Maximum value for color normalization. Overrides qmax.
        time_format (str, optional): Strftime format for time in titles.
            Defaults to "%Y-%m-%dT%H".
        upsample_ratio (int, optional): Factor to upsample data temporally.
            Defaults to 4.
        fps (int, optional): Frames per second for the video. Defaults to 24.
        n_jobs (int, optional): Number of parallel jobs for frame generation.
            Defaults to None (auto-determined).
        verbose (int, optional): Verbosity level for the Animation class.
            Defaults to 0.
    """

    da, crs_ = check_da(da, time_name, x_name, y_name, crs)

    animation = Animation(x=da[x_name], y=da[y_name], crs=crs_, verbose=verbose, borders=borders)
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    unit = da.attrs.get("unit", None) or da.attrs.get("units", None)
    time = da[time_name].dt.strftime(time_format).values
    field = da.name or da.attrs.get("long_name")
    titles = [f"{field} · {t}" for t in time]
    animation(
        data=da.values,
        path=path,
        title=titles,
        norm=norm,
        log=log,
        cmap=cmap,
        qmin=qmin,
        qmax=qmax,
        vmin=vmin,
        vmax=vmax,
        upsample_ratio=upsample_ratio,
        fps=fps,
        label=unit,
        dpi=180,
        n_jobs=n_jobs,
    )
