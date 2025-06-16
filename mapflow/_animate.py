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

X_NAME_CANDIDATES = ("x", "lon", "longitude")
Y_NAME_CANDIDATES = ("y", "lat", "latitude")
TIME_NAME_CANDIDATES = ("time", "t", "times")


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

            timeout = max(10, 0.1 * len(data)) if timeout == "auto" else timeout
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

    @staticmethod
    def _create_video(tempdir, path, fps, timeout):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in (".avi", ".mkv", ".mov", ".mp4"):
            raise ValueError("Output format must be either .avi, .mkv, .mov or .mp4")

        # Base command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite without asking
            "-f",
            "image2",
            "-framerate",
            str(fps),
            "-i",
            str(Path(tempdir) / "frame_%08d.png"),
        ]

        # Add codec and format specific options
        if suffix in (".mkv", ".mov", ".mp4"):
            cmd.extend(
                [
                    "-vcodec",
                    "libx265",
                    "-crf",
                    "22",  # Quality level (0-51, lower is better)
                ]
            )
        elif suffix == ".avi":
            cmd.extend(
                [
                    "-vcodec",
                    "mpeg4",
                    "-q:v",
                    "5",  # Quality level (1-31, lower is better)
                ]
            )

        cmd.append(str(path))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if result.stdout:  # Only print if there's output
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


def process_crs(da, crs):
    if crs is None:
        if "spatial_ref" in da.coords:
            crs = da.spatial_ref.attrs.get("crs_wkt", 4326)
        else:
            crs = 4326
    ret = CRS.from_user_input(crs)
    return ret


def check_da(da: xr.DataArray, time_name, x_name, y_name, crs):
    if not isinstance(da, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray, got {type(da)}")
    for dim in (x_name, y_name, time_name):
        if dim not in da.dims:
            raise ValueError(f"Dimension '{dim}' not found in DataArray dimensions: {da.dims}")
    crs_ = process_crs(da, crs)
    if crs_.is_geographic:
        da[x_name] = xr.where(da[x_name] > 180, da[x_name] - 360, da[x_name])
    ret = da.sortby(x_name).sortby(y_name).sortby(time_name).squeeze()
    if ret.ndim != 3:
        raise ValueError(f"DataArray must have 3 dimensions (time, {y_name}, {x_name}), got {da.ndim} dimensions.")
    ret = ret.transpose(time_name, y_name, x_name)
    return ret, crs_


def _guess_coord_name(da_coords, candidates, provided_name, coord_type_for_error):
    """
    Guesses the coordinate name if not provided.
    Iterates through da_coords, compares lowercased names with candidates.
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


def animate(
    da: xr.DataArray,
    path: str,
    time_name=None,
    x_name=None,
    y_name=None,
    crs=None,
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
    dpi=180,
    n_jobs=None,
    timeout="auto",
    verbose=0,
):
    """
    Creates an animation from an xarray DataArray.

    This function prepares data from an xarray DataArray (e.g., handling
    geographic coordinates, extracting time information for titles) and
    then uses the `Animation` class to generate and save the animation.

    Args:
        da (xr.DataArray): Input DataArray with at least time, x, and y dimensions.
        path (str): Output path for the video file. Supported formats are avi, mov
            and mp4.
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
    # Guess coordinate names if not provided
    actual_time_name = _guess_coord_name(da.coords, TIME_NAME_CANDIDATES, time_name, "time")
    actual_x_name = _guess_coord_name(da.coords, X_NAME_CANDIDATES, x_name, "x")
    actual_y_name = _guess_coord_name(da.coords, Y_NAME_CANDIDATES, y_name, "y")

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
    time = da[actual_time_name].dt.strftime(time_format).values
    field = da.name or da.attrs.get("long_name")
    titles = [f"{field} · {t}" for t in time]
    animation(
        data=da.values,
        path=output_path,
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
        dpi=dpi,
        timeout=timeout,
        n_jobs=n_jobs,
    )
