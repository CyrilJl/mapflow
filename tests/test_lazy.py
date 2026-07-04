import numpy as np
import pytest
import xarray as xr
from matplotlib.colors import LogNorm, Normalize

from mapflow import Animation, QuiverAnimation, animate
from mapflow._classic import PlotModel


class LoadCounter:
    """Sequence wrapper that records which frames are materialized."""

    def __init__(self, arr):
        self.arr = arr
        self.loads = []

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, k):
        self.loads.append(k)
        return self.arr[k]


@pytest.mark.parametrize("ratio", [1, 2, 5])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_iter_upsampled_frames_matches_upsample(ratio, dtype):
    rng = np.random.default_rng(0)
    data = rng.random((5, 4, 3)).astype(dtype)
    expected = Animation.upsample(data, ratio=ratio)
    frames = list(Animation._iter_upsampled_frames(data, ratio=ratio))
    got = np.stack(frames)
    assert got.dtype == expected.dtype
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_iter_upsampled_frames_single_frame():
    data = np.random.default_rng(0).random((1, 4, 3))
    frames = list(Animation._iter_upsampled_frames(data, ratio=5))
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], data[0])


def test_iter_upsampled_frames_empty():
    with pytest.raises(ValueError):
        list(Animation._iter_upsampled_frames(np.empty((0, 4, 3)), ratio=2))


def test_animation_streams_each_frame_once(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.random((6, 16, 16))
    counter = LoadCounter(arr)
    animation = Animation(x=np.linspace(0, 15, 16), y=np.linspace(40, 55, 16))
    animation(counter, tmp_path / "out.mp4", vmin=0.0, vmax=1.0, upsample_ratio=3, dpi=60)
    # vmin/vmax provided: no pass over the data for the norm, each raw frame
    # is loaded exactly once, in order, by the streaming interpolation.
    assert counter.loads == list(range(6))
    assert (tmp_path / "out.mp4").exists()


def test_quiver_streams_each_frame_once(tmp_path):
    rng = np.random.default_rng(0)
    u = LoadCounter(rng.random((5, 12, 12)))
    v = LoadCounter(rng.random((5, 12, 12)))
    animation = QuiverAnimation(x=np.linspace(0, 11, 12), y=np.linspace(40, 51, 12))
    animation.quiver(
        u,
        v,
        tmp_path / "out.mp4",
        vmin=0.0,
        vmax=2.0,
        dpi=60,
        x_name="lon",
        y_name="lat",
    )
    assert u.loads == list(range(5))
    assert v.loads == list(range(5))
    assert (tmp_path / "out.mp4").exists()


def test_animate_lazy_netcdf(tmp_path):
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.random((5, 12, 14)).astype("float32"),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-06")),
            "lat": np.linspace(40, 50, 12),
            "lon": np.linspace(-5, 5, 14),
        },
        name="field",
    )
    nc_path = tmp_path / "data.nc"
    da.to_netcdf(nc_path)
    out = tmp_path / "out.mp4"
    with xr.open_dataarray(nc_path) as lazy:
        # Default quantile-based scaling: exercises the streaming norm pass.
        animate(da=lazy, path=str(out), dpi=60)
    assert out.exists()


def test_norm_streaming_default_bounds_widen_exact():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 20, 20))
    exact = PlotModel._norm(data, None, None, 1, 99, None, log=False)
    streamed = PlotModel._norm_streaming(iter(data), None, None, 1, 99, None, log=False)
    assert type(streamed) is Normalize
    assert streamed.vmin <= exact.vmin
    assert streamed.vmax >= exact.vmax


def test_norm_streaming_log():
    rng = np.random.default_rng(0)
    data = rng.random((4, 10, 10)) + 0.1
    streamed = PlotModel._norm_streaming(iter(data), None, None, 0.01, 99.9, None, log=True)
    assert isinstance(streamed, LogNorm)
    assert streamed.vmin > 0
    assert streamed.vmax > streamed.vmin


def test_norm_streaming_log_no_positive_values():
    data = np.full((3, 5, 5), -1.0)
    streamed = PlotModel._norm_streaming(iter(data), None, None, 0.01, 99.9, None, log=True)
    assert streamed.vmin == pytest.approx(1e-1)
    assert streamed.vmax == pytest.approx(1e0)


def test_norm_streaming_diff_is_symmetric():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 10, 10))
    streamed = PlotModel._norm_streaming(iter(data), None, None, 0.01, 99.9, None, log=False, diff=True)
    assert streamed.vmin == -streamed.vmax


def test_norm_streaming_explicit_bounds_skip_data():
    def frames():
        raise AssertionError("frames should not be consumed when vmin/vmax are given")
        yield

    norm = PlotModel._norm_streaming(frames(), 0.0, 1.0, 0.01, 99.9, None, log=False)
    assert (norm.vmin, norm.vmax) == (0.0, 1.0)
