from importlib.metadata import version

from ._classic import Animation, PlotModel, animate, plot_da
from ._quiver import QuiverAnimation, animate_quiver, plot_da_quiver
from ._stream import StreamAnimation, animate_stream, plot_da_stream
from ._misc import check_ffmpeg

check_ffmpeg()

__all__ = [
    "Animation",
    "PlotModel",
    "animate",
    "plot_da",
    "plot_da_quiver",
    "animate_quiver",
    "QuiverAnimation",
    "plot_da_stream",
    "animate_stream",
    "StreamAnimation",
]

__version__ = version("mapflow")
