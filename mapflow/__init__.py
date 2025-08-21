from importlib.metadata import version

from ._classic import Animation, PlotModel, animate, plot_da
from ._quiver import QuiverAnimation, animate_quiver, plot_da_quiver
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
]

__version__ = version("mapflow")
