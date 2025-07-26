from ._animate import Animation, animate
from ._misc import check_ffmpeg
from ._plot import PlotModel, plot_da

check_ffmpeg()

__all__ = ["Animation", "PlotModel", "animate", "plot_da"]
