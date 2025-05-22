from ._animate import Animation, animate
from ._misc import check_ffmpeg
from ._plot import PlotModel

check_ffmpeg()

__all__ = ["Animation", "PlotModel", "animate"]
