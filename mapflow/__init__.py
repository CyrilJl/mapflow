from importlib.metadata import version

from ._classic import Animation, PlotModel, animate, plot_da
from ._quiver import QuiverAnimation, animate_quiver, plot_da_quiver

__all__ = [
    "Animation",
    "PlotModel",
    "QuiverAnimation",
    "animate",
    "animate_quiver",
    "plot_da",
    "plot_da_quiver",
]

__version__ = version("mapflow")
