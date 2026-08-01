import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon

from mapflow import plot_da, plot_da_quiver


def test_plot_da(air_data):
    plot_da(da=air_data.isel(time=0), show=False)
    plt.close()


def test_plot_da_quiver(air_temperature_gradient_data):
    u = air_temperature_gradient_data["dTdx"].isel(time=0)
    v = air_temperature_gradient_data["dTdx"].isel(time=0)
    plot_da_quiver(u, v, show=False)
    plt.close()
    plot_da_quiver(u, v, subsample=2, show=False)
    plt.close()


def test_plot_da_diff(air_data):
    plot_da(da=air_data.isel(time=0), diff=True, show=False)
    plt.close()


def test_plot_da_with_linestring_borders(air_data):
    polygon = Polygon([(250, 20), (251, 21), (251, 20)])
    line = LineString([(260, 50), (261, 51)])
    gdf = gpd.GeoDataFrame(geometry=[polygon, line], crs="EPSG:4326")
    plot_da(da=air_data.isel(time=0), borders=gdf, show=False)
    plt.close()


def test_plot_da_subsample(air_data):
    plot_da(da=air_data.isel(time=0), subsample=2, show=False)
    plt.close()
