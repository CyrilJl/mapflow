:notoc: true

.. _how_to_use:

How to use
==========

This page provides examples of how to use ``mapflow`` for creating animations and static plots.

Animating a DataArray
---------------------

The main function of ``mapflow`` is ``animate``, which creates a video from an ``xarray.DataArray``.

.. code-block:: python

   import xarray as xr
   from mapflow import animate

   ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
   animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4')

.. raw:: html

    <video width="640" height="480" controls>
      <source src="../_static/animation.mp4" type="video/mp4">
    </video>

Creating a static plot
----------------------

``mapflow`` also provides a simple way to create static plots of 2D ``xarray.DataArray`` objects using the ``plot_da`` function.

.. code-block:: python

   import xarray as xr
   from mapflow import plot_da

   ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
   plot_da(da=ds['t2m'].isel(time=0))

.. image:: ../_static/plot_da.png
   :alt: Sample output of plot_da function
   :align: center
   :width: 50%

Quiver plots
------------

You can also create quiver plots to visualize vector fields. The `plot_da_quiver` function takes two `xarray.DataArray` objects representing the U and V components of the vector field.

.. code-block:: python

    import xarray as xr
    from mapflow import plot_da_quiver

    ds = xr.tutorial.load_dataset("air_temperature_gradient").isel(time=0)
    plot_da_quiver(u=ds["dTdx"], v=ds["dTdy"], subsample=4)

Similarly, you can create quiver animations using the `animate_quiver` function.

.. code-block:: python

    import xarray as xr
    from mapflow import animate_quiver

    ds = xr.tutorial.load_dataset("air_temperature_gradient")
    animate_quiver(u=ds["dTdx"], v=ds["dTdy"], path='quiver_animation.mkv', subsample=3)

Advanced Usage: `PlotModel` and `Animation` classes
---------------------------------------------------

For more control and efficiency when creating multiple plots or animations of the same geographic domain, you can use the `PlotModel` and `Animation` classes directly. These classes pre-compute geographic borders, which can save time.

Using `PlotModel`:

.. code-block:: python

    import xarray as xr
    from mapflow import PlotModel

    ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
    da = ds["t2m"].isel(time=0)

    p = PlotModel(x=da.longitude, y=da.latitude)
    p(da)

Using `Animation`:

.. code-block:: python

    import xarray as xr
    from mapflow import Animation

    ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
    da = ds["t2m"].isel(time=slice(120))

    animation = Animation(x=da.longitude, y=da.latitude, verbose=1)
    animation(da, "animation.mp4")

Key Features
------------

``mapflow`` is designed to be intuitive and requires minimal user input. Here are some of the key features that make it easy to use:

* **Automatic Coordinate Detection**: ``mapflow`` automatically detects the names of the x, y, and time coordinates in your ``xarray.DataArray``. If it fails to find them, you can specify them using the ``x_name``, ``y_name``, and ``time_name`` arguments.

* **Automatic CRS Detection**: The library automatically tries to determine the Coordinate Reference System (CRS) from your data. If no CRS is found, you can pass it directly using the ``crs`` argument.

* **Robust Colorbars**: ``mapflow`` generates a colorbar that is robust to outliers by default, using the 0.01 and 99.9 quantiles. You can also customize the colorbar using the ``vmin``, ``vmax``, and ``cmap`` arguments, or even pass a custom `matplotlib.colors.Normalize` object via the `norm` argument.

* **Integrated World Borders**: ``mapflow`` includes a built-in set of world borders for plotting. If you need to use custom borders, you can provide them as a ``geopandas.GeoSeries`` or ``geopandas.GeoDataFrame`` using the ``borders`` argument.

* **One-line Alternative to Cartopy**: The ``plot_da`` function provides a simple, one-line alternative to creating maps with ``cartopy``, making it quick and easy to visualize your geospatial data.

* **Flexible Output**: Animations can be saved in various formats, including `.mp4`, `.mov`, and `.avi`.

* **Parallel Processing**: Frame generation for animations is done in parallel to speed up the process. You can control the number of parallel jobs with the `n_jobs` argument.
