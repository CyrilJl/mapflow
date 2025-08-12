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
      <source src="_static/animation.mp4" type="video/mp4">
    </video>

Creating a static plot
----------------------

``mapflow`` also provides a simple way to create static plots of 2D ``xarray.DataArray`` objects using the ``plot_da`` function.

.. code-block:: python

   import xarray as xr
   from mapflow import plot_da

   ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
   plot_da(da=ds['t2m'].isel(time=0))

.. image:: /_static/plot_da.png
   :alt: Sample output of plot_da function
   :align: center
   :width: 75%
