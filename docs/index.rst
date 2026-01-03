:notoc: true

.. _index:

Welcome to mapflow's documentation!
===================================

``mapflow`` transforms 3D ``xr.DataArray`` (time, y, x) into video files in one line of code.

Installation
------------

You can install ``mapflow`` using ``pip`` or ``conda``.

With pip
~~~~~~~~

.. code-block:: bash

   pip install mapflow

With conda
~~~~~~~~~~

.. code-block:: bash

   conda install -c conda-forge -y mapflow

Dependencies
~~~~~~~~~~~~

``mapflow`` relies on ``matplotlib`` and ``ffmpeg``. If you're not installing ``mapflow`` from conda-forge, make sure ``ffmpeg`` is installed on your system.

Usage
-----

The main function of ``mapflow`` is ``animate``, which creates a video from an ``xarray.DataArray``.

.. code-block:: python

   import xarray as xr
   from mapflow import animate

   ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
   animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4', pad_inches=0.2)

.. video:: ../_static/animation.mp4
   :width: 640
   :height: 480

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   how_to_use
   api
