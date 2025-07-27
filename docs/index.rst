mapflow
=======

.. image:: ../_static/logo.svg
   :alt: mapflow logo
   :width: 200
   :align: center

.. raw:: html

   <div align="center">

.. image:: https://badge.fury.io/py/mapflow.svg
   :target: https://badge.fury.io/py/mapflow
.. image:: https://anaconda.org/conda-forge/mapflow/badges/version.svg
   :target: https://anaconda.org/conda-forge/mapflow
.. image:: https://github.com/CyrilJl/mapflow/actions/workflows/pytest.yaml/badge.svg
   :target: https://github.com/CyrilJl/mapflow/actions/workflows/pytest.yaml

.. raw:: html

   </div>

``mapflow`` transforms 3D ``xr.DataArray`` in video files in one code line. It relies on ``matplotlib`` and ``ffmpeg``. If you're not installing ``mapflow`` from conda-forge, make sure ``ffmpeg`` is installed on your system.

Installation
------------

.. code-block:: bash

   pip install mapflow

Or:

.. code-block:: bash

   conda install -c conda-forge -y mapflow

Simple usage
------------

.. code-block:: python

   import xarray as xr
   from mapflow import animate

   ds = xr.tutorial.open_dataset("era5-2mt-2019-03-uk.grib")
   animate(da=ds['t2m'].isel(time=slice(120)), path='animation.mp4')

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
