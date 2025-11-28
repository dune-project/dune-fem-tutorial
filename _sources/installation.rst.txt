#########
From PyPi
#########

.. _installation:

.. warning::

   Make sure that the installation folder name and any parent folder names **do not
   contain white spaces** since this will break the installation at various
   points. In general white spaces in filenames and folder names should be
   avoided.

By far the easiest way to get access to `Dune` is by installing into a
virtual environment using `pip`. In some folder e.g. `Dune` setup a virtual
environment and activate it

.. code-block:: bash

  python3 -m venv dune-env  
  source dune-env/bin/activate

Then download and build `Dune` (note that this takes some time to have
coffee/tea at hand):

.. code-block:: bash

  pip install --pre dune-fem

.. note:: This tutorial is based on the upcoming 2.11 version of Dune.
   Older release versions are available in the Python package index and can be
   obtained by removing the `--pre` from the above install command. Note
   that a few of the features described in this tutorial will not be
   available and that the API has changed in some places. See the
   description in the :doc:`changelog<changelog>` for details.

****************************
Additional external packages
****************************

To use additional external packages like `petsc` or `umfpack` which are not
in a default location, the cmake prefix variable can be set during
installation process:

.. code-block:: bash

  CMAKE_FLAGS="-DCMAKE_PREFIX_PATH=PATHLIST" pip install --pre dune-fem

where `PATHLIST` is a semicolon separated list of path containing external packages.

************************
Testing the installation
************************

.. important::

   The current installation requires MPI to be available and requires the
   additional installation of `mpi4py`:

   .. code-block:: bash

      pip install mpi4py

.. note::

   The first time you construct an object of a specific realization of one
   of the Dune interfaces (e.g. here a structured grid),
   the just in time compiler needs to be invoked. This can take quite some
   time - especially for grid realizations. This needs to be done only once
   so rerunning the above code a second time (even using other parameters
   in the `structuredGrid` function) should execute almost instantaniously.

To test that everything works you can download all the scripts described
in this tutorial and try them out

.. code-block:: bash

  python -m dune.fem
  cd fem_tutorial
  python concepts.py

All examples are available as Python scripts or IPython notebooks - for
the later `jupyter` is needed

.. code-block:: bash

  pip install jupyterlab
  jupyter lab

This has been tested with different Linux distributions, MacOS, and using
the *Windows Subsystem for Linux*.


*****************
Hints for Windows
*****************

Some hints on getting Dune to work using the
*Windows Subsystem for Linux* (tested on Windows 11).
Installation in three steps:

1. First we need to `install the wsl`_ (here an `Ubuntu` version):

   Open ``PowerShell`` as administrator and run ``wsl --install``
   (find Windows ``PowerShell`` and right click;
   the first entry should be ``Run as administrator``).
   This step takes quite some time ('get a coffee' long).
   Close the ``PowerShell`` again (enter ``exit``).
   Possibly one needs to restart after this step (second coffee).

2. We need to add some packages and setup a Python virtual environment.
   Open the wsl (Pinguin icon) - again some installation is done.
   Enter a new username and password.
   Then run the following commands:

   .. code-block:: bash

      sudo apt update  # enter the password you used above 
      sudo apt install --reinstall ca-certificates
      sudo apt install python3-dev python3-pip python3-venv cmake
      sudo apt install jupyter-core
      python3 -m venv dune-env
      source dune-env/bin/activate
      pip install jupyterlab

3. We have reached the Dune specific part of the installation as described
above in a lot more detail (I would suggest some tea at this stage)

   .. code-block:: bash

      pip install --pre dune-fem
      python -m dune.fem

   The last step downloads the tutorial scripts into the folder
   ``fem_tutorial``.

Each time to open the Linux terminal (wsl) again you will need to
run the following commands:

.. code-block:: bash

  source ~/dune-env/bin/activate

To work on one of the scripts from the tutorial you can either use
``jupyter-lab``

.. code-block:: bash

  cd ~/fem_tutorial/
  jupyter lab &

then open the given link in your favourite web browser.

Instead of using the notebooks you can also run the python scripts from
the command line, e.g., run

.. code-block:: bash

  python concepts.py

in the ``fem_tutorial`` folder.

.. _install the wsl: https://learn.microsoft.com/en-us/windows/wsl/setup/environment

###########
From Source
###########

.. note::

   We strongly encourage the use of a python virtual environment and the
   following instructions are written assuming that a virtual environment is
   activated.

************
Requirements
************

The following dependencies are needed for Dune-Fem python binding:

* At least C++17 compatible C++ compiler (e.g. g++ 9 or later)
* python (3.7 or later)

* **Required** Dune modules (release 2.9 or later)

  * dune-common (https://gitlab.dune-project.org/core/dune-common.git)
  * dune-geometry (https://gitlab.dune-project.org/core/dune-geometry.git)
  * dune-grid (https://gitlab.dune-project.org/core/dune-grid.git)
  * dune-istl (https://gitlab.dune-project.org/core/dune-istl.git)
  * dune-localfunctions (https://gitlab.dune-project.org/core/dune-localfunctions.git)
  * dune-alugrid  (https://gitlab.dune-project.org/extensions/dune-alugrid.git)
  * dune-fem (https://gitlab.dune-project.org/dune-fem/dune-fem.git)


* **Optional** Dune modules (release 2.9 or later)
   
  * dune-spgrid (https://gitlab.dune-project.org/extensions/dune-spgrid.git)
  * dune-polygongrid (https://gitlab.dune-project.org/extensions/dune-polygongrid.git)
  * dune-fem-dg (https://gitlab.dune-project.org/dune-fem/dune-fem-dg.git)
  * dune-vem (https://gitlab.dune-project.org/dune-fem/dune-vem.git)

The optional Dune modules are only need for the parts of the tutorial discussing extension modules.
  
  
**********************************
Building the Required Dune Modules
**********************************

Read the instructions on how to `build Dune with Python support`_ which also
links to general instructions on how to `build Dune modules`_.

.. _build Dune modules: https://dune-project.org/doc/installation
.. _build Dune with Python support: https://dune-project.org/doc/installation/installation-pythonbindings/
.. _dune-fem-dg: https://gitlab.dune-project.org/dune-fem/dune-fem-dg
.. _example script: https://gitlab.dune-project.org/dune-fem/dune-fem-dg/-/blob/master/scripts/build-dune-fem-dg.sh?ref_type=heads
   
The `dune-fem-dg`_ module offers an `example script`_ to build all required modules from source. 

Test your completed installation by opening a Python terminal and running

.. code:: python

   import math
   from dune.grid import structuredGrid
   from dune.fem.function import gridFunction
   grid = structuredGrid([0,0],[1,1],[10,10])
   @gridFunction(grid,name="test",order=2)
   def f(x):
      return math.sin(x.two_norm*2*math.pi)
   f.plot()

If you have everything set up correctly (and have `matplotlib`) you should
get a colored figure and are hopefully ready to go...

###############
Troubleshooting
###############

* **Issue with C++ compiler:**
  If the gnu compiler is used, version needs to be 7 or later. This can be checked in terminal with

  .. code-block:: bash

    g++ --version

  If your version is out of date, you will need to upgrade your system to use Dune

* **A terminated compilation:**
  If you get a `CompileError` exception with an output of the form:

  .. code-block:: bash

    raise CompileError(buffer_to_str(stderr))
      dune.generator.exceptions.CompileError: c++: fatal error: Killed signal
      terminated program cc1plus
      compilation terminated.

  then this could be caused by the compiler running out of memory.
  This can especially happen on a HPC cluster where memory can be
  restricted. An option is to run a small part of the simulation on
  a single core with a lot of memory so that all modules are compiled
  before starting the production run which might require less memory
  than the compilation process.

* **Python version:**
  It is possible that the python version may be an issue. The scripts
  require `python3` including the development package being installed.
  If during the Dune installation you get the error

  .. code-block:: none

    fatal error: pyconfig.h: No such file or directory

  This can probably be fixed by installing additional python3 libraries with e.g.

  .. code-block:: bash

    sudo apt-get install libpython3-dev

* **MPI not found:**
  One other problem is that a default version of Open MPI may already be installed.
  This will lead to errors where Dune appears to be looking in the wrong directory for Open MPI
  (e.g. `usr/lib/openmpi` instead of the home directory where the script installs it).
  This can be solved by running

  .. code-block:: bash

    make uninstall

  in the original MPI install directory, followed by removing the folder. It will then be necessary to reinstall Open MPI and Dune. It may also be necessary to direct mpi4py to the new MPI installation. It is possible to check whether this is a problem by running python and trying out 

  .. code-block:: python

    from mpi4py import MPI

  If it comes up with an error, this can be fixed by installing mpi4py manually using the following commands

  .. code-block:: bash

    git clone https://bitbucket.org/mpi4py/mpi4py.git
    cd mpi4py
    python setup.py build --mpicc=/path/to/openmpi/bin/mpicc
    python setup.py install --user

* If you get a

  .. code-block:: bash

    RuntimeError: ParallelError [Communication:dune-env/include/dune/common/parallel/mpicommunication.hh:???]:
    You must call MPIHelper::instance(argc,argv) in your main() function before using the MPI Communication!

  then the system did not correctly detect the missing `mpi4py` package -
  you need to install it using e.g.

  .. code-block:: bash

    pip install mpi4py

* **User warning from numpy:**

  .. code-block:: none

    UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.

  This is caused by some shared library using the compiler flag
  ``fast-math``.
  Check for example that you are not using this flag in your cmake setup for Dune.
  See `here <https://moyix.blogspot.com/2022/09/someones-been-messing-with-my-subnormals.html/>`_
  for a detailed description.

* **Using petsc and petsc4py:**
  It is important the all `dune` modules and `petsc4py` use the same
  version of `petsc`. An runtime exception will be thrown if some
  incompatibility is detected. The simplest way to avoid any issues is to
  set the `PETSC_DIR` environment variable during the building, e.g., `pip`
  installation of the `dune` packages and running the scripts. 
  The following gives an example of how to first install `petsc` and
  `petsc4py` into a virtual environment and then install `dune-fem`.
  Assuming that the virtual environment with `python` version `X.Y`
  is in `$HOME/dune-env` and we want a few extra `petsc` packages we could run:

  .. code-block:: bash

   pip install mpi4py
   PETSC_CONFIGURE_OPTIONS="--download-hypre --download-parmetis --download-ml --download-metis --download-scalapack" pip install petsc
   export PETSC_DIR=$HOME/dune-env/lib/pythonX.Y/site-packages/petsc
   pip install petsc4py
   pip install --pre dune-fem

  After this the `PETSC_DIR` environment variable should not be required anymore.
  This is all a bit experimental so let us know if it is not working.
  Of course an installed `petsc` or build from source should work in the
  same way, without the need for the `pip install petsc` line, just make
  sure to set the `PETSC_DIR` during the installation of `dune-fem`.

* **Newly installed software is not used:**
  After for example adding `petsc` to your system one needs to remove an existing `dune-py` module which
  contains the jit compiled modules. New software components are not
  automatically picked up. One can run

  .. code-block:: bash

    python -m dune info

  to find the location of the `dune-py` folder. That folder needs to be
  removed before the new component can be used.

  If issues remain reinstalling all `dune` packages might be required -
  including the cached wheels:

  .. code-block:: bash

    pip uninstall -y `pip freeze | grep dune`
    pip cache remove "dune*"

  When running `pip install` for the `dune` packages setting prefix path
  for cmake as described above might be necessary if the
  new package is not in a default location.

* **Output to terminal seems a bit random:**
  the issue is (probably) that
  Python buffers its `print` output and C++ does not. So if in a mixed
  program both are writing to the terminal (or piped into a file) the C++
  output often appears before the Python output. This can be fixed by (i)
  adding `flush=True` to the Python print statements or setting the
  environment variable `PYTHONUNBUFFERED` to some non zero value.
