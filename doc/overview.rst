############################
An Overview of this Document
############################

This document tries to describe the main concepts needed to get a new user
started on solving complex partial differential equations using the
`Dune-Fem`_ python bindings. Some parts are still quite raw and we would
greatly appreciate any help improving this document. In addition if you
would like to promote your own work then please upload a script showcasing
some simulation you did based on this package - it is possibly as easy as
providing us with a Python script. More details on how to help
us improve and extend this document see the section on :ref:`contributing`.

The Python bindings are based on `pybind11`_ :cite:`Pybind11-Paper` and a detailed
description on how we export the polymorphic interfaces in `Dune`_ is
provided in :cite:`Dune-Python-Paper` which describes the Dune module
`Dune-Python`_ which after the Dune 2.7 release has been directly
integrated into the Dune core modules. If you have a version newer than 2.8
then note that the `Dune-Python`_ module is obsolete.

Here is a quick summary of the different parts of this document - all of
the code is available for download in form of both a Notebook or a script.
Links to either are included on each page. To get the full tutorial (both
scripts and notebooks) you can also run (after installing `dune-fem`)

.. code-block:: bash

  python -m dune.fem

which will download everything into a subfolder `fem_tutorial`.
Or the script can be obtained by cloning the git repository
https://gitlab.dune-project.org/dune-fem/dune-fempy
where the scripts can be found in the demo folder.

If you are already familiar with the main concepts or want to get an idea
of the range of this package take a look at our :ref:`gallery` page where we

#. build upon the general concepts described in this tutorial to solve a
   range of more complex PDEs.
   These example are hopefully useful starting point for new projects.
#. describe two larger extension modules build on top of dune-fem which
   provide Python bindings: `Dune-Fem-Dg`_ and `Dune-Vem`_ which focus on Discontinuous Galerkin methods for
   advection-diffusion (especially advection dominated problems) and the
   implementations of the virtual element method, respectively.

The main body of this tutorial then contains the following:

#. First some remarks on getting the package to work: the simplest
   approach is to use the Python package index (pip) to install the software
   into a new virtual environment.
   Working from the git sources is a good option for more advanced users
   who want to change the underlying Dune C++ source files.
#. A scalar Laplace problem and a simple scalar, non linear time dependent partial
   differential equation are used to describe the basic concepts.
   This leads through the steps required to set up the problem, solve the system of equations,
   and visualize the results.
#. After the introduction to the underlying concepts and how to get a
   simple problem up and running, we discuss:

   * how to add more complicated boundary conditions to the problem formulation.
   * how to use different solver backends (including build in solvers, solvers and
     preconditioners available in `Dune-Istl`_, `scipy`_, `PETSc`_ and
     also `petsc4py`_ see also :cite:`PETSc-Paper,SciPy-Paper`).
   * how to enable either multithreading or MPI parallelization for your problem
   * how to backup and restore data including discrete functions and the
     grid hierarchy as is useful for example to checkpoint a simulation.
   * more details on constructing a grid using different formats including for
     example the Dune Grid Format (`DGF`_) or `gmsh`_ but also using simple python structures like
     dictionaries for describing general unstructured grids is available.
     Results can be plotted using `matplotlib`_ (for 2d) and for example
     `mayavi`_ (for 2d and 3d). For more advanced plotting options data can
     be exported using `vtk` format and then analysed for example using
     `paraview`_. All of this is demonstrated in the provided examples.
   * discuss the grid interface in more details, i.e., how to iterate over
     the grid, access geometrical information, and attach data to elements
     of the grid.
#. We then discuss some further topics:

   * Local grid refinement and coarsening is a central feature of
     `Dune-Fem`_. Here we show how to use it for stationary and
     time dependent problems. Grid adaptivity makes use of special grid views.
   * Other views are also
     available, one of these can be used to deform the grid given an
     arbitrary (discrete) vector field. This is used to compute the evolution
     of a surface under mean curvature flow.
   * We complete our discussion by demonstrating how to straightforwardly
     extend the functionality of the package and
     improve performance by important additional C++ algorithms and classes.
#. Finally other projects are presented some of them developed by
   the authors of this document, some contributed by other
   users. If you have used this package then we would like to
   hear about it and would ask you to contribute to this
   chapter. Doing so is quite easy (see :ref:`contributing` for details).


.. _mayavi: https://docs.enthought.com/mayavi/mayavi
.. _paraview: https://www.paraview.org
.. _matplotlib: https://matplotlib.org
.. _gmsh: https://pypi.org/project/pygmsh
.. _DGF: https://dune-project.org/doxygen/master/group__DuneGridFormatParser.html#details
.. _tutorial compatible with the 2.7 release: https://dune-project.org/sphinx/content/sphinx/dune-fem-2.7/
.. _scipy: https://www.scipy.org
.. _PETSc: https://www.mcs.anl.gov/petsc
.. _petsc4py: https://bitbucket.org/petsc/petsc4py
.. _docker: https://www.docker.com
.. _pybind11: https://github.com/pybind/pybind11
.. _Dune: https://www.dune-project.org
.. _Dune-Python: https://www.dune-project.org/modules/dune-python/
.. _Dune-Istl: https://www.dune-project.org/modules/dune-istl/
.. _Dune-Fem: https://www.dune-project.org/modules/dune-fem/
.. _Dune-Vem: https://www.dune-project.org/modules/dune-vem/
.. _Dune-Fem-Dg: https://www.dune-project.org/modules/dune-fem-dg/
.. _UFL: https://bitbucket.org/fenics-project/ufl
.. _RefElem: https://dune-project.org/doxygen/master/group__GeometryReferenceElements.html#details
.. _NumtourCFD: https://numtourcfd.pages.math.cnrs.fr/doc/NumtourCFD/v1.0.1-alpha.2/applications/validation/bercovier-engelman.html
