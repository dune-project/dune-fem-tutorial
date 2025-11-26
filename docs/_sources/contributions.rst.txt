.. _contributing:

#############################
Contributing to this tutorial
#############################

Contributions are very welcomed - these can include but are not restricted to

* pointing out typos or suggesting places were the description should be improved
  (open an MR in https://gitlab.dune-project.org/dune-fem/dune-fempy
  or send us an email)

* showcasing what you are doing with the Python bindings for `dune`:
  we would like to showcase what user have used this package for,
  by adding sections to the :ref:`userprojects` chapter as described in the
  next section.

* providing an improvement or a small example:

  To make changes or add something to one of the existing scripts, please
  note that the notebooks are generated from the Python scripts using the
  package `jupytext`. So any changes should be done in the scripts. To check
  the effect of any change in the notebook simply install the Python packages
  `jupytext` (and `jupyter`) and after the change run

  .. code-block:: rst

    jupytext script.py --execute --set-kernel python3 --output script_nb.ipynb

  Then you can have a look at the notebook in the usual way.
  Once you are finished open a MR or send us the modified script - any help
  will certainly be appreciated!

  If you want to build the full tutorial yourself then you need to build
  this dune module using `dunecontrol` and then run `make doc` in the cmake
  build directory. The tutorial can then be viewed by loading
  `doc/html/index.html` into a web browser.
  Note that the notebooks used for the html tutorial are part of the
  repository so that the Python bindings are not needed to run `make doc`.
  Some Python packages are needed in addition to `sphinx`:
  `nbsphinx, nbconvert, ipython, jupytext,  sphinx_rtd_theme`.

  Rebuilding the notebook is done in the `doc` subfolder using standard
  `make`, e.g., after a change to `concepts.py` run
  ```
  cd doc
  touch concepts.py
  make concepts_nb.ipynb
  cd ../build-cmake
  make doc
  ```
  All packages needed to rebuild the full tutorial (which takes quite some
  time) are listed in the file `reqCI.txt` in the main folder and can be
  installed using
  ```
  pip install -r reqCI.txt
  ```

  Adding a new notebook involves the following steps

  * adding the script with the markdown/python code into the `demos` folder
  * adding a link to the script in the `doc` folder
  * adding the `script_nb.ipynb` file to `doc/Makefile` and both the script
    and the notebook to `CMakeLists.txt`.
  * adding the notebook to one of the `rst` files, e.g., `doc/furtherexamples.rst`
  * running `make script_nb.ipynb` in the `doc` folder (not in the build directory).
  * running `make doc` in the cmake build directory

The script in `demos`, the link to that in `doc` and the notebook
generated in `doc` by running `make script_nb.ipynb` should be added to the repository. 

The above assumes that you starting from a new script - if you already have
a notebook you would like to add then you can use

```
jupytext --to py:percent notebook.ipynb
```

to generate the script in the correct format and then the above can be
applied.

#############################
How to showcase your own work
#############################

To provide an example of what you have been doing with this package,
send us a file *projectname_descr.rst* with a short
description of your project with the authors and links to the project web
page, journal article and so on. This could look like this:

.. code-block:: rst

   ########################################################
   *dune-vem*: implementation of the virtual element method
   ########################################################

   .. sectionauthor:: Andreas Dedner <a.s.dedner@warwick.ac.uk>, Martin Nolte <nolte.mrtn@gmail.com>

   This module is based on DUNE-FEM
   (https://gitlab.dune-project.org/dune-fem/dune-fem)
   and provides implementation for the Virtual Element Method.
   The code is available in the module
   https://gitlab.dune-project.org/dune-fem/dune-vem.

A python script *myproject.py* showcasing your project can be provided as
well. But it should run with a standard set of Python packages and Dune
modules - and not take ages to run. For longer running example simply link
to results on your project webpage.

The script can contain both *markdown* parts for
additional descriptions between python code blocks similar to
a jupyter notebook
(see e.g. the :download:`script<vemdemo.py>` for the virtual element project).
This file will be translated into a :download:`jupyter notebook<vemdemo_nb.ipynb>`
which will also be made available for download. The python module *jupytext* and the sphinx
extension *nbsphinx* is used to generate the restructured text file used in the
documentation. The syntax for combining markdown cells and python code
is straightforward:

.. code-block::

   # %% [markdown]
   # # Laplace problem
   #
   # We first consider a simple Laplace problem with Dirichlet boundary conditions
   # \begin{align*}
   #   -\Delta u &= f, && \text{in } \Omega, \\
   #           u &= g, && \text{on } \partial\Omega,
   # \end{align*}
   # First some setup code:

   # %%
   import dune.vem
   from dune.grid import cartesianDomain, gridFunction
   from dune.vem import voronoiCells
   from ufl import *
   import dune.ufl

   # %% [markdown]
   # Now we define the model starting with the exact solution:

   # %%
   uflSpace = dune.ufl.Space(2, dimRange=1)
   x = SpatialCoordinate(uflSpace)
   exact = as_vector( [x[0]*x[1] * cos(pi*x[0]*x[1])] )

   # next the bilinear form
   u = TrialFunction(uflSpace)
   v = TestFunction(uflSpace)
   a = (inner(grad(u),grad(v))) * dx

If you have any questions or something in unclear let us know!
