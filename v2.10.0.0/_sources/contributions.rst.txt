.. _contributing:

#################################
How to showcase your own project
#################################

Contributions are very welcomed - this can include pointing out typos,
suggestion places were the description should be improved, providing this
improvement, and so on. In addition we would like to showcase what user
have used this package for, by adding sections to the :ref:`userprojects` chapter:

To do this provide us with a file *projectname_descr.rst* with a short
description of your project with the authors and links to the project web
page, journal article and so on.
The following is an example taken from the description file of the
:download:`vem project<vemdemo_descr.rst>`

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

Your python script *myproject.py* showcasing your project can contain both *markdown* parts for
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
