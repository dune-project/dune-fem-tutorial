#############################################
Mixed-dimensional PDEs: the Dune-MMesh module
#############################################
:download:`(notebook) <mmesh_nb.ipynb>` :download:`(script) <mmesh.py>`

.. .. sectionauthor:: Samuel Burbulla (samuel.burbulla@mathematik.uni-stuttgart.de)

`Dune-MMesh <github.com/samuelburbulla/dune-mmesh>`_ is a DUNE grid implementation tailored for numerical applications with physical interfaces.
The package wraps `CGAL <https://www.cgal.org>`_ triangulations and exports a predefined set of facets as a separate interface grid.
In combination with DUNE-FEM, it is well-suited for the numerical discretization of mixed-dimensional systems of partial differential equations.

For more details, see `documentation`_ and `publication`_.

.. _documentation: https://dune-mmesh.readthedocs.io/en/latest/

.. _publication: https://joss.theoj.org/papers/10.21105/joss.03959

.. toctree::
   :maxdepth: 3

   mmesh_nb
