####################################
DDFEM: Diffuse Domain Python package
####################################

.. sectionauthor:: Andreas Dedner <a.s.dedner@warwick.ac.uk>

Based on `UFL`_ and the Python bindings discussed in this tutorial we have
developed a package implementing a wide range of Diffuse Domain Methods to
solve PDEs on complex domains. The methods avoids constructing a grid for
the computational domain $\Omega$ by embedding it in a simpler domain,
e.g., a square. The original domain is described by providing a signed
distance function. This is used to modify the PDE so that the original
problem is solved within $\Omega$ maintaining the correct boundary
conditions. DDFEM provides a number of ways to construct the signed
distance function and to transform the PDE.
The package is available on `pypi`_ and there is a
`tutorial`_ with a detailed description of the package and on how to use
and extend it.
The derivation of the methods and details on the packages can be found here
:cite:`DDFEMMethod-Paper,DDFEMImpl-Paper`

.. _UFL: https://bitbucket.org/fenics-project/ufl
.. _pypi: https://pypi.org/project/ddfem/
.. _tutorial: https://ddfem.readthedocs.io/en/latest/index.html
