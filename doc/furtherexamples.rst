.. _gallery:

############
Main Gallery
############

.. nbgallery::

   discontinuousgalerkin_nb
   elasticity_nb
   spiral_nb
   wave_nb
   stokes
   cylinder_nb
   mixed_poisson_nb
   evalues_laplace_nb
   biharmonic_IPC0_nb
   crystal_nb
   mcf_nb
   laplace-dwr_nb

#################
Extension Modules
#################

-----------------------------------------------
Discontinuous Galerkin Methods with DUNE-FEM-DG
-----------------------------------------------

The add on module `Dune-Fem-Dg`_ provides a number of DG algorithms focusing
on solving systems of evolution equations of advection-diffusion-reaction type of the form

.. math::
  \partial_t U + \nabla\cdot\Big( F_c(x,t,U) - F_v(x,t,U,\nabla U) \Big) = S(U).


The implemented method include a wide range of methods for DG
discretization of the diffusion term including `CDG2`, `BR2`, `IP`, and many
others. The advection term can be discretized using a `local Lax-Friedrichs`
flux, specialized fluxes e.g. `HLLE` for the Euler equations, or user defined
fluxes. To stabilize the DG method for advection dominated problems
limiters with troubled cell indicators are available. Finally we use a
method of lines approach for the time stepping based on explicit, implicit
or, `IMEX` Runge-Kutta schemes using matrix-free `Newton-Krylov` solvers. As a
final note the module can also be used to solve first order hyperbolic
problems using a Finite-Volume method with linear reconstruction.
A detailed description of the module is found in the :cite:`DuneFemDG`.

.. _Dune-Fem-Dg: https://gitlab.dune-project.org/dune-fem/dune-fem-dg

.. nbgallery::

   euler_nb
   chemical_nb


-------------------------------------
Virtual Element Methods with DUNE-VEM
-------------------------------------

:download:`demo notebook <vemdemo_nb.ipynb>` :download:`demo script <vemdemo.py>`
:download:`Cahn-Hilliard notebook <vemdemo_nb.ipynb>` :download:`Cahn-Hilliard script <vemdemo.py>`

.. .. sectionauthor:: Andrea Cnagiani, Andreas Dedner <a.s.dedner@warwick.ac.uk>, Martin Nolte <nolte.mrtn@gmail.com>

This module is based on `dune-fem <https://gitlab.dune-project.org/dune-fem/dune-fem>`_
and provides implementation for the Virtual Element Method.
You can install the package from Pypi by running :code:`pip install dune-vem`.
The sources for
the methods is available in the `dune-vem git repository <https://gitlab.dune-project.org/dune-fem/dune-vem>`_.
See our `publication`_ for details on our approach to add virtual element
spaces to existing finite element software frameworks.
The examples from the paper can be reproduced using the scripts collected in
https://gitlab.dune-project.org/dune-fem/dune-vem-paper.
For a focus on the analysis and testing of spaces for forth order problems have a look `here`_.

.. _here: https://academic.oup.com/imajna/advance-article-abstract/doi/10.1093/imanum/drab003/6174313?redirectedFrom=fulltext

.. _publication: https://arxiv.org/abs/2208.08978

.. nbgallery::

   vemdemo_nb
   chimpl_nb

.. todo:: Give some details on the Uzawa algorithm, add dune-fempy version and an algorithm implementation
.. todo:: dune-fem-dg examples need more details in notebooks
.. todo:: add an example using dolfin-dg and the model2ufl functions
