.. sectionauthor:: Andreas Dedner <a.s.dedner@warwick.ac.uk>, Robert Kl\ |oe|\ fkorn <robert dot klofkorn at math dot lu dot se>, Martin Nolte <nolte.mrtn@gmail.com>

######################################################
Discontinuous Galerkin Methods: the DUNE-FEM-DG Module
######################################################

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

.. todo:: dune-fem-dg examples need more details in notebooks

.. toctree::
   :maxdepth: 2

   euler_nb
   chemical_nb
