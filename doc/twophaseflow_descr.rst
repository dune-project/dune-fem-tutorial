###############################################
HP adaptive DG scheme for twophase flow problem
###############################################
:download:`(notebook) <twophaseflow_nb.ipynb>` :download:`(script)<twophaseflow.py>`

.. sectionauthor:: Andreas Dedner <a.s.dedner@warwick.ac.uk>, Birane Kane <birane.kane@ians.uni-stuttgart.de>, Robert Kl\ |oe|\ fkorn <robert dot klofkorn at math dot lu dot se>, Martin Nolte <nolte.mrtn@gmail.com>


We solve a two phase flow model using an hp adaptive higher order
discontinuous Galerkin scheme :cite:`TwoPhaseFlow`.

This examples presents a framework for solving two-phase flow problems in porous media. 
The discretization is based on a Discontinuous Galerkin method and includes local
grid adaptivity and local choice of polynomial degree. The method is implemented using
the Python frontend described here. The example contains a number of time stepping 
approaches ranging from a classical IMPES method to a fully coupled implicit scheme. 
The implementation of the discretization is very flexible allowing to test different 
formulations of the two-phase flow model and adaptation strategies. For further
details we refer to :cite:`TwoPhaseFlow`.

.. toctree::
   :maxdepth: 3

   twophaseflow_nb
