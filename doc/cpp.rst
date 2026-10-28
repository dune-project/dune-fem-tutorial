.. sectionauthor:: Andreas Dedner <a.s.dedner@warwick.ac.uk>, Robert Kl\ |oe|\ fkorn <robert dot klofkorn at math dot lu dot se>, Martin Nolte <nolte.mrtn@gmail.com>

.. _algorithms:

#######################
Using C++ Code Snippets
#######################

.. index::
   triple: Algorithm; C++; User defined code

In this section we demonstrate how it possible to use small pieces of C++
code to either extend the existing functionality or improve the efficiency
of the code by moving code from Python to C++.
We will first describe how to define grid functions based on some
simple C++ code.
We will then move parts of an algorithm from
Python to C++. The DUNE interfaces exported to
Python are very close to their C++ counterpart so that rapid prototyping of
new algorithms can be carried out using Python and then easily moved to
C++. This will be demonstrated in the following examples:

Debugging the C++ snippets is not always straightforward see the later
section for some :ref:`hints for developers</developers.rst>`

.. toctree::
   :maxdepth: 2

   cppfunctions_nb
   lineplot_nb
   mcf-algorithm_nb
   laplace-dwr-algorithm_nb
