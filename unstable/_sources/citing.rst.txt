###################
Citing this project
###################

If you found this tutorial helpful for getting your own projects up and
running please cite this project:

Title: Python Bindings for the DUNE-FEM module
*Authors: Andreas Dedner, Martin Nolte, and Robert Klöfkorn*
Publisher: Zenodoo, 2020
DOI 10.5281/zenodo.3706994


Note that the `Dune`_ project consists of a number of different modules which can be
used through the Python interface described in this tutorial. Many people
have worked and are working on this software package. If you have used it
in your work then all of those contributing to the project past and present
would appreciate an acknowledgment of their work.
To make finding the right entries to cite easy you can add

.. code:: python

  import dune.citationfile

to your script - this will generate a bibtex file

.. code:: bash

  myscript.py.bib

(assuming your script is named `myscript.py`) containing bibtex entries for
the Dune modules that were imported in the script. Alternatively, add

.. code:: python

  import dune.citations

to get the list printed to the terminal. A full list of citations for all
installed Dune modules can be obtained by running

.. code:: python

  python -m dune citations [--file filename]


.. _Dune: https://www.dune-project.org
