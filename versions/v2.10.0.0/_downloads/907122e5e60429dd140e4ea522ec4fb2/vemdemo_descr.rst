############################################
Virtual Element Methods: the DUNE-VEM module
############################################
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

.. toctree::
   :maxdepth: 3

   vemdemo_nb
   chimpl_nb
