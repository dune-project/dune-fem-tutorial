Web page builder for DUNE-FEM tutorial
======================================

This repository build and deploys the [DUNE-FEM tutorial][tutoriallink].

Add tutorial versions
---------------------

To add a specific tutorial version push a branch with version tag as
name to this repository, i.e. a branch `v2.11.0.3`.
This branch should contain a directory `doc` that contains the tutorial
including the ipython notebooks.

Build a tutorial versions
-------------------------

To build and deploy the tutorial as web page run the workflow `dune-fem
tutorial` under [actions][actions] with the specific version to be built, i.e. `v2.11.0.3`.


[tutoriallink]: https://dune-project.github.io/dune-fem-tutorial
[actions]: https://github.com/dune-project/dune-fem-tutorial/actions
