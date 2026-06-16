==================
Since 2.11 release
==================

.. contents::
   :local:

Breaking changes
================

- The implementation of `CellDiameter` was corrected. Before it would behave 
  like `MaxCellEdgeLength` but now it's correctly computing the maximum distance
  between two points in an element. To keep the previous behavior, one should
  switch to `MaxCellEdgeLength`. Otherwise, use `dune.ufl.DuneCellDiameter`.

General changes
===============

Features
========

- UFLs `CellAvg` is now supported to compute the cell average of a quantity. 
- BDFM space for cubes was implemented. 
- RT0 for tetrahedrons was added.
- Eisenstat-Walker can now be configured by the user.

Bugfixes
========

- Unique face orientation for RT and BDM spaces changed to a simpler
  implementation.

