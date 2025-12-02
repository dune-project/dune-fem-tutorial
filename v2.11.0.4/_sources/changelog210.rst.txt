==================
Since 2.10 release
==================

.. contents::
   :local:

Breaking changes
================

- With the switch to the newer UFL version there `components` method
  on the combined space clashes with the new property on the UFL spaces
  with the same name. This could not be fixed in a backward compatible way
  so that using `space.components` leads to an error. Replace
  `space.components` with `space.subSpaces`.

General changes
===============

Bugfixes
========

Pickling of discrete functions
------------------------------

- A bug was fixed that could cause loading a pickled discrete function
  (space) to segfault.
  There might still be an issue with the backend methods, e.g., `as_numpy`,
  to be not added to a discrete function after load. This has been fixed but
  requires the JIT module to be rebuild. If you encounter this issue run
  ```python
  python dune remove femspace
  ```
  and retry. Unfortunately the pickle files will also have to be regenerated...
