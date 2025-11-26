#!/bin/bash

TUTORIALWEB=https://dune-project.github.io/dune-fem-tutorial/

VERSIONS=
while read arg; do
  VERSIONS="$VERSIONS $arg"
done

echo "Versions are $VERSIONS"
echo "##########################
DUNE-FEM Tutorial Versions
##########################

Previous versions of the DUNE-FEM tutorial:
" > versions.rst

# write bullet points
for VER in $VERSIONS; do
  echo "* \`$VER\`_" >> versions.rst
done

echo "" >> versions.rst

# write links
for VER in $VERSIONS; do
  echo ".. _\`$VER\`: $TUTORIALWEB/$VER" >> versions.rst
done
