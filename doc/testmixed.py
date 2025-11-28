import dune.fem
from dune.grid import structuredGrid
from dune.alugrid import aluSimplexGrid
from dune.fem.space import raviartThomas, bdm, lagrange, dgonb
from ufl import TrialFunction, TestFunction, dx, inner
from dune.fem.operator import galerkin

vertices  = [ (0,0), (0,1), (1,1), (1,0), (0.5,0.5)]
triangles = [ [0,1,4], [1,2,4], [2,3,4], [3,0,4] ]
gridView  = aluSimplexGrid({"vertices":vertices, "simplices":triangles})
# space = lagrange(gridView,order = 1)
# space = dgonb(gridView,order = 1)
space = raviartThomas(gridView,order = 1)
sigma     = TrialFunction(space)
tau       = TestFunction(space)
M         = galerkin(inner(sigma,tau)*dx).linear()
