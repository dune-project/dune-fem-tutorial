import matplotlib.pyplot as plt
import numpy as np
from ufl import *
from dune.alugrid import aluSimplexGrid
from dune.fem.space import lagrange
from dune.fem import assemble
from scipy.sparse.linalg import eigs
from dune.fem.operator import galerkin
from dune.ufl import DirichletBC
from dune.ufl import Constant
import scipy.sparse as sps

vertices = np.array( [(0,0), (1,0), (1,1), (0,1),(0.5,0.5)] )
triangles = np.array( [(0,1,4), (1,2,4), (2,4,3), (3,4,0)] )
gridView = aluSimplexGrid({"vertices": vertices, "simplices": triangles})
gridView.hierarchicalGrid.globalRefine(4)
space = lagrange(gridView, order=1)
u = TrialFunction(space)
v = TestFunction(space)
A = assemble( inner(grad(u),grad(v))*dx ).as_numpy
M = assemble( inner(u,v)*dx ).as_numpy
num_sols = 10
eigvals, eigvecs = eigs(A, M = M, sigma = 0, k = num_sols)
eigvals.real
pairs = [ n**2 + m**2 for n in range(num_sols) for m in range(num_sols)]
pairs.sort()
reference_eigvals = [np.pi**2 * pair for pair in pairs[:num_sols] ]
reference_eigvals
solutions = [space.function(name = "eigenvalue"+str(i)) for i in range(num_sols)]
for k in range(num_sols):
    solutions[k].as_numpy[:] = eigvecs[:,k].real
# gridView.writeVTK( "Neumann_Eigenvalues", pointdata = solutions )
rows, cols = num_sols//3, 3
fig, axs = plt.subplots(rows, cols, figsize=(10,10))
for k in range(rows):
    for l in range(cols):
        if k*cols+l < len(solutions):
            solutions[k*cols+l].plot( figure=(fig, axs[k][l]) )
MassMatrix = M
dbc = DirichletBC(space,0)
LHSOp = galerkin([inner(grad(u),grad(v))*dx, dbc])
A = LHSOp.linear().as_numpy
M = assemble([inner(u,v)*dx, dbc]).as_numpy
M = M - assemble([Constant(0)*inner(u,v)*dx, dbc]).as_numpy
M = assemble([inner(u,v)*dx, dbc]).as_numpy
dirichlet_zero_dofs = LHSOp.dirichletIndices()
N = M.shape[1]
chi_interior = np.ones(N)
chi_interior[dirichlet_zero_dofs] = 0.0
D = sps.spdiags(chi_interior, [0], N, N).tocsr()
M = D * M * D
eigvals, eigvecs = eigs(A, M = M, sigma = 0, k = num_sols)
solutions = [space.function(name = "eigenvalue"+str(i)) for i in range(num_sols)]
eigvals.real
pairs = [ n**2 + m**2 for n in range(1,num_sols+1) for m in range(1,num_sols+1)]
pairs.sort()
reference_eigvals = [np.pi**2 * pair for pair in pairs[:num_sols] ]
reference_eigvals
for k in range(num_sols):
    solutions[k].as_numpy[:] = eigvecs[:,k].real
# gridView.writeVTK( "Dirichlet_Eigenvalues", pointdata = solutions )
rows, cols = num_sols//3, 3
fig, axs = plt.subplots(rows, cols, figsize=(10,10))
for k in range(rows):
    for l in range(cols):
        if k*cols+l < len(solutions):
            solutions[k*cols+l].plot( figure=(fig, axs[k][l]) )