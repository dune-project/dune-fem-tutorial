import matplotlib.pyplot as plt
import numpy as np
from dune.grid import yaspGrid, cartesianDomain
from dune.alugrid import aluSimplexGrid
from dune.fem.space import dgonbhp, dglegendrehp, raviartThomas, bdm, bdfm
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 div, dx, sin, cos, pi, inner, grad)
from dune.ufl import DirichletBC
from dune.fem import assemble, integrate
from dune.fem.function import gridFunction
from scipy.sparse.linalg import cg, splu, LinearOperator

def getGridSpace(element,space,order, dim=2):
    if element == "simplex" and dim == 2:
        vertices = [ (0,0), (0,1), (1,1), (1,0), (0.5,0.5)]
        triangles = [ [0,1,4], [1,2,4], [2,3,4], [3,0,4] ]
        gridView = aluSimplexGrid({"vertices":vertices, "simplices":triangles})
    else:
        domain = cartesianDomain([0]*dim, [1]*dim, [4-dim]*dim)
        if element == "simplex":
            gridView = aluSimplexGrid(domain)
        else:
            gridView = yaspGrid(domain)
    if space == "RT":
        spaceHDiv = raviartThomas(gridView,order = order)
    elif space == 'BDM':
        spaceHDiv = bdm(gridView,order = order+1)
    else:
        spaceHDiv = bdfm(gridView,order = order+1)
    maxOrder = 4 if dim == 2 else 1 # max polynomial order for DG space
    if element == "simplex" or not space == "RT":
        spaceDG = dgonbhp(gridView, order = order, maxOrder = maxOrder)
    else:
        spaceDG = dglegendrehp(gridView, order = order, maxOrder = maxOrder)
    print(f"{mth} ({gridView.type}) order = {spaceHDiv.order}\n--------------------------------------",flush=True)
    return gridView, spaceHDiv, spaceDG
def getMatrices(spaceHDiv, spaceDG, dirichlet):
    dim = spaceHDiv.gridView.dimension
    x = SpatialCoordinate(spaceHDiv)
    f = sin if dirichlet else cos
    exSol = f(pi*x[0])*f(pi*x[1])
    if dim > 2:
        exSol *= f(pi*x[2])
    f = -div(grad(exSol))
    sigma = TrialFunction(spaceHDiv)
    tau   = TestFunction(spaceHDiv)
    u     = TrialFunction(spaceDG)
    v     = TestFunction(spaceDG)
    A     = inner(sigma,tau)*dx
    BT    = inner(u,div(tau))*dx
    B     = inner(div(sigma),v)*dx
    if not dirichlet:
        dbc = DirichletBC(spaceHDiv,grad(exSol))
    else:
        dbc = None
    rhs_b = f*v*dx
    return (assemble([A,dbc]).as_numpy,
            assemble([BT,dbc]).as_numpy,
            assemble(B).as_numpy,
            assemble(rhs_b).as_numpy,
            exSol)
def schur_op(B,BT,invA,x):
    y = BT @ x
    y2 = invA.solve(y)
    return B @ y2
def solveSys(invA,B,BT, target,rhs):
    schur_comp = LinearOperator( (target.shape[0],target.shape[0]),
                                 matvec = lambda x: schur_op(B,BT,invA,x) )
    target[:], info = cg(schur_comp, rhs, x0 = target)
def simulate(gridView, spaceHDiv, spaceDG, dirichlet):
    gradSol = spaceHDiv.function(name = "gradSol")
    sol = spaceDG.function(name = "sol")
    refs = 4
    if gridView.dimension <= 2: # we can only use myplot to plot 2d
        fig, axs = plt.subplots(1,refs, figsize=(10,10))
    print("     L^2         H^1         div",flush=True)
    for i in range(refs):
        gridView.hierarchicalGrid.globalRefine()
        A,BT,B,b,exSol = getMatrices(spaceHDiv, spaceDG, dirichlet)
        invA = splu(A.tocsc())
        solveSys(invA,B,BT, target = sol.as_numpy[:],rhs = b)
        gradSol.as_numpy[:] = invA.solve(-BT @ sol.as_numpy)
        err = np.sqrt( integrate([
                         (sol-exSol)**2,
                         inner(gradSol-grad(exSol),gradSol-grad(exSol)),
                         (div(gradSol-grad(exSol)))**2
                       ], order=8) )
        print(f"{err[0]:.6e} {err[1]:.6e} {err[2]:.6e}",flush=True)
        if gridView.dimension <= 2:
            gridFunction(div(gradSol)).plot( level=spaceDG.order+1, figure=(fig, axs[i]) )
    print("\n--------------------------------------",flush=True)
mth = 'RT'
element = 'cube'
for order in [0,1,2,3,4]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=False)
mth = 'BDM'
element = 'simplex'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order, dim=2)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=False)
mth = 'RT'
element = 'simplex'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)
mth = 'BDM'
element = 'cube'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)
mth = 'BDFM'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)
mth = 'RT'
element = 'simplex'
for order in [0]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order, dim=3)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)
element = 'cube'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order, dim=3)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)