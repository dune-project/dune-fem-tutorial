# %% [markdown]
#
# \DeclareMathOperator{\Div}{Div}
# \newcommand{\dx}{{\rm dx}}
# # Mixed method for the Poisson equation
# .. index:: Equations; Poisson (Mixed formulation)
#
# .. index:: Methods; Mixed FEM
#
# .. index:: Spaces; Raviart Thomas and BDM
#
# .. sectionauthor:: contribution by Philip Herbert <P.Herbert@sussex.ac.uk>
#
# We use Dune to solve a simple elliptic problem using a mixed method.
# This requires the use of $H(\Div)$ spaces.
#
# We are interested in the problem that
# $u \in H^1_0(\Omega)$ satisfies
# \begin{align*}
# -\Delta u &= f  && \text{in } \Omega,\\
# u &= 0     && \text{on } \partial \Omega.
# \end{align*}
# in variational form for some $f \in H^{-1}(\Omega)$.
#
# It may however be convenient to rewrite the condition $-\Delta u = f$ as $-\Div ( \nabla u ) = f$, and replace $\nabla u$ with $\sigma$.
# Since we wish to take divergence of $\sigma$, it makes sense to require that it is in $H(\Div)$.
# We thus end up with the following system of equations:
# \begin{align*}
# \Div \sigma &=-f  && \text{in } \Omega,\\
# \sigma - \nabla u &= 0  && \text{in } \Omega,\\
# u &= 0     && \text{on } \partial \Omega.
# \end{align*}
#
# Performing integration by parts and multiplying by appropriate test functions, we arrive at the problem to find $(\sigma,u) \in H(\Div,\Omega) \times L^2(\Omega)$ such that
# \begin{align*}
# \int_\Omega \left(\sigma \cdot \tau + u \Div \tau  + v \Div \sigma \right) \dx = \int_{\Omega} -f v\dx
# \end{align*}
# for all $(\tau, v) \in H(\Div,\Omega) \times L^2(\Omega)$.
# Note that in this formulation the condition
# $u = 0$ on $\partial \Omega$ is a \emph{natural} boundary condition.
# Other boundary conditions can be considered. For example Neumann
# boundary conditions need to be enforced through the correct choice of the
# space, i.e., are essential boundary conditions.
# So switching from $u=0$ on the boundary to $\nabla u\cdot n=0$ leads to the
# same variational formulation but over the spaces $L^2$ and
# \begin{align*}
# H_0(\Div,\Omega):=\{\sigma\in H(\Div,\Omega)\colon \sigma\cdot n=0\;\text{on}\; \partial\Omega \}~.
# \end{align*}

# One may verify that the system introduced above is well posed.
# We are left to choose an appropriate pair of spaces to approximate
# $H(\Div,\Omega) \times L^2(\Omega)$.
# One such example is the pair of Raviart-Thomas elements and DG functions
# of suitable order.

# %% [markdown]
# We start by setting up a grid (either a cube or a simplex grid) and the
# corresponding pair of spaces:

# %%
import matplotlib.pyplot as plt
import numpy as np

from dune.grid import yaspGrid, cartesianDomain
from dune.alugrid import aluSimplexGrid
# we use the hp spaces here to avoid re-compilation
from dune.fem.space import dgonbhp, dglegendrehp, raviartThomas, bdm, bdfm

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

    if element == "simplex" or not space == "RT":
        spaceDG = dgonbhp(gridView, order = order)
    else:
        spaceDG = dglegendrehp(gridView, order = order)
    print(f"{mth} ({gridView.type}) order = {spaceHDiv.order}\n--------------------------------------",flush=True)
    return gridView, spaceHDiv, spaceDG

# %% [markdown]
# Finally, we setup the problem, extracting the matrices used for solving
# the Shur complement problem:

# %%
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 div, dx, sin, cos, pi, inner, grad)
from dune.ufl import DirichletBC
from dune.fem import assemble, integrate
from dune.fem.function import gridFunction
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

# %% [markdown]
# The system we are interested in solving is a saddle point.
# To make this easier to solve, we use a Schur complement solver.
# Phrasing our problem in block matrix formation as
# \begin{equation} \begin{pmatrix} A & B^t\\ B & 0 \end{pmatrix} \begin{pmatrix} \sigma_h \\ v_h \end{pmatrix} = \begin{pmatrix} 0 \\ -f_h \end{pmatrix}, \end{equation}
# it holds that $\sigma_h = - A^{-1} B^t u_h$ and $B \sigma_h = -f_h$, hence $B A^{-1} B^t u_h = f_h$.

# %%
from scipy.sparse.linalg import cg, splu, LinearOperator
def schur_op(B,BT,invA,x):
    y = BT @ x
    y2 = invA.solve(y)
    return B @ y2
def solveSys(invA,B,BT, target,rhs):
    schur_comp = LinearOperator( (target.shape[0],target.shape[0]),
                                 matvec = lambda x: schur_op(B,BT,invA,x) )
    target[:], info = cg(schur_comp, rhs, x0 = target)

# %% [markdown]
# A simple loop to compute the experimental order of convergence and
# plotting the laplacian of the result (we of course actually plot
# ${\rm div\sigma_h}$

# %%
def simulate(gridView, spaceHDiv, spaceDG, dirichlet):
    gradSol = spaceHDiv.function(name = "gradSol")
    sol = spaceDG.function(name = "sol")

    refs = 4
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

# %% [markdown]
# Test this on cubes with the RT space and on simplices using BDM

# %%
mth = 'RT'
element = 'cube'
for order in [0,1,2,3,4]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=False)

mth = 'BDM'
element = 'simplex'
for order in [0,1]:
    # Note that the bdm space is constructed with order+1
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order, dim=2)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=False)

# %% [markdown]
# Let's switch the grids around

# %%
mth = 'RT'
element = 'simplex'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)

# %%
mth = 'BDM'
element = 'cube'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)

# %%
mth = 'BDFM'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)

# %% [markdown]
# Let's also check 3d

# %%
mth = 'RT'
element = 'simplex'
for order in [0]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order, dim=3)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)

element = 'cube'
for order in [0,1]:
    gridView, spaceHDiv, spaceDG = getGridSpace(element, mth, order, dim=3)
    simulate(gridView, spaceHDiv, spaceDG, dirichlet=True)
