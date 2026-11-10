# <markdowncell>
# # Crystal Growth
# Here we demonstrate crystallisation on the surface of a liquid due to cooling from [@Guyer].
# Before anything else let us set up the grid and the function space. We use the default DoF storage available in ```dune-fem``` - this can be changed for example to ```istl``` or ```petsc```.


# <codecell>
import time
import dune.fem as fem
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from dune.fem.space import lagrange as solutionSpace

fem.parameter.append({"fem.verboserank":-1})

# storage   = "istl"
storage   = "petsc"
order     = 1
dimDomain = 2      # we are solving this in 2D
dimRange  = 2      # we have a system with two unknowns
domain    = cartesianDomain([4, 4], [8, 8], [40, 40])  # fails with 20x20 in adapt due to petsc error
gridView  = adaptiveGridView( leafGridView( domain, dimgrid=dimDomain ) )

space = solutionSpace(gridView, dimRange=dimRange, order=order, storage=storage)

# <markdowncell>
# We want to solve the following system of equations of variables $\phi$ (phase field) and $T$ (temperature field)
#
# \begin{gather*}
# \tau \frac{\partial \phi}{\partial t} = \nabla \cdot D \nabla \phi + \phi(1-\phi)m(\phi, T), \\
# \frac{\partial T}{\partial t} = D_T \nabla^2 T + \frac{\partial \phi}{\partial t},
# \end{gather*}
#
# where $D_T$ = 2.25, m is given by
#
# \begin{equation*}
# m(\phi, T) = \phi - \frac{1}{2} - \frac{\kappa_1}{\pi} \arctan(\kappa_2 T),
# \end{equation*}
#
# D is given by
#
# \begin{equation*}
# D = \alpha^2(1+c\beta)\left(\begin{array}{cc}
# 1 + c\beta & -c \frac{\partial \beta}{\partial \psi} \\
# c \frac{\partial \beta}{\partial \psi} & 1+c\beta
# \end{array}\right),
# \end{equation*}
#
# and where $\beta = \frac{1-\Phi^2}{1+\Phi^2}$, $\Phi = \tan \left( \frac{N}{2} \psi \right)$, $\psi = \theta + \arctan \left(\frac{\partial \phi/ \partial y}{\partial \phi / \partial x} \right)$ and $\theta$, $N$ are constants.
#
# Let us first set up the parameters for the problem.


# <codecell>
alpha        = 0.015
tau          = 3.e-4
kappa1       = 0.9
kappa2       = 20.
c            = 0.02
N            = 6.


# <markdowncell>
# We define the initial data and create a function from it. We use this value to set up our solution.


# <codecell>
from dune.fem.function import globalFunction
def initial(x):
    r  = (x - [6, 6]).two_norm
    return [ 0 if r > 0.3 else 1, -0.5 ]
initial_gf = globalFunction(gridView, "initial", order+1, initial)
u_h = space.interpolate(initial_gf, name="solution")
u_h_n = u_h.copy()


# <markdowncell>
# As we will be discretising in time, we define the unknown data as $u = (\phi_1, \Delta T_1)$, while given data (from the previous time step) is $u_n = (\phi_0, \Delta T_0)$ and test function $v = (v_0, v_1)$.


# <codecell>
from ufl import TestFunction, TrialFunction, Constant
from dune.ufl import Constant
u = TrialFunction(space)
v = TestFunction(space)
dt = Constant(0, "dt")    # time step


# <markdowncell>
# For the numerical scheme, we discretise the time derivatives in the usual way, and we obtain the weak form by multiplying by a test function and integrating by parts. We also express the system using vectors.
#
# This gets us the following equation.
#
# \begin{equation}
# \int \left( \alpha^2 \frac{dt}{\tau} (D_n\nabla \phi_1) \cdot \nabla v_0 + dt \ D_T \nabla T_1 \cdot \nabla v_1 + \textbf{u} \cdot \textbf{v} - \textbf{s} \cdot \textbf{v} \right) \ dx
# =
# \int (\textbf{u}_n \cdot \textbf{v} - \phi_0 v_1) \ dx
# \end{equation}
#
# where
#
# \begin{equation}
# \textbf{s} = \left( \frac{dt}{\tau}\phi_1(1-\phi_1)m(\phi_1, T_1), \phi_1 \right)^T
# \end{equation}
#
# and
#
# $D_n$ is the anisotropic diffusion using the previous solution $\textbf{u}_n$ to compute the entries.
#
# First we put in the right hand side which only contains explicit data.


# <codecell>
from ufl import inner, dx
a_ex = (inner(u_h_n, v) - inner(u_h_n[0], v[1])) * dx


# <markdowncell>
# For the left hand side we have the spatial derivatives and the implicit parts.


# <codecell>
from ufl import pi, atan, atan_2, tan, grad, as_vector, inner, dot
psi        = pi/8.0 + atan_2(grad(u_h_n[0])[1], (grad(u_h_n[0])[0]))
Phi        = tan(N / 2.0 * psi)
beta       = (1.0 - Phi*Phi) / (1.0 + Phi*Phi)
dbeta_dPhi = -2.0 * N * Phi / (1.0 + Phi*Phi)
fac        = 1.0 + c * beta
diag       = fac * fac
offdiag    = -fac * c * dbeta_dPhi
d0         = as_vector([diag, offdiag])
d1         = as_vector([-offdiag, diag])
m          = u[0] - 0.5 - kappa1 / pi*atan(kappa2*u[1])
s          = as_vector([dt / tau * u[0] * (1.0 - u[0]) * m, u[0]])
a_im = (alpha*alpha*dt / tau * (inner(dot(d0, grad(u[0])),
        grad(v[0])[0]) + inner(dot(d1, grad(u[0])), grad(v[0])[1]))
        + 2.25 * dt * inner(grad(u[1]), grad(v[1]))
        + inner(u,v) - inner(s,v)) * dx


# <markdowncell>
# We set up the scheme with some parameters.


# <codecell>
from dune.fem.scheme import galerkin as solutionScheme
solverParameters = {
        "newton.tolerance": 1e-8,
        "newton.linear.tolerance": 1e-10,
        "newton.linear.preconditioning.method": "jacobi",
        "newton.verbose": False,
        "newton.linear.verbose": False
    }
scheme = solutionScheme(a_im == a_ex, space, solver="bicgstab", parameters=solverParameters)
scheme.model.dt = 0.0002


# <markdowncell>
# We set up the adaptive method. We start with a marking strategy based on the value of the gradient of the phase field variable.

# <codecell>
from dune.ufl import expression2GF
indicator = expression2GF(gridView, dot(grad(u_h[0]),grad(u_h[0])), 0, name="indicator")
# <markdowncell>
# We do the initial refinement of the grid.


# <codecell>
maxLevel   = 8
startLevel = 3
gridView.hierarchicalGrid.globalRefine(startLevel)
u_h.interpolate(initial_gf)
for i in range(startLevel, maxLevel):
    marked = fem.mark(indicator,1.4,1.2,0,maxLevel)
    # print("marked:",marked,"from elements",gridView.size(0))
    fem.adapt(gridView.hierarchicalGrid)
    fem.loadBalance(gridView.hierarchicalGrid)
    # print(gridView.size(0), end="\n")
    u_h.interpolate(initial_gf)
# print()


# <markdowncell>
# Let us start by plotting the initial state of the material, which is just a small circle in the centre.


# <codecell>
from dune.fem.plotting import plotComponents
import matplotlib.pyplot as pyplot
from dune.fem.function import levelFunction, partitionFunction
import matplotlib
vtk = gridView.sequencedVTK("crystal", pointdata=[u_h],
       celldata=[levelFunction(gridView), partitionFunction(gridView)])

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['figure.figsize'] = [10, 5]
# plotComponents(u_h, cmap=pyplot.cm.jet, show=[0])
gridView.writeVTK("parcrystal_start",pointdata=[u_h])


# <markdowncell>
# Finally we set up the time loop and solve the problem - each time this cell is run the simulation will progress to the given ```endTime``` and then the result is shown. The simulation can be progressed further by rerunning the cell while increasing the ```endTime```.


# <codecell>
vtk = gridView.sequencedVTK("parcrystal", pointdata=[u_h])
start    = time.time()
endTime  = 0.08
saveStep = 0.01
saveTime = saveStep
t = 0.0

iters = 0

start = time.time()
while t < endTime:
    u_h_n.assign(u_h)
    info = scheme.solve(target=u_h)
    iters += info["linear_iterations"]
    if t > saveTime:
        saveTime += saveStep
        vtk()
        # plotComponents(u_h, cmap=pyplot.cm.jet)
        print(t, gridView.size(0), info, end="\n")
    t += scheme.model.dt
    marked = fem.mark(indicator,1.4,1.2,0,maxLevel)
    # plotComponents(u_h, cmap=pyplot.cm.jet)
    fem.adapt(u_h)
    # plotComponents(u_h, cmap=pyplot.cm.jet)
    fem.loadBalance(u_h)
    # plotComponents(u_h, cmap=pyplot.cm.jet)
timing = time.time()-start
print("\n runtime:", timing, "iters:",iters)

# plotComponents(u_h, cmap=pyplot.cm.jet)
gridView.writeVTK("parcrystal_end",pointdata=[u_h])
