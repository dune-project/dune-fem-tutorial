from matplotlib import pyplot as plt
import dune.fem as fem
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from dune.fem.space import lagrange as solutionSpace
from dune.fem import threading
from dune.fem import GridMarker
from dune.common import comm
from ufl import dot,sqrt,conditional,as_vector, SpatialCoordinate
from ufl import TestFunction, TrialFunction
from dune.ufl import Constant
from ufl import inner, dx
from ufl import pi, atan, tan, grad, inner
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.plotting import plotComponents
import matplotlib.pyplot as pyplot
from dune.fem.function import levelFunction, partitionFunction
import matplotlib

threading.use = max(4,threading.max) # use at most 4 threads
rank = comm.rank
size = comm.size
order = 1
dimDomain = 2     # we are solving this in 2D
dimRange = 2      # we have a system with two unknowns
domain = cartesianDomain([4, 4], [8, 8], [3, 3])
gridView  = adaptiveGridView( leafGridView( domain ) )
space = solutionSpace(gridView, dimRange=dimRange, order=order, storage="numpy")
alpha        = 0.015
tau          = 3.e-4
kappa1       = 0.9
kappa2       = 20.
c            = 0.02
N            = 6.
x = SpatialCoordinate(space)
r = sqrt( dot( x-as_vector([6,6]), x-as_vector([6,6])) )
initial = as_vector( [conditional(r>0.3,0,1), -0.5] )
u_h = space.interpolate(initial, name="solution")
u_h_n = u_h.copy()
u = TrialFunction(space)
v = TestFunction(space)
dt = Constant(0, "dt")    # time step
a_ex = (inner(u_h_n, v) - inner(u_h_n[0], v[1])) * dx
try:
    from ufl import atan2
except ImportError: # remain compatible with version 2022 of ufl
    from ufl import atan_2 as atan2
psi        = pi/8.0 + atan2(grad(u_h_n[0])[1], (grad(u_h_n[0])[0]))
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
solverParameters = {
        "nonlinear.tolerance": 1e-8,
        "linear.tolerance": 1e-10,
        "nonlinear.verbose": False,
        "linear.verbose": False
    }
scheme = solutionScheme(a_im == a_ex, space, solver="gmres", parameters=solverParameters)
indicator = dot(grad(u_h[0]),grad(u_h[0]))
maxLevel = 11 if size==1 else 13
startLevel = 5
gridView.hierarchicalGrid.globalRefine(startLevel)
u_h.interpolate(initial)
# default marking strategy based on indicator values
marker = GridMarker(indicator,
                    refineTolerance  = 1.4,
                    coarsenTolerance = 1.2,
                    minLevel=0, maxLevel=maxLevel)
for i in range(startLevel, maxLevel):
    fem.gridAdapt(marker, u_h)
    u_h.interpolate(initial)
    print(gridView.size(0), end=" ")
print()
vtk = gridView.sequencedVTK("crystal", pointdata=[u_h],
         celldata=[levelFunction(gridView), partitionFunction(gridView)])
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['figure.figsize'] = [10, 5]
dt.value = 0.0005
t = 0.0
def plot(count,u_h):
    plotArgs = {"linewidth":0.7, "onlyContours":True,"contours":[0.5],
                "contourWidth":4, "contourColor":"yellow"}
    if size == 1: # serial run
        u_h[0].plot(**plotArgs)
    else:
        fig = plt.figure(figsize=(10,10))
        u_h[0].plot(**plotArgs, figure=fig)
        fig.savefig(f"crystal{rank}{size}_{count}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    vtk()
count = 0
endTime = 0.1
saveInterval = endTime/5
plot(count, u_h)
count += 1
nextSaveStep = saveInterval
while t < endTime:
    u_h_n.assign(u_h)
    scheme.solve(target=u_h)
    if t > nextSaveStep - dt.value/3:
        plot(count,u_h)
        count += 1
        nextSaveStep += saveInterval
    t += dt.value
    fem.gridAdapt(marker, u_h) # can also be a list or tuple of function to prolong/restrict
plot(count, u_h)
fig = pyplot.figure(figsize=(30,10))
plotComponents(u_h, figure=fig, cmap=pyplot.cm.jet)