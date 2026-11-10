import numpy
import matplotlib.pyplot as pyplot
from dune.fem.plotting import plotPointData as plot
import dune.grid as grid
import dune.fem as fem
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from dune.fem.space import lagrange as solutionSpace
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.function import gridFunction
from dune.fem import integrate
from dune.fem import GridMarker
from ufl import *
from dune.ufl import DirichletBC
from functools import partial
from dune.common import comm
from dune.fem.space import finiteVolume as estimatorSpace
from dune.fem.operator import galerkin as estimatorOp
from dune.fem.function import levelFunction

try:
    from ufl import atan2
except ImportError: # remain compatible with version 2022 of ufl
    from ufl import atan_2 as atan2
# print can be used in parallel runs as before but will only produce output on rank 0
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None
# set the angle for the corner (0<angle<=360)
cornerAngle = 320.
# use a second order space
order = 2
def setup():
    vertices = numpy.zeros((8, 2))
    vertices[0] = [0, 0]
    for i in range(0, 7):
        vertices[i+1] = [numpy.cos(cornerAngle/6*numpy.pi/180*i),
                         numpy.sin(cornerAngle/6*numpy.pi/180*i)]
    triangles = numpy.array([[2,1,0], [0,3,2], [4,3,0],
                             [0,5,4], [6,5,0], [0,7,6]])
    domain = {"vertices": vertices, "simplices": triangles}
    gridView = adaptiveGridView( leafGridView(domain) )
    gridView.hierarchicalGrid.globalRefine(2)
    space = solutionSpace(gridView, order=order, storage="numpy")
    from dune.fem.scheme import galerkin as solutionScheme
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)
    # exact solution for this angle
    Phi = cornerAngle / 180 * pi
    phi = atan2(x[1], x[0]) + conditional(x[1] < 0, 2*pi, 0)
    exact = dot(x, x)**(pi/2/Phi) * sin(pi/Phi * phi)
    a = dot(grad(u), grad(v)) * dx
    # set up the scheme
    laplace = solutionScheme([a==0, DirichletBC(space, exact, 1)], solver="cg",
                parameters={"linear.preconditioning.method":"jacobi"})
    uh = space.interpolate([0], name="solution")
    return uh, exact, laplace
uh, exact, laplace = setup()
h1error = gridFunction(dot(grad(uh - exact), grad(uh - exact)))
errorGlobal = []
dofsGlobal  = []
for count in range(10):
    laplace.solve(target=uh)
    error = numpy.sqrt(integrate(h1error, order=5))
    errorGlobal += [error]
    dofsGlobal  += [uh.space.size]
    print(count, ": size=", uh.space.gridView.size(0), "error=", error)
    fem.globalRefine(1,uh)
uh, exact, laplace = setup()
fvspace = estimatorSpace(uh.space.gridView)
estimate = fvspace.interpolate([0], name="estimate")
u = TrialFunction(uh.space)
v = TestFunction(fvspace)
hT = MaxCellEdgeLength(fvspace)
he = MaxFacetEdgeLength(fvspace)('+')
n = FacetNormal(fvspace)
estimator_ufl = hT**2 * (div(grad(u)))**2 * v * dx +        he * inner(jump(grad(u)), n('+'))**2 * avg(v) * dS
estimator = estimatorOp(estimator_ufl)
tolerance = 0.05
h1error = gridFunction(dot(grad(uh - exact), grad(uh - exact)))
fig = pyplot.figure(figsize=(10,10))
count = 0
errorVector    = []
estimateVector = []
dofs           = []
marker = GridMarker( estimate, # grid function to be evaluated at cell center
                     0.6,                    # refinement tolerance
                     strategy = 'doerfler',  # marking strategy doerfler
                     layered = 0.1)          # parameter for Doerfler layered strategy.
while True:
    laplace.solve(target=uh)
    if count%9 == 8:
        plot(uh, figure=(fig, 131+count//9), colorbar=False)
    error = numpy.sqrt(integrate(h1error, order=5))
    estimator(uh, estimate)
    eta = numpy.sqrt( sum(estimate.dofVector) )
    dofs           += [uh.space.size]
    errorVector    += [error]
    estimateVector += [eta]
    if count%3 == 2:
        print(count, ": size=", uh.space.gridView.size(0), "estimate=", eta, "error=", error)
    if eta < tolerance:
        break
    fem.gridAdapt(marker, uh) # can also be a list or tuple of function to prolong/restrict
    count += 1
plot(uh, figure=(fig, 131+2), colorbar=False)
pyplot.close('all')
fig = pyplot.figure(figsize=(35,15))
plot(uh, figure=(fig, 131), xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=2)
plot(uh, figure=(fig, 132), xlim=(-0.1, 0.1), ylim=(-0.1, 0.1),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=2)
plot(uh, figure=(fig, 133), xlim=(-0.02, 0.02), ylim=(-0.02, 0.02),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=2)
fig = pyplot.figure(figsize=(35,15))
levels = levelFunction(uh.space.gridView)
plot(levels, figure=(fig, 131), xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=0.5)
plot(levels, figure=(fig, 132), xlim=(-0.1, 0.1), ylim=(-0.1, 0.1),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=0.5)
plot(levels, figure=(fig, 133), xlim=(-0.02, 0.02), ylim=(-0.02, 0.02),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=0.5)
pyplot.close('all')
pyplot.loglog(dofsGlobal,errorGlobal,label="H^1 error (global refine)")
pyplot.loglog(dofs,errorVector,label=" H^1 error (adaptive)")
pyplot.loglog(dofs,estimateVector,label="estimator (adaptive)")
pyplot.grid(visible=True, which='major', color='black', linestyle='-')
pyplot.grid(visible=True, which='minor', color='black', linestyle='--')
pyplot.legend(frameon=True,facecolor="white",framealpha=1)