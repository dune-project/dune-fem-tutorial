import matplotlib.pyplot as pyplot
import numpy
from dune.fem.plotting import plotPointData as plot
import dune.grid as grid
import dune.fem as fem
import dune.common as common
import dune.generator.algorithm as algorithm
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from dune.fem.space import lagrange as solutionSpace
from dune.alugrid import aluConformGrid as leafGridView
from dune.ufl import DirichletBC
from dune.fem.function import gridFunction
from dune.fem import GridMarker
from ufl import *
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.utility import pointSample
from dune.fem.space import finiteVolume as estimatorSpace
from dune.fem.operator import galerkin as estimatorOp
from dune.fem.function import levelFunction

try: # this is needed to remain compatible with version 2022 of ufl
    from ufl import atan_2 as atan2
except:
    pass
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
    space = solutionSpace(gridView, order=order)
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
    uh = space.interpolate(0, name="solution")
    return uh, exact, laplace
uh, exact, laplace = setup()
spaceZ = solutionSpace(uh.space.gridView, order=order+1)
u = TrialFunction(spaceZ)
v = TestFunction(spaceZ)
x = SpatialCoordinate(spaceZ)
a = dot(grad(u), grad(v)) * dx
dual = solutionScheme([a==0,DirichletBC(spaceZ,0)], solver="cg")
z = spaceZ.interpolate(0, name="dual")
zh = uh.copy(name="dual_h")
point = common.FieldVector([0.4,0.4])
pointFunctional = z.copy("dual_rhs")
eh = gridFunction(abs(exact-uh),name="error", order=order+1)
def getFunctional(cpp=True):
    if cpp: # use the C++ version
        computeFunctional = algorithm.load("pointFunctional", "laplace-dwr.hh",
                                      point, pointFunctional, eh)
    else: # this is the equivalent implementation using the bindings
        def computeFunctional(point, pointFunctional, eh):
            space = pointFunctional.space
            en, xLoc = pointSample(space.gridView, point)
            phiVal = numpy.array( space.evaluateBasis(en, xLoc) )[:,0] # issue here with scalar function spaces
            idx = space.mapper()(en)
            lf = eh.localFunction()
            lf.bind(en)
            pointFunctional.as_numpy[ idx ] += phiVal
            return lf(xLoc)
    return computeFunctional
fvspace = estimatorSpace(uh.space.gridView)
estimate = fvspace.interpolate([0], name="estimate")
u = TrialFunction(uh.space)
v = TestFunction(fvspace)
n = FacetNormal(fvspace)
estimator_ufl = abs(div(grad(u)))*abs(z-zh) * v * dx +\
                abs(inner(jump(grad(u)), n('+')))*abs(avg(z-zh)) * avg(v) * dS
estimator = estimatorOp(estimator_ufl)
h1error = dot(grad(uh - exact), grad(uh - exact))
def compute(computeFunctional, tolerance):
    fig = pyplot.figure(figsize=(20,10))
    count = 0
    errorVector    = []
    estimateVector = []
    dofs           = []
    hTol = 1.
    marker = GridMarker( estimate, lambda : hTol )
    while True:
        laplace.solve(target=uh)
        if count%9 == 6:
            plot(uh, figure=(fig, 121), colorbar=False, linewidth=1)
        pointFunctional.clear()
        error = computeFunctional(point, pointFunctional,eh)
        dual.solve(target=z, rightHandSide=pointFunctional)
        zh.interpolate(z)
        estimator(uh, estimate)
        eta = sum(estimate.dofVector)
        dofs           += [uh.space.size]
        errorVector    += [error]
        estimateVector += [eta]
        if count%3 == 2:
            print(count, ": size=", uh.space.gridView.size(0), "estimate=", eta, "error=", error)
        if eta < tolerance:
            break
        # update refinement tolerance
        hTol = eta/uh.space.gridView.size(0)
        fem.gridAdapt(marker, uh) # can also be a list or tuple of function to prolong/restrict
        count += 1
    plot(uh, figure=(fig, 122), colorbar=False, linewidth=1)
compute( getFunctional(cpp=False), tolerance=1e-4 )
compute( getFunctional(cpp=True), tolerance=1e-6 )
pyplot.close('all')
fig = pyplot.figure(figsize=(30,10))
plot(uh, figure=(fig, 131), xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=2)
plot(uh, figure=(fig, 132), xlim=(-0.1, 0.5), ylim=(-0.1, 0.5),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=2)
plot(uh, figure=(fig, 133), xlim=(-0.02, 0.5), ylim=(-0.02, 0.5),
        gridLines="white", colorbar={"shrink": 0.75}, linewidth=2)
fig = pyplot.figure(figsize=(30,10))
levels = levelFunction(uh.space.gridView)
plot(levels, figure=(fig, 131), xlim=(-0.5, 0.5), ylim=(-0.5, 0.5),
        gridLines="white", colorbar={"shrink": 0.75})
plot(levels, figure=(fig, 132), xlim=(-0.1, 0.5), ylim=(-0.1, 0.5),
        gridLines="white", colorbar={"shrink": 0.75})
plot(levels, figure=(fig, 133), xlim=(-0.02, 0.5), ylim=(-0.02, 0.5),
        gridLines="white", colorbar={"shrink": 0.75})