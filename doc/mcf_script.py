from ufl import *
from dune.ufl import Constant, DirichletBC
import dune.ufl
import dune.geometry as geometry
import dune.fem as fem
from dune.fem.plotting import plotPointData as plot
import matplotlib.pyplot as pyplot
from dune.fem.view import geometryGridView
from dune.fem.space import lagrange as solutionSpace
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.scheme import galerkin as solutionScheme

order = 2
R0 = 2.
gridView = leafGridView("sphere.dgf", dimgrid=2, dimworld=3)
space = solutionSpace(gridView, dimRange=gridView.dimWorld, order=order)
u = TrialFunction(space)
v = TestFunction(space)
x = SpatialCoordinate(space)
positions = space.interpolate(x * (1 + 0.5*sin(2*pi*(x[0]+x[1]))*cos(0.25*pi*x[2])), name="position")
surface = geometryGridView(positions)
space = solutionSpace(surface, dimRange=surface.dimWorld, order=order)
solution = space.interpolate(x, name="solution")
theta = 0.5
I = Identity(3)
dt = Constant(0, "dt")
a = (inner(u - x, v) + dt * inner(theta*grad(u) + (1 - theta)*I, grad(v))) * dx
scheme = solutionScheme(a == 0, space, solver="cg")
count = 0
t = 0.
end_time = 0.05
dt.value = 0.005
fig = pyplot.figure(figsize=(10, 10))
plot(solution, figure=(fig, 131+count%3), colorbar=False,
     gridLines="", triplot=True)
while t < end_time:
    scheme.solve(target=solution)
    t += scheme.model.dt
    count += 1
    positions.assign(solution)
    if count % 4 == 0:
        plot(solution, figure=(fig, 131+count%3), colorbar=False,
             gridLines="", triplot=True)
pyplot.close('all')
referenceView = leafGridView("soap.dgf", dimgrid=2, dimworld=3)
referenceView.hierarchicalGrid.globalRefine(8)
space = solutionSpace(referenceView, dimRange=referenceView.dimWorld,
        order=order)
x         = SpatialCoordinate(space)
positions = space.interpolate(x * (1 + 0.5*sin(2*pi*(x[0]+x[1]))*cos(0.25*pi*x[2])), name="position")
gridView  = geometryGridView(positions)
space     = solutionSpace(gridView, dimRange=gridView.dimWorld, order=order)
u     = TrialFunction(space)
phi   = TestFunction(space)
dt    = Constant(0.01, "timeStep")
t     = Constant(0.0, "time")
endTime      = 0.4
saveInterval = 0.1
uh     = space.interpolate(x, name="uh")
uh_old = uh.copy()
xForm = inner(grad(u), grad(phi)) * dx
form = dot(u - uh_old, phi) * dx + dt * xForm
bc = DirichletBC(space,x)
solverParameters =\
       {"nonlinear.tolerance": 1e-9,
        "linear.tolerance": 1e-11,
        "linear.preconditioning.method": "jacobi",
        "nonlinear.verbose": False,
        "linear.preconditioning.iterations": 3}
scheme = solutionScheme([form == 0, bc], space, solver="cg",
                        parameters=solverParameters)
nextSaveTime = saveInterval
count = 0
try:
    from mayavi import mlab
    from mayavi.tools.notebook import display
    def plot3d(uh,level):
        x, triangles = uh.space.gridView.tessellate(level)
        mlab.init_notebook("png") # use x3d for interactive plotting in the notebook
        mlab.figure(bgcolor = (1,1,1))
        s = mlab.triangular_mesh(x[:,0],x[:,1],x[:,2],triangles)
        display( s )
except (ImportError, ModuleNotFoundError):
    print("mayavi module not found so not showing plot - ignored")
    def plot3d(uh,level):
        pass
plot3d(uh,2)
fig = pyplot.figure(figsize=(50, 10))
plot(solution, figure=(fig, 151+count), colorbar=False,
     gridLines="", triplot=True)
while t.value < endTime:
    uh_old.assign(uh)
    info = scheme.solve(target=uh)
    t.value += dt.value
    positions.assign(uh)
    if t.value >= nextSaveTime or t.value >= endTime:
        nextSaveTime += saveInterval
        count += 1
        plot(uh, figure=(fig, 151+count), colorbar=False,
             gridLines="", triplot=True)
plot3d(uh,2)