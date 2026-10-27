from matplotlib import pyplot
import numpy
from dune.alugrid import aluSimplexGrid as leafGridView3d
from dune.fem.space import lagrange as solutionSpace
from dune.fem.scheme import galerkin as solutionScheme
from ufl import TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx, conditional, sqrt
import dune.generator.algorithm as algorithm
from dune.common import FieldVector
from dune.fem.function import gridFunction
from dune.fem.utility import lineSample

try:
    import pygmsh
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon([
              [ 0.0,  0.5, 0.0], [-0.1,  0.1, 0.0], [-0.5,  0.0, 0.0],
              [-0.1, -0.1, 0.0], [ 0.0, -0.5, 0.0], [ 0.1, -0.1, 0.0],
              [ 0.5,  0.0, 0.0], [ 0.1,  0.1, 0.0] ], mesh_size=0.05)
        geom.twist(
            poly,
            translation_axis=[0, 0, 1], rotation_axis=[0, 0, 1], point_on_axis=[0, 0, 0],
            angle=numpy.pi / 3,
        )
        mesh = geom.generate_mesh(verbose=False)
    points, cells = mesh.points, mesh.cells_dict
    domain3d = {"vertices":points.astype("float"), "simplices":cells["tetra"]}
except ImportError: # pygmsh not installed - use a simple cartesian domain
    print("pygmsh module not found using a simple Cartesian domain - ignored")
    from dune.grid import cartesianDomain
    domain3d = cartesianDomain([-0.25,-0.25,0],[0.25,0.25,1],[30,30,60])
gridView3d  = leafGridView3d(domain3d)
space3d = solutionSpace(gridView3d, order=1)
u = TrialFunction(space3d)
v = TestFunction(space3d)
x = SpatialCoordinate(space3d)
scheme3d = solutionScheme((dot(grad(u),grad(v))+u*v)*dx ==
                        conditional(dot(x,x)<.01,100,0)*v*dx,
                        solver='cg')
uh3d = space3d.interpolate(0,name="solution")
info = scheme3d.solve(target=uh3d)
x0, x1 = FieldVector([0,0,0]), FieldVector([0,0,1])
p,v = algorithm.run('sample', 'utility.hh', uh3d, x0, x1, 100)
x,y = numpy.zeros(len(p)), numpy.zeros(len(p))
length = (x1-x0).two_norm
for i in range(len(x)):
    x[i] = (p[i]-x0).two_norm / length
    y[i] = v[i][0]
pyplot.plot(x,y)
absGrad = gridFunction(sqrt(dot(grad(uh3d),grad(uh3d))))
p,v = algorithm.run('sample', 'utility.hh', absGrad, x0, x1, 100)
for i in range(len(x)):
    y[i] = v[i][0]
pyplot.plot(x,y)
absGrad = gridFunction(grad(uh3d))
p,v = algorithm.run('sample', 'utility.hh', absGrad, x0, x1, 100)
dx,dy = numpy.zeros(len(p)), numpy.zeros(len(p))
for i in range(len(x)):
    dx[i] = v[i][0]
    dy[i] = v[i][1]
pyplot.plot(x,dx)
pyplot.plot(x,dy)
x,y = lineSample(uh3d,[0,0,0],[0,0,1],100)
pyplot.plot(x,y)