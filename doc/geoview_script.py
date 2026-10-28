from matplotlib import pyplot
from ufl import sqrt, SpatialCoordinate, as_vector
import dune.ufl
from dune.grid import structuredGrid
from dune.fem.view import geometryGridView
from dune.fem.function import gridFunction
from ufl import cos,sin
from dune.fem.plotting import plotPointData as plot

square = structuredGrid([0,0],[1,1],[10,10])
x = SpatialCoordinate(dune.ufl.cell(2))
transform = as_vector([ (x[0]+x[1])/sqrt(2), (-x[0]+x[1])*sqrt(2) ])
diamond = gridFunction(transform,square,order=1,name="diamond")
diamond = geometryGridView(diamond)
diamond.plot()
square = structuredGrid([0],[25],[100])
theta = SpatialCoordinate(dune.ufl.cell(1))[0]
transform = as_vector([cos(theta),sin(theta)])*(25-theta)
diamond = gridFunction(transform,square,order=1,name="diamond")
spiral = geometryGridView(diamond)
plot(theta, gridView=spiral)