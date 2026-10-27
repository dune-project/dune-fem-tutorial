from matplotlib import pyplot
from dune.fem.plotting import plotPointData as plot
from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange as solutionSpace
from dune.fem.scheme import galerkin as solutionScheme
from ufl import *
import dune.ufl
from dune.fem.view import geometryGridView

gridView = leafGridView([0, 0], [1, 0.15], [100, 15])
space = solutionSpace(gridView, dimRange=2, order=2, storage="istl")
displacement = space.interpolate([0,0], name="displacement")
x = SpatialCoordinate(space)
dbc = dune.ufl.DirichletBC(space, as_vector([0,0]), x[0]<1e-10)
lamb = 0.1     # 1st Lamé coefficient (\lambda)
mu   = 1       # 2nd Lamé coefficient (\mu)
rho  = 1/1000. # material density
g    = 9.8     # gravitational acceleration
epsilon = lambda u: 0.5*(nabla_grad(u) + nabla_grad(u).T)
sigma = lambda u: lamb*nabla_div(u)*Identity(2) + 2*mu*epsilon(u)
forcing = dune.ufl.Constant( [0,-rho*g], "forcing")
u = TrialFunction(space)
v = TestFunction(space)
equation = inner(sigma(u), epsilon(v))*dx == dot(forcing,v)*dx
scheme = solutionScheme([equation, dbc], solver='cg',
            parameters = {"linear.preconditioning.method": "ilu"} )
info = scheme.solve(target=displacement)
fig = pyplot.figure(figsize=(20,10))
displacement.plot(gridLines=None, figure=(fig, 121), colorbar="horizontal")
s = sigma(displacement) - (1./3)*tr(sigma(displacement))*Identity(2)
von_Mises = sqrt(3./2*inner(s, s))
plot(von_Mises, gridView=gridView, gridLines=None, figure=(fig, 122), colorbar="horizontal")
position = space.interpolate( x+displacement, name="position" )
beam = geometryGridView( position )
beam.plot()