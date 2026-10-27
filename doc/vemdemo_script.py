from matplotlib import pyplot
import numpy
from dune.grid import cartesianDomain, gridFunction
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import discreteFunction
from dune.fem import integrate
import dune.fem
from ufl import *
import dune.ufl
from dune.fem.view import geometryGridView

try:
    import dune.vem
except:
    print("This example needs 'dune.vem' - skipping")
    import sys
    sys.exit(0)
polyGrid = dune.vem.polyGrid( dune.vem.voronoiCells([[-0.5,-0.5],[1,1]], 50, lloyd=100) )
indexSet = polyGrid.indexSet
@gridFunction(polyGrid, name="cells")
def polygons(en,x):
    return polyGrid.hierarchicalGrid.agglomerate(indexSet.index(en))
polygons.plot(colorbar="horizontal")
order = 1
space = dune.vem.vemSpace( polyGrid, order=order, storage="numpy",
                           testSpaces=[-1,order-1,order-2])
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)
exact = x[0]*x[1] * cos(pi*x[0]*x[1])
massCoeff = 1+sin(dot(x,x))       # factor for mass term
diffCoeff = 1-0.9*cos(dot(x,x))   # factor for diffusion term
a = (diffCoeff*inner(grad(u),grad(v)) + massCoeff*dot(u,v) ) * dx
b = (-div(diffCoeff*grad(exact)) + massCoeff*exact ) * v * dx
dbc = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]
parameters = {"nonlinear.verbose": False,
              "linear.tolerance": 1e-12,
              "linear.preconditioning.method": "jacobi",
              "linear.verbose": False,
              "penalty": 10*order*order,  # for the dg schemes
              }
df = space.function(name="solution")
scheme = dune.vem.vemScheme( [a==b, *dbc], space, solver="cg",
                             gradStabilization=diffCoeff,
                             massStabilization=massCoeff,
                             parameters=parameters )
info = scheme.solve(target=df)
print("size of space:",space.size,flush=True)
df.plot()
space = dune.vem.vemSpace( polyGrid, order=order, storage="numpy",
                           testSpaces=[0,order-2,order-2])
df = space.function(name="solution")
scheme = dune.vem.vemScheme( [a==b, *dbc], space, solver="cg",
                             gradStabilization=diffCoeff,
                             massStabilization=massCoeff,
                             parameters=parameters )
info = scheme.solve(target=df)
print("size of space:",space.size,flush=True)
df.plot()
print("Polygon diameters:",space.diameters())
methods = [ ### "[legend,space,scheme,spaceKwargs,schemeKwargs]"
            ["lagrange",
             dune.fem.space.lagrange,dune.fem.scheme.galerkin,{},{}],
            ["dg",
             dune.fem.space.dgonb, dune.fem.scheme.dg,  {}, {"penalty":diffCoeff}],
            ["vem-conforming",
             dune.vem.vemSpace,    dune.vem.vemScheme,
                {"testSpaces":[0,order-2,order-2]},  # conforming vem space
                {"gradStabilization":diffCoeff, "massStabilization":massCoeff}],
            ["vem-nonconforming",
             dune.vem.vemSpace,    dune.vem.vemScheme,
                 {"testSpaces":[-1,order-1,order-2]},  # non-conforming vem space
                 {"gradStabilization":diffCoeff, "massStabilization":massCoeff}],
            ["bb-dg",
             dune.vem.bbdgSpace,   dune.vem.bbdgScheme, {}, {"penalty":diffCoeff}],
   ]
def compute(grid, space,spaceArgs, schemeName,schemeArgs):
    space = space( grid, order=order, **spaceArgs )
    df = space.function(name="solution")
    scheme = schemeName( [a==b, *dbc], space, solver="cg", **schemeArgs,
                         parameters=parameters )
    info = scheme.solve(target=df)
    edf = exact-df
    err = [inner(edf,edf),
           inner(grad(edf),grad(edf))]
    errors = [ numpy.sqrt(e) for e in integrate(err) ]
    return df, errors, info
fig = pyplot.figure(figsize=(5*len(methods),10))
figPos = 111+10*len(methods)
for i,m in enumerate(methods):
    dfs,errors,info = compute(polyGrid, m[1],m[3], m[2],m[4])
    print("method (",m[0],"):",
          "dofs: ",dfs.space.size,
          "L^2: ", errors[0], "H^1: ", errors[1],
          info["linear_iterations"], flush=True)
    dfs.plot(figure=(fig,figPos+i), gridLines=None, colorbar="horizontal")
space = dune.vem.vemSpace( polyGrid, order=1, testSpaces=[-1,0,-1])
u = TrialFunction(space)
v = TestFunction(space)
x = SpatialCoordinate(space)
exact = (x[0] - x[0]*x[0] ) * (x[1] - x[1]*x[1] )
Dcoeff = lambda u: 1.0 + u**2
a = (Dcoeff(u) * inner(grad(u), grad(v)) ) * dx
b = -div( Dcoeff(exact) * grad(exact) ) * v * dx
dbcs = [dune.ufl.DirichletBC(space, exact, i+1) for i in range(4)]
scheme = dune.vem.vemScheme( [a==b, *dbcs], space, gradStabilization=Dcoeff(u),
                             solver="cg", parameters=parameters)
solution = space.function(name="solution")
info = scheme.solve(target=solution)
edf = exact-solution
errors = [ numpy.sqrt(e) for e in
           integrate([inner(edf,edf),inner(grad(edf),grad(edf))]) ]
print("non linear problem:", errors )
solution.plot(gridLines=None)
L, W = 1, 0.2
beamGrid = dune.vem.polyGrid( dune.vem.voronoiCells([[0,0],[L,W]], 120) )
indexSet = beamGrid.indexSet
@gridFunction(beamGrid, name="cells")
def polygons(en,x):
    return beamGrid.hierarchicalGrid.agglomerate(indexSet.index(en))
polygons.plot(colorbar="horizontal")
space = dune.vem.vemSpace( beamGrid, order=2, dimRange=2, conforming=True)
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma
x = SpatialCoordinate(space)
dbc = dune.ufl.DirichletBC(space, as_vector([0,0]), x[0]<1e-10)
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
def sigma(u):
    return lambda_*nabla_div(u)*Identity(2) + 2*mu*epsilon(u)
u = TrialFunction(space)
v = TestFunction(space)
f = dune.ufl.Constant((0, -rho*g))
a = inner(sigma(u), epsilon(v))*dx
b = dot(as_vector([0,-rho*g]),v)*dx
displacement = space.function(name="displacement")
scheme = dune.vem.vemScheme( [a==b, dbc], space,
        gradStabilization = as_vector([lambda_+2*mu, lambda_+2*mu]),
        solver="cg", parameters=parameters )
info = scheme.solve(target=displacement)
fig = pyplot.figure(figsize=(20,10))
displacement.plot(gridLines=None, figure=(fig, 121), colorbar="horizontal")
s = sigma(displacement) - (1./3)*tr(sigma(displacement))*Identity(2)
von_Mises = sqrt(3./2*inner(s, s))
plot(von_Mises, gridView=beamGrid, gridLines=None, figure=(fig, 122), colorbar="horizontal")
position = space.interpolate( x+displacement, name="position" )
beam = geometryGridView( position )
beam.plot()
ncC1testSpaces = [ [0], [order-3,order-2], [order-4] ]
polyGrid = dune.vem.polyGrid( dune.vem.voronoiCells([[0,0],[1,1]], 50, lloyd=100) )
space = dune.vem.vemSpace( polyGrid, order=order, testSpaces=ncC1testSpaces)
x = SpatialCoordinate(space)
exact = sin(2*pi*x[0])**2*sin(2*pi*x[1])**2
laplace = lambda w: div(grad(w))
H = lambda w: grad(grad(w))
u = TrialFunction(space)
v = TestFunction(space)
a = ( inner(H(u),H(v)) ) * dx
b = laplace(laplace(exact)) * v * dx
dbcs = [dune.ufl.DirichletBC(space, [0], i+1) for i in range(4)]
scheme = dune.vem.vemScheme( [a==b, *dbcs], space, hessStabilization=1,
                             solver="cg", parameters=parameters )
solution = discreteFunction(space, name="solution")
info = scheme.solve(target=solution)
edf = exact-solution
errors = [ numpy.sqrt(e) for e in
           integrate([inner(edf,edf),inner(grad(edf),grad(edf))]) ]
print("bi-laplace errors:", errors )
solution.plot(gridLines=None)