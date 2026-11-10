from matplotlib import pyplot as plt
import numpy as np
from dune.grid import structuredGrid as leafGridView
from dune.fem import globalRefine
from ufl import SpatialCoordinate
from dune.ufl import cell
from dune.fem import integrate
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import gridFunction
from dune.fem.function import gridFunction
from dune.fem.space import lagrange as solutionSpace
from dune.grid import OutputType
from ufl import (TestFunction, TrialFunction, SpatialCoordinate, FacetNormal,
                 dx, ds, grad, div, inner, dot, sin, cos, pi, conditional)
from dune.ufl import Constant, DirichletBC
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.scheme import galerkin as solutionScheme
from ufl import FacetNormal, ds, dx, inner, dot, div, grad, sqrt, exp, TrialFunction, TestFunction
from dune.ufl import Constant
from dune.utility import components

gridView = leafGridView([0, 0], [1, 1], [4, 4])
gridView.plot()
gridView.hierarchicalGrid.globalRefine(1)
gridView.plot()
globalRefine(1,gridView.hierarchicalGrid)  # the first argument is the number of refinements to perform
gridView.plot()
globalRefine(-1,gridView.hierarchicalGrid) # coarsen again by one level
gridView.plot()
x = SpatialCoordinate(cell(2))
exact = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
print( integrate(exact, gridView=gridView, order=5) )
plot(exact, gridView=gridView)
gridView.writeVTK('exact', pointdata={'exact': exact})
exact_gf = gridFunction(exact, gridView, name="ufl")
mass = 0
for element in gridView.elements:
    mass += exact_gf(element,[0.5,0.5]) * element.geometry.volume
print(mass)
@gridFunction(gridView,name="callback",order=1)
def exactLocal(element,xLocal):
    x = element.geometry.toGlobal(xLocal)
    return 1/2.*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
@gridFunction(gridView,name="callback",order=1)
def exactGlobal(x):
    return 1/2.*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
lf = exactGlobal.localFunction()
mass = 0
for element in gridView.elements:
    lf.bind(element)
    mass += lf([0.5,0.5]) * element.geometry.volume
    lf.unbind()
print(mass)
print( integrate(abs(exact-exactLocal)) )
gf = gridFunction(exact-exactLocal+exactGlobal, name="ufl")
fig = plt.figure(figsize=(25,10))
gf.plot(figure=(fig,121))
exactGlobal.plot(figure=(fig,122))
space = solutionSpace(gridView, order=2)
mapper = space.mapper()
for i,element in enumerate(gridView.elements):
    print( mapper(element) )
    if i == 4: break # only show indices for first few elements
localPoint = [0.1,0.1] # this is in the reference element
for i,element in enumerate(gridView.elements):
    print( space.evaluateBasis(element, localPoint) )
    if i == 4: break
# The next line fails on MacOS (with a Bus Error 10)
# (tested 28/04/2024). Removed for now
# print("convert to a numpy array:", np.array( space.jacobianBasis(element, localPoint) ) )
# initialized with an analytical expression
u_h = space.interpolate(exact, name='u_h')
# initialized with zeros
u_h = space.function(name='u_h')
u_h.interpolate( lambda x: 1/2.*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1 )
u_h.plot()
u_h_n = u_h.copy(name="previous")
u_h_n.clear()
u_h_n.assign( u_h )
localUh = u_h.localFunction()
mass = 0
for element in gridView.elements:
    localUh.bind(element) # using u_h(element,[0.5,0.5]) also works
    mass += localUh([0.5,0.5]) * element.geometry.volume
    localUh.unbind()
print(mass)
u_h.plot(gridLines="white", linewidth=2)
gridView.writeVTK('uh', pointdata=[u_h])
print( integrate(abs(exact-u_h)) )
for d in u_h.dofVector:
    print(d,end=", ")
print()
u_h_n *= 2
u_h   += u_h_n
# this is equivalent to
u_h.axpy(2,u_h_n)
for d in u_h.dofVector:
    print(d,end=", ")
print()
alpha = Constant(1,name="mass")   # start with alpha=1
x     = SpatialCoordinate(space)
u     = TrialFunction(space)
v     = TestFunction(space)
f     = -div( grad(exact) )
g_N   = grad(exact)
n     = FacetNormal(space)
b     = f*v*dx + dot(g_N,n)*conditional(x[0]>=1e-8,1,0)*v*ds
a     = dot(grad(u), grad(v)) * dx + alpha*u*v * dx
dbc   = DirichletBC(space,exact,x[0]<=1e-8)
scheme = solutionScheme([a == b, dbc], solver='cg')
scheme.solve(target = u_h)
h1error = dot(grad(u_h - exact), grad(u_h - exact))
error = np.sqrt(integrate(h1error))
print("Number of elements:",gridView.size(0),
      "number of dofs:",space.size,"H^1 error:", error)
print(scheme.model.mass, alpha.value)
scheme.model.mass = 2
print(scheme.model.mass, alpha.value)
alpha.value = 0
print(scheme.model.mass, alpha.value)
scheme.solve(target = u_h)
h1error = dot(grad(u_h - exact), grad(u_h - exact))
error = np.sqrt(integrate(h1error))
print("Number of elements:",gridView.size(0),
      "number of dofs:",space.size,"H^1 error:", error)
u_h.plot()
alpha.value = 1
scheme.solve(target = u_h)
u_h.plot()
space = solutionSpace(gridView, order=2)
u_h   = space.function(name='u_h')
u_h_n = u_h.copy(name="previous")
x = SpatialCoordinate(space)
initial = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
dt = Constant(0, name="dt")    # time step
t  = Constant(0, name="t")     # current time
u = TrialFunction(space)
v = TestFunction(space)
abs_du = lambda u: sqrt(inner(grad(u), grad(u)))
K = lambda u: 2/(1 + sqrt(1 + 4*abs_du(u)))
a = ( dot((u - u_h_n)/dt, v) \
    + 0.5*dot(K(u)*grad(u), grad(v)) \
    + 0.5*dot(K(u_h_n)*grad(u_h_n), grad(v)) ) * dx
exact = lambda t: exp(-2*t)*(initial - 1) + 1
f     = lambda s: -2*exp(-2*s)*(initial - 1) - div( K(exact(s))*grad(exact(s)) )
g = lambda s: K(exact(s))*grad(exact(s))
n = FacetNormal(space)
b = 0.5*(f(t)+f(t+dt))*v*dx + 0.5*dot(g(t)+g(t+dt),n)*v*ds
scheme = solutionScheme(a == b, solver='cg')
#scheme.setQuadratureOrders( 2*space.order, 2*space.order+1 )
endTime    = 0.25
exact_end  = exact(endTime)
l2error = gridFunction(dot(u_h - exact_end, u_h - exact_end), name="l2error")
h1error = gridFunction(dot(grad(u_h - exact_end), grad(u_h - exact_end)), name="h1error")
dt.value = 0.01
time = 0
u_h.interpolate(initial)
while time < (endTime - 1e-6):
    t.value = time
    u_h_n.assign(u_h)
    scheme.solve(target=u_h)
    time += dt.value
errors = [np.sqrt(e) for e in integrate([l2error,h1error])]
print('grid size:', gridView.size(0))
print('\t | u_h - u | =', '{:0.5e}'.format(errors[0]))
print('\t | grad(uh - u) | =', '{:0.5e}'.format(errors[1]))
u_h.plot()
gridView.writeVTK('forchheimer', pointdata={'u': u_h, 'l2error': l2error, 'h1error': h1error})
globalRefine(1, gridView.hierarchicalGrid)
dt.value /= 2
u_h.interpolate(initial)
time = 0
while time < (endTime - 1e-6):
    t.value = time
    u_h_n.assign(u_h)
    scheme.solve(target=u_h)
    time += dt.value
errorsFine = [np.sqrt(e) for e in integrate([l2error,h1error])]
print('grid size:', gridView.size(0))
print('\t | u_h - u | =', '{:0.5e}'.format(errorsFine[0]))
print('\t | grad(uh - u) | =', '{:0.5e}'.format(errorsFine[1]))
u_h.plot()
eocs = [ round(np.log(fine/coarse)/np.log(0.5),2)
         for fine,coarse in zip(errorsFine,errors) ]
print("EOCs:",eocs)
# to get a list of all available components:
components()
# to get for example all available grid implementations:
components("grid")
components("space")
components("function")
components("scheme")