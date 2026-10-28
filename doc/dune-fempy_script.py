import numpy as np
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from ufl import (TestFunction, TrialFunction, SpatialCoordinate,
                 dx, grad, inner, dot, sin, cos, pi )
from dune.fem import assemble
from scipy.sparse.linalg import spsolve as solver
from dune.fem import integrate
from ufl import conditional, FacetNormal, ds, div
from dune.ufl import DirichletBC
from dune.fem.scheme import galerkin as solutionScheme
from dune.ufl import Constant
from dune.fem.operator import galerkin

gridView = structuredGrid([0, 0], [1, 1], [20, 20])
space = lagrange(gridView, order=1)
u_h   = space.interpolate(0, name='u_h')
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)
f = (8*pi**2+1) * cos(2*pi*x[0])*cos(2*pi*x[1])
a = ( inner(grad(u),grad(v)) + u*v ) * dx
l = f*v * dx
mat,rhs = assemble(a==l)
A = mat.as_numpy
b = rhs.as_numpy
y = u_h.as_numpy
y[:] = solver(A,b)
u_h.plot()
exact = cos(2*pi*x[0])*cos(2*pi*x[1])
e_h = u_h-exact
squaredErrors = integrate([e_h**2,inner(grad(e_h),grad(e_h))])
print("L^2 and H^1 errors:",[np.sqrt(e) for e in squaredErrors])
print("average:",integrate(exact,gridView=gridView) )
# since the integrand is scalar, the following is equivalent:
print("average:",assemble(exact*dx,gridView=gridView) )
exact = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
a   = dot(grad(u), grad(v)) * dx
f   = -div( grad(exact) )
g_N = grad(exact)
n   = FacetNormal(space)
l   = f*v*dx + dot(g_N,n)*conditional(x[0]>=1e-8,1,0)*v*ds
dbc = DirichletBC(space,exact,x[0]<=1e-8)
mat,rhs = assemble([a==l,dbc])
u_h.as_numpy[:] = solver(mat.as_numpy, rhs.as_numpy)
u_h.plot()
e_h = u_h-exact
squaredErrors = integrate([e_h**2,inner(grad(e_h),grad(e_h))])
print("L^2 and H^1 errors:",[np.sqrt(e) for e in squaredErrors])
l   = conditional(dot(x,x)<0.1,1,0)*v*dx
dbc = DirichletBC(space,x[1]*(1-x[1]),x[0]<=1e-8)
rhs = assemble([l,dbc])
u_h.as_numpy[:] = solver(mat.as_numpy, rhs.as_numpy)
u_h.plot()
a = ( inner(grad(u),grad(v)) + (1+u)**2*u*v ) * dx
l = dot(x,x)*v * dx
scheme = solutionScheme(a == l, solver='cg')
u_h.clear() # set u_h to zero as initial guess for the Newton solver
info = scheme.solve(target=u_h)
print(info)
u_h.plot()
exact = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
a  = ( inner(grad(u),grad(v)) + (1+u)**2*u*v ) * dx
bf = (-div(grad(exact)) + (1+exact)**2*exact) * v * dx
bg = dot(grad(exact),n)*v*ds
scheme = solutionScheme(a == bf+bg, solver="cg")
errors = 0,0
loops = 2
for eocLoop in range(loops):
    u_h.interpolate(0)
    info = scheme.solve(target=u_h)
    errors_old = errors
    l2error, h1error = dot(u_h-exact, u_h-exact), dot(grad(u_h-exact), grad(u_h-exact))
    errors = [np.sqrt(e) for e in integrate([l2error,h1error])]
    if eocLoop == 0:
        eocs = ['-','-']
    else:
        eocs = [ round(np.log(e/e_old)/np.log(0.5),2) for e,e_old in zip(errors,errors_old) ]
    print("step:", eocLoop, ", size:", gridView.size(0))
    print("\t\t L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(*errors),", eocs =", eocs)
    print("\t\t solver info=",info)
    u_h.plot()
    if eocLoop < loops-1:
        gridView.hierarchicalGrid.globalRefine(1)
epsilon = Constant(1,name="eps")   # start with epsilon=1
x,u,v   = SpatialCoordinate(space), TrialFunction(space), TestFunction(space)
f       = dot(x,x)
a       = epsilon*dot(grad(u), grad(v)) * dx + (1+u_h)**2*u*v * dx
b       = f*v*dx
scheme = solutionScheme(a == b, solver='cg')
w_h    = space.interpolate(0,name="w_h")
scheme.solve(target = w_h)
w_h.plot()
op = galerkin(a == b)
print(scheme.model.eps, epsilon.value)
epsilon.value = 0                          # change to the problem which matches our exact solution
print(scheme.model.eps, epsilon.value)
u_h.interpolate(0)
scheme.solve(target = w_h)
w_h.plot()