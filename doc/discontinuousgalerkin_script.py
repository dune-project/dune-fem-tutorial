import numpy, math, sys
from matplotlib import pyplot
from dune.grid import cartesianDomain
from dune.grid import yaspGrid as leafGridView
from dune.fem.space import dglegendre as dgSpace
from dune.fem.scheme import galerkin as solutionScheme
from dune.ufl import Constant
from ufl import ( TestFunction, TrialFunction, SpatialCoordinate, FacetNormal,
                  dx, ds, grad, div, grad, dot, inner, sqrt, exp, atan, conditional,
                  as_vector, avg, jump, dS, CellVolume, FacetArea )
from dune.fem.operator import molGalerkin as molOperator

domain   = cartesianDomain([-1, -1], [1, 1], [20, 20], overlap=1)
gridView = leafGridView(domain)
space    = dgSpace(gridView, order=2)
parallel = gridView.comm.size > 1
u    = TrialFunction(space)
v    = TestFunction(space)
n    = FacetNormal(space)
he   = avg( CellVolume(space) ) / FacetArea(space)
hbnd = CellVolume(space) / FacetArea(space)
x    = SpatialCoordinate(space)
eps = Constant(0.1,"eps")
b    = as_vector([1,0])
hatb = (dot(b, n) + abs(dot(b, n)))/2.0
dD   = conditional((1+x[0])*(1-x[0])<1e-10,1,0)
g    = conditional(x[0]<0,atan(10*x[1]),0)
beta = 10 * space.order**2 if space.order > 0 else 1
dn = lambda u: dot(grad(u),n)
aInternal     = dot(eps*grad(u) - b*u, grad(v)) * dx
diffSkeleton  = eps*beta/he*jump(u)*jump(v)*dS -\
                eps*avg(dn(u))*jump(v)*dS -\
                eps*jump(u)*avg(dn(v))*dS
diffSkeleton += eps*beta/hbnd*(u-g)*v*dD*ds -\
                eps*dot(grad(u),n)*v*dD*ds
advSkeleton   = jump(hatb*u)*jump(v)*dS
advSkeleton  += ( hatb*u + (dot(b,n)-hatb)*g )*v*dD*ds
form          = aInternal + diffSkeleton + advSkeleton
scheme = solutionScheme(form==0, solver="gmres",
            parameters={"linear.preconditioning.method":"jacobi",
                       }
                        )
uh = space.interpolate(0, name="solution")
info = scheme.solve(target=uh)
if parallel:
    gridView.writeVTK("u_h", pointdata=[uh])
else:
    uh.plot()
eps.value = 1e-5 # could also use scheme.model.eps = 1e-5
uh.interpolate(0)
scheme.solve(target=uh)
if parallel:
    gridView.writeVTK("u_h", pointdata=[uh])
else:
    uh.plot()
if parallel:
    sys.exit(0)
N = 200
domain1d = cartesianDomain([0], [1], [N], overlap=1)
gridView = leafGridView( domain1d )
space    = dgSpace(gridView, order=1)
u    = TrialFunction(space)
v    = TestFunction(space)
n    = FacetNormal(space)
x    = SpatialCoordinate(space)
speed = 0.5
b     = as_vector([speed])
hatb  = (dot(b, n) + abs(dot(b, n)))/2.0
dD    = conditional(x[0]<0.01,1,0)
g     = 1
advInternal = dot(-b*u, grad(v)) * dx
advSkeleton = jump(hatb*u)*jump(v) * dS
advBnd      = ( hatb*u + (dot(b,n)-hatb)*g )*v * dD*ds
form        = (advInternal + advSkeleton + advBnd)
op = molOperator(form)
uh = space.interpolate(conditional(x[0]<0.25,1,0), name="solution")
uh.plot()
w = uh.copy()
un = uh.copy()
cfl = 0.1
tau = cfl/(speed*N)
t = 0
while t<1:
    un.assign(uh)
    op(uh, w)
    uh.axpy(-tau,w)   # with numpy backend equivalent to 'uh.as_numpy[:] -= tau*w.as_numpy[:]'
    op(uh, w)
    uh.axpy(-tau,w)   # with numpy backend equivalent to 'uh.as_numpy[:] -= tau*w.as_numpy[:]'
    uh *= 0.5
    uh.axpy(0.5,un)
    t += tau
uh.plot()