import numpy
from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange
from dune.fem import integrate
from ufl import TestFunction, TrialFunction, SpatialCoordinate, FacetNormal, \
                dx, ds, div, grad, dot, inner, exp, sin, conditional
from dune.fem.scheme import galerkin
from dune.fem.scheme import linearized
from dune.ufl import DirichletBC
from dune.common.checkconfiguration import assertCMakeHave, ConfigurationError
import dune.fem
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from ufl import (TestFunction, TrialFunction, SpatialCoordinate,
                 dx, grad, inner, dot, sin, cos, pi )

gridView = leafGridView([0, 0], [1, 1], [24, 24])
space = lagrange(gridView, order=2, storage='istl')
u_h = space.interpolate(0, name='u_h')
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )
exact = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
a  = ( inner(grad(u),grad(v)) + (1+u)**2*u*v ) * dx
bf = (-div(grad(exact)) + (1+exact)**2*exact) * v * dx
bg = dot(grad(exact),n)*v*ds
def printResult(method,error,info):
    print(method,"L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
                 *[ numpy.sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]),
          "\n\t","solver info=",info,flush=True)
scheme = galerkin(a == bf+bg, solver='cg',
                  parameters={"linear.preconditioning.method":"jacobi",
                              "nonlinear.forcing":"eisenstatwalker",
                              "linear.errormeasure":"residualreduction",
                              "nonlinear.tolerance":1e-7},
                 )
u_h.interpolate(2)               # start with the boundary conditions at the top
info = scheme.solve(target=u_h)
printResult("Default",u_h-exact,info)
u_h.plot()
class Scheme1:
    def __init__(self, scheme, u0):
        self.scheme = scheme
        self.jacobian = scheme.linear(ubar=u0) # obtain a linear operator for the Newton method
    def solve(self, target):
        res = target.copy(name="residual")
        dh  = target.copy(name="direction")
        n, linIter = 0,0
        while True:
            self.scheme(target, res)
            absF = res.scalarProductDofs(res)
            if absF < 1e-7**2: # this is the same tolerance we set above for the built-in Newton solver
                break
            self.scheme.jacobian(target,self.jacobian)  # assemble the linearization
            dh.clear()
            info = self.jacobian.solve(target=dh,rightHandSide=res)
            linIter += info["linear_iterations"]
            target -= dh
            n += 1
        return {"iterations":n, "linear_iterations":linIter}
u_h.interpolate(2)                        # start with the boundary conditions at the top
scheme_cls = Scheme1(scheme,u0=u_h)
info = scheme_cls.solve(target=u_h)
printResult("Newton 1",u_h-exact,info)
class Scheme2:
    def __init__(self, scheme, u0):
        self.scheme = scheme
        self.linear = linearized(scheme, ubar=u0,
                      parameters={ "method":"gmres", "gmres.restart":50,
                               "errormeasure":"absolute",
                            })
    def solve(self, target):
        prev = target.copy()
        n, linIter = 0,0
        while True:
            self.linear.setup(target)
            info = self.linear.solve(target=target)
            linIter += info["linear_iterations"]
            prev.sub( target )
            absDiff = prev.scalarProductDofs(prev)
            print(n,absDiff)
            if absDiff < 1e-7**2:
                break
            prev.assign(target)
            n += 1
        return {"iterations":n, "linear_iterations":linIter}
u_h.clear()                         # reset u_h - so we start the Newton method with u_h=0
scheme_cls = Scheme2(scheme,u_h)
info = scheme_cls.solve(target=u_h)
printResult("Newton 2 (gmres)",u_h-exact,info)
model = ( inner(grad(u), grad(v)) + (u - conditional(dot(x,x)<0.6,10.,0.))*v ) * dx
sc  = galerkin(model==0, solver="cg",
               parameters={"linear.verbose":False,"nonlinear.verbose":True}
              )
sol = space.interpolate(0, name="u_h")
jac = linearized(sc, parameters={ "method":"gmres", "gmres.restart":100, })
info = jac.solve(target=sol)
print(info,flush=True)
sol.clear()
info = sc.solve(target=sol)
print(info,flush=True)
model = ( inner(grad(u), grad(v)) - 15*conditional(dot(x,x)<0.6,1.,0.) * v ) * dx
dbcs  = [ DirichletBC(space,-1,1), DirichletBC(space, 1,2) ]
op  = galerkin([model==0, *dbcs], space)
sol = space.interpolate(0, name="u_h")
lin = op.linear()
op.setConstraints(sol)
rhs = sol.copy()
op(sol,rhs)
rhs *= -1
op.setConstraints(rhs)
lin.solve(rightHandSide=rhs,target=sol)
sol.plot()
scheme = galerkin([a == bf, DirichletBC(space,exact)], solver='cg')
u_h.interpolate(2)                        # start with the boundary conditions at the top
scheme_cls = Scheme1(scheme,u0=u_h)
info = scheme_cls.solve(target=u_h)
printResult("Dirichlet, Newton 1",u_h-exact,info)
u_h.interpolate(2)
scheme_cls = Scheme2(scheme,u0=u_h)
info = scheme_cls.solve(target=u_h)
printResult("Dirichlet, Newton 2",u_h-exact,info)
try:
    assertCMakeHave("HAVE_PETSC")
    petsc = True
except ConfigurationError:
    print("Dune not configured with petsc - skipping example")
    petsc = False
if petsc:
    spacePetsc = lagrange(gridView, order=2, storage='petsc')
    schemePetsc = galerkin([a == bf+bg, DirichletBC(space,exact,2)],
                           space=spacePetsc, solver="cg",
                           parameters={"linear.preconditioning.method":"hypre"})
    uPetsc_h = spacePetsc.interpolate(2, name='u_h')
    info = schemePetsc.solve(target=uPetsc_h)
    printResult("PETSc",uPetsc_h-exact,info)
spaceIstl = lagrange(gridView, order=2, storage='istl')
schemeIstl = galerkin(a == bf+bg, solver=("istl","cg"),
                      parameters={"linear.preconditioning.method":"amg-jacobi"})
u_h.interpolate(2)
info = schemeIstl.solve(target=u_h)
printResult("Dune-Istl",u_h-exact,info)
gridView = structuredGrid([0, 0], [1, 1], [200, 200])
space = lagrange(gridView, order=1)
u_h   = space.interpolate(0, name='u_h')
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)
f = (8*pi**2+1) * cos(2*pi*x[0])*cos(2*pi*x[1])
a = ( inner(grad(u),grad(v)) + u*v ) * dx
l = f*v * dx
scheme = galerkin( a==l, solver="cg", parameters=
                   {"linear.tolerance": 1e-12,
                    "linear.verbose": True,
                    "linear.preconditioning.method": "jacobi",
                    "linear.errormeasure": "relative",
                    "logging": "log-jacobi"
                   } )
info = scheme.solve(target=u_h)
try:
    from pprint import pprint as _pprint
    pprint = lambda *args,**kwargs: _pprint(*args,**kwargs,width=200,compact=False)
except ImportError:
    pprint = print
pprint(dune.fem.parameter.log())
pprint(scheme.parameterLog())
param = dict(scheme.parameterLog()) # this method returns a set of pairs which we can convert to a dictionary
param["logging"] = "Dirichlet" # only needed to use the `parameterLog` method
param["linear.verbose"] = False
scheme2 = galerkin( [a==l,DirichletBC(space,0)], parameters=param )
u_h.interpolate(2)
info = scheme2.solve(target=u_h)
pprint(scheme2.parameterLog())
try:
    scheme = galerkin( a==l, solver="cg", parameters=
                       {"linear.tolerance": 1e-12,
                        "linear.verbose": True,
                        "linear.preconditioning.method": "help",
                        "linear.errormeasure": "relative",
                        "logging": "precon"
                       } )
    scheme.solve(target=u_h)
except RuntimeError as rte:
    print(rte)
space = lagrange(gridView, order=2)
help(scheme)