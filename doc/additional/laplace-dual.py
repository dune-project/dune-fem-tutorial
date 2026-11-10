try:
    get_ipython().magic('matplotlib inline # can also use notebook or nbagg')
except:
    pass
import numpy
from ufl import *
from dune.common import FieldVector
from dune.generator import algorithm
from dune.grid import cartesianDomain
from dune.fem import doerflerMark, adapt, loadBalance, globalRefine
from dune.fem.function import integrate, levelFunction
from dune.ufl import DirichletBC, expression2GF
imported dune.create as create

newtonParameter = {"tolerance": 1e-10, "verbose": "true",
                   "linear.tolerance": 1e-25,
                   "linear.preconditioning.method": "ilu",
                   "linear.preconditioning.iterations": 1, "linear.preconditioning.relaxation": 1.2,
                   "linear.verbose": "true"}

order       = 2
coarseLevel = 2
tolerance   = 1e-5

vertices = numpy.array([(0,0), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1)])
triangles = numpy.array([(2,1,0), (0,3,2), (4,3,0,), (0,5,4), (6,5,0), (0,7,6)])
domain = {"vertices": vertices, "simplices": triangles}
grid   = create.view("adaptive", grid=gridType, constructor=domain)
grid.hierarchicalGrid.globalRefine(coarseLevel)
spc = create.space("dglagrange", grid, dimRange=1, order=order, storage="istl")

u = TrialFunction(spc)
v = TestFunction(spc)
x = SpatialCoordinate(spc)
n, h = FacetNormal(spc), MinFacetEdgeLength(spc)

phi    = atan_2(x[1], x[0]) + conditional(x[1] < 0, 2*numpy.pi, 0)
exact  = as_vector([inner(x,x)**(0.5*180/270) * sin((180/270) * phi)])
s      = (1-x[0]**2)*(1-x[1]**2)
s      = -sin(1.5*pi*s)
exact *= s
strong = -div(residual[0]*grad(u[0])) + residual[1]
a      = ( inner(residual[0]*grad(u[0]),grad(v[0])) + inner(residual[1],v[0]) ) * dx
forcing = replace(strong,{u:exact})
b       = forcing * v[0] * dx

mu     = 10*D*order*order / avg(h)
bndmu  = 10*D*order*order / h
a  -= ( inner(outer(jump(u), n('+')), avg(residual[0]*grad(v))) +\
        inner(avg(residual[0]*grad(u)), outer(jump(v), n('+'))) ) * dS
a  += mu * inner(jump(u), jump(v)) * dS
a  -= ( inner(outer(u, n), residual[0]*grad(v)) +\
        inner(residual[0]*grad(u), outer(v, n)) ) * ds
b  -= ( inner(outer(exact, n), residual[0]*grad(v)) ) * ds
a  += bndmu * inner(u, v) * ds
b  += bndmu * inner(exact, v) * ds

fvspace = create.space("finitevolume", grid, dimRange=1, storage="istl")
hT = MaxCellEdgeLength(spc)
he = MaxFacetEdgeLength(spc)('+')
n = FacetNormal(spc)
dualWeight = reconSpace.interpolate([0],name="dual_recon")
estimator_ufl = ( (forcing-strong)*dualWeight[0] ) * v[0] * dx +\
                ( inner(jump(residual[0]*grad(u[0])), n('+'))*\
                  avg(dualWeight[0]) ) * avg(v[0]) * dS
estimator = create.operator("galerkin", estimator_ufl, spc, fvspace)
estimator.setQuadratureOrders(4*order+2,4*order+2)

uh = spc.interpolate([0], name="dg")
scheme = create.scheme("galerkin", [a==b,bnd],
            spc, solver="cg",
            parameters={"fem.solver.newton." + k: v for k, v in newtonParameter.items()})
scheme.solve(target=uh)

op = create.operator("galerkin", inner(jump(u),jump(v))*dS, spc)
w = spc.interpolate([0],name="tmp")
op(uh, w)
dgError = [ numpy.sqrt( integrate(grid,(uh[0]-exact[0])**2,order=7) ),
            numpy.sqrt( integrate(grid,inner(grad(uh[0]-exact[0]),grad(uh[0]-exact[0])),order=7)\
            + w.scalarProductDofs(uh)) ]
l2Errors = [dgError[0]]
h1Errors = [dgError[1]]

zh = spc.interpolate([0],name="dual_h")
dualOp = create.scheme("galerkin", [adjoint(a)==0],
                 spc, solver="cg",
                 parameters={"newton." + k: v for k, v in newtonParameter.items()})
pointFunctional = spc.interpolate([0],name="pointFunctional")
point = FieldVector([0.6,0.4])
errors = [ expression2GF(grid, exact-s, reconOrder) for s in solutions ]
dualErrors = algorithm.run('pointFunctional', 'pointfunctional.hh', point, pointFunctional, *errors)
dualOp.solve(target=zh, rhs=pointFunctional)
dualWeight.project(zh-zh)

for i in range(levels):
        if error[2][useEstimate] < tolerance:
            print("COMPLETED:",error[2][useEstimate],"<",tolerance)
            break
        marked = mark(estimate, error[2][useEstimate]/grid.size(0))
        print("elements marked:", marked,"/",grid.size(0),flush=True)
        if sum(marked)==0: break
        adapt(uh)
        loadBalance(uh)
        level += 1
    _, solutions, error = compute(uh)
