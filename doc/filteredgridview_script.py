from ufl import SpatialCoordinate, dot
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.view import filteredGridView
from dune.fem.space import lagrange
from dune.fem.utility import inspectBoundaryIds
from dune.ufl import DirichletBC
from dune.fem.scheme import galerkin
from ufl import ( TrialFunction, TestFunction, FacetNormal, conditional,
                  div, grad, dx, ds )
from dune.fem.function import gridFunction

gridView = leafGridView( cartesianDomain([-1,0],[1,1],[50,25]) )
space = lagrange(gridView, order=2)
x = SpatialCoordinate(space)
solution = space.interpolate(dot(x,x),name="solution")
solution.plot(colorbar="horizontal")
print("number of dofs:", solution.size,\
      "integral over original domain",solution.integrate())
inspectBoundaryIds(gridView).plot(colorbar="horizontal")
def filterIds(e):
    r = e.geometry.center.two_norm
    if r < 0.5:
        return 10
    elif r < 0.75:
        return 15
    else:
        return 20
filteredView10 = filteredGridView(gridView, filterIds, domainId=10,
                                  useFilteredIndexSet=True)
space10 = lagrange(filteredView10, order=2)
solution10 = space10.interpolate(dot(x,x),name="solution")
solution10.plot(colorbar="horizontal")
print("number of dofs:", solution10.size,\
      "integral over filtered domain 10",solution10.integrate())
solution.clear()
filteredView10.interpolate(solution10,solution)
solution.plot(colorbar="horizontal")
solution10.clear()
filteredView10.interpolate(solution,solution10)
solution10.plot(colorbar="horizontal")
inspectBoundaryIds(filteredView10).plot(colorbar="horizontal")
filteredView15 = filteredGridView(gridView, filterIds, domainId=15)
space15 = lagrange(filteredView15, order=2)
x = SpatialCoordinate(space15)
solution15 = space15.interpolate(dot(x,x),name="solution")
solution15.plot(colorbar="horizontal")
print("number of dofs:", solution15.size,\
      "integral over filtered domain 15",solution15.integrate())
inspectBoundaryIds(filteredView15).plot(colorbar="horizontal")
solution15 = space15.function(name="solution")
solution15.interpolate(dot(x,x))
solution.clear()
solution.assign(solution15)  # here the '0' on the domain which is not in the filter is important
solution.plot()
solution15.assign(solution)
solution15.plot()
filteredView20 = filteredGridView(gridView, filterIds, domainId=20,
                                  useFilteredIndexSet=True)
space20 = lagrange(filteredView20, order=2)
x = SpatialCoordinate(space20)
solution20 = space20.interpolate(dot(x,x),name="solution")
solution20.plot(colorbar="horizontal")
print("number of dofs:", solution20.size,\
      "integral over filtered domain 20",solution20.integrate())
inspectBoundaryIds(filteredView20).plot(colorbar="horizontal")
exact = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
u     = TrialFunction(space10)
v     = TestFunction(space10)
f     = -div( grad(exact) )
g_N   = grad(exact)
n     = FacetNormal(space10)
b     = f*v*dx + dot(g_N,n)*conditional(x[0]>=1e-8,1,0)*v*ds
a     = dot(grad(u), grad(v)) * dx
dbc   = DirichletBC(space10,exact,x[0]<=1e-8)
scheme = galerkin([a == b, dbc], solver='cg')
scheme.solve(target = solution10)
solution10.plot(colorbar="horizontal")
gridFunction(exact,gridView=filteredView10).plot(level=3,colorbar="horizontal")
u     = TrialFunction(space15)
v     = TestFunction(space15)
a     = dot(grad(u), grad(v)) * dx
dbc1  = DirichletBC(space15,1,10) # lower internal boundary (domain id there is 10)
dbc2  = DirichletBC(space15,2,20) # upper internal boundary (domain id there is 20)
scheme = galerkin([a == 0, dbc1, dbc2], solver='cg')
scheme.solve(target = solution15)
solution15.plot(colorbar="horizontal")