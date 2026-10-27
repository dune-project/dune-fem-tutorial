import matplotlib.pyplot as plt
import numpy as np
from ufl import *
from dune.alugrid import aluSimplexGrid
from dune.fem.space import lagrange
from dune.ufl import DirichletBC
from scipy.sparse.linalg import spsolve
from dune.fem import assemble
from dune.fem import integrate
from math import sqrt

vertices = np.array( [(0,0), (1,0), (1,1), (0,1),(0.5,0.5)] )
triangles = np.array( [(0,1,4), (1,2,4), (2,4,3), (3,4,0)] )
gridView = aluSimplexGrid({"vertices": vertices, "simplices": triangles})
gridView.hierarchicalGrid.globalRefine(1)
order = 2
space = lagrange(gridView, order=order)
u = TrialFunction(space)
v = TestFunction(space)
x = SpatialCoordinate(space)
exactFunction = sin(pi*x[0]) ** 2 * sin(pi*x[1])**2
laplace = lambda fn : div(grad(fn))
f = laplace(laplace(exactFunction))
a = laplace(u) * laplace(v)*dx - f*v * dx
normal = FacetNormal(space)
a+= - avg(div(grad(u))) * jump(grad(v),normal) * dS
a+= - avg(div(grad(v))) * jump(grad(u),normal) * dS # a symmetrisation
a+= - div(grad(u)) * inner(grad(v),normal) * ds
a+= - div(grad(v)) * inner(grad(u),normal) * ds
beta = 10 * order**2 / FacetArea(space)
a+= beta * inner(jump(grad(u),normal),jump(grad(v),normal))*dS
dbc = DirichletBC(space,exactFunction)
a+= beta * inner(grad(u-exactFunction),normal) * inner(grad(v),normal)*ds
solution = space.function(name = "discrete_solution")
aa, ff = assemble([a ==0, dbc])
solution.as_numpy[:] = spsolve( aa.as_numpy, ff.as_numpy )
dofs = [space.size]
errorL2 = [ sqrt(integrate((solution - exactFunction)**2, gridView, order = 2*order)) ]
errorH1 = [ sqrt(integrate(inner(grad( solution - exactFunction),grad( solution - exactFunction) ), gridView, order = 2*(order-1) ))]
errorH2 = [ sqrt(integrate(laplace(solution - exactFunction)**2, gridView, order = 2*(order-1)) + sum(assemble(jump(grad(solution),normal)**2*avg(v)*dS).as_numpy ) + sum(assemble(inner(grad(solution),normal)**2*v*ds).as_numpy )) ] # need to get the error
maxRefs = 3
for _ in range(maxRefs):
    gridView.hierarchicalGrid.globalRefine()
    aa, ff = assemble([a ==0, dbc])
    solution.as_numpy[:] = spsolve( aa.as_numpy, ff.as_numpy )
    dofs.append(space.size)
    errorL2.append( sqrt(integrate((solution - exactFunction)**2, gridView, order = 2*order) ))
    errorH1.append( sqrt(integrate(inner(grad( solution - exactFunction),grad( solution - exactFunction) ), gridView, order = 2*(order-1) )) )
    errorH2.append( sqrt(integrate(laplace(solution - exactFunction)**2, gridView, order = 2*(order-1)) + sum(assemble(jump(grad(solution),normal)**2*avg(v)*dS).as_numpy ) + sum(assemble(inner(grad(solution),normal)**2*v*ds).as_numpy ) ))
for i in range(len(dofs)):
    print(dofs[i], "\t", errorL2[i], "\t", errorH1[i], "\t", errorH2[i])
plt.plot( dofs, errorL2, label = "L2", marker = 'x' )
plt.plot( dofs, errorH1, label = "H1", marker = 'x' )
plt.plot( dofs, errorH2, label = "H2", marker = 'x' )
plt.plot( dofs, [(d/dofs[0])**(-1) for d in dofs], label = r"dofs$^{-1}$")
plt.plot( dofs, [(d/dofs[0])**(-1/2) for d in dofs], label = r"dofs$^{-1/2}$")
plt.legend()
plt.loglog();
gridView.hierarchicalGrid.globalRefine(-maxRefs)
exactFunction = sin(pi*x[0]) * sin(pi*x[1])
f = laplace(laplace(exactFunction))
a = laplace(u) * laplace(v)*dx - f*v * dx
a+= - avg(div(grad(u))) * jump(grad(v),normal) * dS
a+= - avg(div(grad(v))) * jump(grad(u),normal) * dS
beta = 10 * order**2 / FacetArea(space)
a+= beta * inner(jump(grad(u),normal),jump(grad(v),normal))*dS
aa, ff = assemble([a ==0, dbc])
solution.as_numpy[:] = spsolve( aa.as_numpy, ff.as_numpy )
dofs = [space.size]
errorL2 = [ sqrt(integrate((solution - exactFunction)**2, gridView, order = 2*order)) ]
errorH1 = [ sqrt(integrate(inner(grad( solution - exactFunction),grad( solution - exactFunction) ), gridView, order = 2*(order-1) ))]
errorH2 = [ sqrt(integrate(laplace(solution - exactFunction)**2, gridView, order = 2*(order-1)) + sum(assemble(jump(grad(solution),normal)**2*avg(v)*dS).as_numpy ) ) ]
for _ in range(maxRefs):
    gridView.hierarchicalGrid.globalRefine()
    aa, ff = assemble([a ==0, dbc])
    solution.as_numpy[:] = spsolve( aa.as_numpy, ff.as_numpy )
    dofs.append(space.size)
    errorL2.append( sqrt(integrate((solution - exactFunction)**2, gridView, order = 2*order) ))
    errorH1.append( sqrt(integrate(inner(grad( solution - exactFunction),grad( solution - exactFunction) ), gridView, order = 2*(order-1) )) )
    errorH2.append( sqrt(integrate(laplace(solution - exactFunction)**2, gridView, order = 2*(order-1)) + sum(assemble(jump(grad(solution),normal)**2*avg(v)*dS).as_numpy ) ))
for i in range(len(dofs)):
    print(dofs[i], "\t", errorL2[i], "\t", errorH1[i], "\t", errorH2[i])
plt.plot( dofs, errorL2, label = "L2", marker = 'x' )
plt.plot( dofs, errorH1, label = "H1", marker = 'x' )
plt.plot( dofs, errorH2, label = "H2", marker = 'x' )
plt.plot( dofs, [(d/dofs[0])**(-1) for d in dofs], label = r"dofs$^{-1}$")
plt.plot( dofs, [(d/dofs[0])**(-1/2) for d in dofs], label = r"dofs$^{-1/2}$")
plt.legend()
plt.loglog();