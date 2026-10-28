from matplotlib import ticker
from matplotlib import pyplot as plt
from dune.fem.plotting import plotComponents
import numpy, io
from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange as solutionSpace
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.function import gridFunction
from dune.ufl import DirichletBC, Constant
from ufl import TestFunction, TrialFunction, SpatialCoordinate,\
                dx, ds, grad, inner, sin, pi
from dune.fem.function import boundaryFunction
from dune.ufl import BoundaryId
from ufl import ds, grad, dot, inner, conditional, eq, And, FacetNormal, Not
import dune.alugrid

gridView = leafGridView([0, 0], [1, 1], [4, 4])
vecSpace = solutionSpace(gridView, dimRange=2, order=2)
x = SpatialCoordinate(vecSpace)
vec = vecSpace.interpolate([0,0], name='u_h')
uVec,vVec = TrialFunction(vecSpace), TestFunction(vecSpace)
a  = inner(grad(uVec), grad(vVec)) * dx
a += ( ( 10*(1-2*uVec[1])+uVec[0] ) * vVec[0] +\
       ( 10*sin(2*pi*x[1]) + (1-2*uVec[0])**2*uVec[1] ) * vVec[1]
     ) * dx
# Define Dirichlet boundary conditions for first component on all boundaries
bc = DirichletBC(vecSpace,[sin(2*pi*(x[0]+x[1])),None])
vecScheme = solutionScheme( [a == 0, bc],
        parameters={"linear.tolerance": 1e-9} )
vecScheme.solve(target=vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
bcBottom = DirichletBC(vecSpace,[sin(2*pi*(x[0]+x[1])),1],x[1]<1e-10)
vecScheme = solutionScheme( [a == 0, bc, bcBottom],
        parameters={"linear.tolerance": 1e-9} )
vecScheme.solve(target=vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
test = vec.copy()
test.clear()
@gridFunction(gridView ,name="bnd",order=2)
def bnd(x):
    return [numpy.sin(2*numpy.pi*(x[0]+x[1])),1]
bcBottom = DirichletBC(vecSpace,bnd,x[1]<1e-10)
vecScheme = solutionScheme( [a == 0, bc, bcBottom],
        parameters={"linear.tolerance": 1e-9} )
vecScheme.solve(target=test)
plotComponents(test, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
assert sum([ abs(td-vd) for td,vd in zip(test.dofVector, vec.dofVector)] ) /\
       len(vec.dofVector) < 1e-8
for i in vecScheme.dirichletIndices():
    vec.dofVector[i] = 0
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
vec.clear()
vecScheme.setConstraints(vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
vec.interpolate(bnd)
vecScheme.setConstraints([0,]*len(vec), vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
a  = a + uVec[0]*uVec[0] * vVec[1] * ds
vecScheme = solutionScheme( [a == 0, bc],
        parameters={"linear.tolerance": 1e-9} )
vecScheme.solve(target=vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
boundaryFunction(gridView).plot()
gridView.hierarchicalGrid.globalRefine(3)
space = solutionSpace(gridView, order=2)
x,n = SpatialCoordinate(space), FacetNormal(space)
u,v = TrialFunction(space), TestFunction(space)
a  = inner(grad(u), grad(v)) * dx
fbnd  = (u-10 * v *                    # middle of left and right boundary
         conditional(abs(x[1]-0.5)<0.1,1,0) ) * ds( (1,2) )
fbnd += 2 * v * ds(3)                  # bottom boundary
dbc   = DirichletBC(space, 1,          # middle of top boundary
           And( eq(BoundaryId(space),4), abs(x[0]-0.5)<0.2 ))
scheme   = solutionScheme( [a == fbnd, dbc] )
solution = space.interpolate(0, name='u_h')
scheme.solve(target=solution)
solution.plot()
fbnd  = (-50 * v *                     # middle of left and right boundary
         conditional(abs(x[1]-0.5)<0.1,1,0) ) * ds( (1,2) )
dbc   = [ DirichletBC(space, 1,        # not middle of left and right boundary
            And( And( BoundaryId(space)<=2,
                 abs(x[1]-0.5)>0.1 ), abs(x[1]-0.5)<0.3 ) ),
          DirichletBC(space, 3,        # not middle of left and right boundary
            Not( And(BoundaryId(space)<=2,abs(x[1]-0.5)<0.3) ) ), ]
scheme   = solutionScheme( [a == fbnd, *dbc] )
solution = space.interpolate(0, name='u_h')
scheme.solve(target=solution)
solution.plot()
gridView.hierarchicalGrid.globalRefine(1)
scheme.solve(target=solution)
solution.plot()
domain = {'vertices': [[0.        , 0.        ],
                       [1.        , 0.        ],
                       [1.        , 1.1       ],
                       [0.        , 1.1       ],
                       [0.25      , 0.        ],
                       [0.5       , 0.        ],
                       [0.75      , 0.        ],
                       [1.        , 0.19166667],
                       [1.        , 0.55      ],
                       [1.        , 0.74166667],
                       [0.75      , 1.1       ],
                       [0.5       , 1.1       ],
                       [0.25      , 1.1       ],
                       [0.        , 0.74166667],
                       [0.        , 0.55      ],
                       [0.        , 0.19166667],
                       [0.25      , 0.35833333],
                       [0.25      , 0.55      ],
                       [0.25      , 0.90833333],
                       [0.5       , 0.19166667],
                       [0.5       , 0.55      ],
                       [0.5       , 0.74166667],
                       [0.75      , 0.35833333],
                       [0.75      , 0.55      ],
                       [0.75      , 0.90833333]],
          'cubes': [[ 0,  4, 15, 16],
                    [15, 16, 14, 17],
                    [14, 17, 13, 18],
                    [13, 18,  3, 12],
                    [ 4,  5, 16, 19],
                    [16, 19, 17, 20],
                    [17, 20, 18, 21],
                    [18, 21, 12, 11],
                    [ 5,  6, 19, 22],
                    [19, 22, 20, 23],
                    [20, 23, 21, 24],
                    [21, 24, 11, 10],
                    [ 6,  1, 22,  7],
                    [22,  7, 23,  8],
                    [23,  8, 24,  9],
                    [24,  9, 10,  2]],
          'boundaries': [[2, 0, 4],
                         [2, 4, 5],
                         [2, 5, 6],
                         [2, 6, 1],
                         [2, 1, 7],
                         [2, 7, 8],
                         [2, 8, 9],
                         [2, 9, 2]],
          'defaultBndId': 5}
gridView = dune.alugrid.aluCubeGrid(domain)
gridView.plot()
fig, axs = plt.subplots(1, 2, figsize=(12,6))
boundaryFunction(gridView).plot(gridLines="white", linewidth=3, figure=(fig,axs[0]))
gridView.hierarchicalGrid.globalRefine(1)
boundaryFunction(gridView).plot(gridLines="white", linewidth=3, figure=(fig,axs[1]))