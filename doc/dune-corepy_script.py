import time, numpy, sys
import matplotlib.pyplot as pyplot
import dune.fem
from dune.common import FieldVector
import dune.geometry
from dune.grid import cartesianDomain, yaspGrid
from dune.alugrid import aluConformGrid
from dune.grid import tensorProductCoordinates
from dune.grid import Marker
from dune.alugrid import aluSimplexGrid
from dune.geometry import quadratureRules

x = FieldVector([0.25,0.25,0.25])
print(x)
print(numpy.array(x))
geometryType = dune.geometry.simplex(2)
referenceElement = dune.geometry.referenceElement(geometryType)
print("\t".join(str(c) for c in referenceElement.corners))
for p in dune.geometry.quadratureRule(geometryType, 3):
    print("position:",p.position,f"({type(p.position)})", "weight:",p.weight,f"({type(p.weight)})")
points,weights = dune.geometry.quadratureRule(geometryType, 3).get()
print("quadrature points:",points,f"({type(points)},{points.shape})")
print("quadrature weights:",weights,f"({type(weights)},{weights.shape})")
domain = cartesianDomain([0, 0], [1, 0.25], [15, 4])
yaspView = yaspGrid(domain)
yaspView.plot()
yaspView.hierarchicalGrid.globalRefine()
yaspView.plot()
aluView = aluConformGrid(domain)
aluView.plot()
aluView.hierarchicalGrid.globalRefine()
aluView.plot()
aluView.hierarchicalGrid.globalRefine()
aluView.plot()
domain = cartesianDomain([0, 0], [1, 0.25], [8, 2])
yaspView = yaspGrid(domain)
aluView = aluConformGrid(domain)
yaspView.hierarchicalGrid.globalRefine(2)
yaspView.plot()
aluView.hierarchicalGrid.globalRefine(4)
aluView.plot()
cst = dune.grid.tensorProductCoordinates([[1,2,4,8,16],[6,8,10,12,14,16]])
yaspView = dune.grid.yaspGrid(cst)
yaspView.plot()
vertices = [(0,0), (1,0), (1,0.6), (0,0.6), (-1,0.6), (-1,0), (-1,-0.6), (0,-0.6)]
triangles = [(2,0,1), (0,2,3), (4,0,3), (0,4,5), (6,0,5), (0,6,7)]
aluView = aluConformGrid({"vertices": vertices, "simplices": triangles})
aluView.plot(figsize=(5,5))
aluView.hierarchicalGrid.globalRefine(2)
aluView.plot(figsize=(5,5))
aluView.plot(figsize=(5,5))
for i in range(1,4):
    def mark(e):
        x = e.geometry.center
        return Marker.refine if x.two_norm < 0.64**i else Marker.keep
    aluView.hierarchicalGrid.adapt(mark)
    aluView.plot(figsize=(5,5))
vertices = aluView.coordinates()
triangles = [aluView.indexSet.subIndices(e, 2) for e in aluView.elements]
aluView = aluSimplexGrid({"vertices": vertices, "simplices": triangles})
vertices = [(0,0), (1,0), (1,1), (0,1)]
triangles = [(2,0,1), (0,2,3)]
unitSquare = aluSimplexGrid({"vertices": vertices, "simplices": triangles})
print(unitSquare.size(0),"elements and",unitSquare.size(2),"vertices")
for codim in range(0, unitSquare.dimension+1):
    for entity in unitSquare.entities(codim):
        print(", ".join(str(c) for c in entity.geometry.corners))
for element in unitSquare.elements:
    print(", ".join(str(element.geometry.center)))
for edge in unitSquare.edges:
    print(", ".join(str(c) for c in edge.geometry.corners))
for element in unitSquare.elements:
    print("Element vertices ")
    for vertex in element.subEntities(unitSquare.dimension):
        print(f" {vertex.geometry.center}")
for element in unitSquare.elements:
    for isec in unitSquare.intersections( element ):
        if isec.outside:
            print(f"outside {isec.outside.geometry.center}")
        if isec.boundary:
            print(f"boundary {isec.geometry.center} with normal {isec.centerUnitOuterNormal}")
@dune.fem.function.gridFunction(aluView, name="cos", order=3)
def f(x):
    return numpy.cos(2.*numpy.pi/(0.3+abs(x[0]*x[1])))
@dune.fem.function.gridFunction(aluView, name="hat0", order=3)
def hat0(element,hatx):
    return 1-hatx[0]-hatx[1]
hatx = FieldVector([1./3., 1./3.])
maxValue = max(f(e, hatx) for e in f.gridView.elements)
maxValue = max(f(e.geometry.toGlobal(hatx)) for e in f.gridView.elements)
hat0.plot()
f.plot()
f.plot(level=3,xlim=[0,0.4], ylim=[0,0.4])
rules = quadratureRules(5)
integral = 0
for e in aluView.elements:
    geo = e.geometry
    for qp in rules(e.type):
        x,w = qp.position, qp.weight
        integral += f(e,x)*w*geo.integrationElement(x)
print("integral of function 'f':",integral)
integral = 0
for e in aluView.elements:
    points, weights = rules(e.type).get()
    ies = e.geometry.integrationElement(points)
    values = f(e,points)
    integral += numpy.sum(values*ies*weights,axis=-1)
print("integral of function 'f':",integral)
dimR = 2
@dune.fem.function.gridFunction(aluView, name="g", order=3)
def g(x):
    return [numpy.cos(2.*numpy.pi/(0.3+abs(x[0]*x[1])))]*dimR
@dune.fem.function.gridFunction(aluView, name="g", order=3)
def g(x):
    if type(x[0]) is float:
        return [numpy.cos(2.*numpy.pi/(0.3+abs(x[0]*x[1]))),1]
    else:
        return [numpy.cos(2.*numpy.pi/(0.3+abs(x[0]*x[1]))),[1]*len(x[0])]
integral = g.dimRange*[0]
for e in aluView.elements:
    points, weights = rules(e.type).get()
    ies = e.geometry.integrationElement(points)
    values = g(e,points)
    integral += numpy.sum(values*ies*weights,axis=-1)
print("integral of function 'g':",integral)
layout = [2,4,3]
mapper = unitSquare.mapper([2, 4, 3])
print( mapper.size, sum([layout[c]*unitSquare.size(c) for c in range(3)]) )
layout = {dune.geometry.quadrilateral: 4, dune.geometry.triangle: 1}
mapper = unitSquare.mapper(layout)
def interpolate(grid):
    mapper = grid.mapper({dune.geometry.vertex: 1})
    data = numpy.zeros(mapper.size)
    for v in grid.vertices:
        data[mapper.index(v)] = f(v.geometry.center)
    return mapper, data
mapper, data = interpolate(aluView)
@dune.fem.function.gridFunction(aluView, name="p12d", order=1)
def p12dEvaluate(e, x):
    bary = 1-x[0]-x[1], x[0], x[1]
    idx = mapper.subIndices(e, 2)
    return sum(b * data[i] for b, i in zip(bary, idx))
@dune.fem.function.gridFunction(aluView, name="error", order=3)
def error(e, x):
    return abs(p12dEvaluate(e, x)-f(e, x))
hatx = FieldVector([1./3., 1./3.])
print(max(error(e, hatx) for e in aluView.elements))
for i,e in enumerate(aluView.elements):
    print(mapper(e))
    if i==4: break
p12dEvaluate.plot(figsize=(9,9), gridLines=None, colorbar="horizontal")
f.plot(level=2, figsize=(9,9), gridLines=None, colorbar="horizontal")
error.plot(level=2, figsize=(9,9), gridLines=None, colorbar="horizontal")
pd = {"exact": f, "discrete": p12dEvaluate, "error": error}
aluView.writeVTK("interpolation", pointdata=pd)
aluView.writeVTK("interpolation_subsampled", subsampling=2, pointdata=pd)
level = 3
triangulation = f.gridView.triangulation(level)
z = f.pointData(level)[:,0]
try:
    from mayavi import mlab
    from mayavi.tools.notebook import display
    mlab.init_notebook("png")
    mlab.figure(bgcolor = (1,1,1))
    s = mlab.triangular_mesh(triangulation.x, triangulation.y, z*0.5,
                             triangulation.triangles)
    display( s )
    mlab.close(all=True)
except (ImportError, ModuleNotFoundError):
    print("mayavi module not found so not rendering plot - ignored")
    pass