from dune.grid import structuredGrid
from dune.alugrid import aluSimplexGrid
from dune.fem.space import dgonb, dglegendre, raviartThomas, bdm
from dune.fem.operator import galerkin
from ufl import TrialFunction, TestFunction, dx, dot

def getGridSpace1(element,order,dgspace):
    if element == "simplex":
        vertices = [ (0,0), (0,1), (1,1), (1,0), (0.5,0.5)]
        triangles = [ [0,1,4], [1,2,4], [2,3,4], [3,0,4] ]
        gridView = aluSimplexGrid({"vertices":vertices, "simplices":triangles})
    else:
        gridView = structuredGrid([0, 0], [1, 1], [2, 2])

    spaceDG = dgspace(gridView, order = order)
    return gridView, spaceDG

def getGridSpace2(element,space,order):
    if element == "simplex":
        vertices = [ (0,0), (0,1), (1,1), (1,0), (0.5,0.5)]
        triangles = [ [0,1,4], [1,2,4], [2,3,4], [3,0,4] ]
        gridView = aluSimplexGrid({"vertices":vertices, "simplices":triangles})
    else:
        gridView = structuredGrid([0, 0], [1, 1], [2, 2])

    if space == "RT":
        spaceHDiv = raviartThomas(gridView,order = order)
    else:
        spaceHDiv = bdm(gridView,order = order+1)
    if element == "simplex" or not space == "RT":
        spaceDG = dgonb(gridView, order = order)
    else:
        spaceDG = dglegendre(gridView, order = order)
    return gridView, spaceHDiv, spaceDG

def getMass(gridView, space):
    gridView.hierarchicalGrid.globalRefine()
    u = TrialFunction(space)
    v = TestFunction(space)
    return galerkin(dot(u,v)*dx).linear()

gv, sp = getGridSpace1("simplex",0,dgonb)
m = getMass(gv,sp)
gv, sp = getGridSpace1("cube",0,dglegendre)
m = getMass(gv,sp)

gv, spaceHDiv, spaceDG = getGridSpace2("cube", "RT", 0)
m = getMass(gv,spaceHDiv)
gv, spaceHDiv, spaceDG = getGridSpace2("simplex", "BDM", 0)
m = getMass(gv,spaceHDiv)
