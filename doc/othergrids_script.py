import numpy
from matplotlib import pyplot
from dune.alugrid import aluSimplexGrid as leafGridView
from dune.grid import OutputType, reader
from dune.fem.space import lagrange as solutionSpace
from dune.fem.scheme import galerkin as solutionScheme
from ufl import TestFunction, TrialFunction, SpatialCoordinate
from ufl import dx, grad, grad, dot, inner, conditional, sin
from dune.grid.mesh2dgf import mesh2DGF
from dune.grid import reader
from dune.fem.function import boundaryFunction
import tempfile
import gmsh
import dune.alugrid
from dune.grid import onedGrid

domain2d = (reader.dgf, "triangle.dgf")
aluView = leafGridView(domain2d, dimgrid=2)
aluView.plot(figsize=(5,5))
waveDomain = (reader.gmsh, "wave_tank.msh")
waveGrid = leafGridView(waveDomain, dimgrid=2)
waveGrid.plot(figure=pyplot.figure(figsize=(10,10)))
try:
    import pygmsh
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon([
              [ 0.0,  0.5, 0.0], [-0.1,  0.1, 0.0], [-0.5,  0.0, 0.0],
              [-0.1, -0.1, 0.0], [ 0.0, -0.5, 0.0], [ 0.1, -0.1, 0.0],
              [ 0.5,  0.0, 0.0], [ 0.1,  0.1, 0.0] ], mesh_size=0.05)
        geom.twist(
            poly,
            translation_axis=[0, 0, 1], rotation_axis=[0, 0, 1], point_on_axis=[0, 0, 0],
            angle=numpy.pi / 3,
        )
        mesh = geom.generate_mesh(verbose=False)
    points, cells = mesh.points, mesh.cells_dict
    domain3d = {"vertices":points.astype("float"), "simplices":cells["tetra"]}
except ImportError: # pygmsh not installed - use a simple cartesian domain
    print("pygmsh module not found using a simple Cartesian domain - ignored")
    from dune.grid import cartesianDomain
    domain3d = cartesianDomain([-0.25,-0.25,0],[0.25,0.25,1],[30,30,60])
gridView3d  = leafGridView(domain3d)
space3d = solutionSpace(gridView3d, order=1)
u = TrialFunction(space3d)
v = TestFunction(space3d)
x = SpatialCoordinate(space3d)
scheme3d = solutionScheme((inner(grad(u),grad(v))+inner(u,v))*dx ==
                        conditional(dot(x,x)<.01,100,0)*v*dx,
                        solver='cg')
uh3d = space3d.interpolate([0],name="solution")
scheme3d.solve(target=uh3d)
gridView3d.writeVTK('3dexample', pointdata=[uh3d],
                    outputType=OutputType.appendedraw)
dgf, _,_ = mesh2DGF((points, cells))
domain3d = (reader.dgfString, dgf)
gridView3d  = leafGridView(domain3d, dimgrid=3)
with open ("3dmesh.dgf", "w") as file:
    file.write(dgf)
try:
    import numpy as np
    import meshio
    mesh = meshio.read("wave_tank.msh")
    points = np.delete( mesh.points, 2, 1) # remove the z component from the points
    cells = mesh.cells_dict
    waveDomain = {"vertices":points.astype("float"), "simplices":cells["triangle"]}
    waveGrid = leafGridView(waveDomain)
    waveGrid.plot(figure=pyplot.figure(figsize=(10,10)),xlim=[0.3,1.7],ylim=[-0.2,1.2])
except ImportError:
    print("This example requires the meshio package which can be installed using pip.")
if meshio:
    from dune.alugrid import aluCubeGrid
    mesh = meshio.read("quads.msh")
    points2d = np.delete(mesh.points,2,1)
    cells = mesh.cells_dict['quad']
    cells[:, [2, 3]] = cells[:, [3, 2]]
    domain = {"vertices":points2d, "cubes":cells}
    quadGrid  = aluCubeGrid(domain)
    quadGrid.plot()
try:
    with pygmsh.occ.Geometry() as geom:
        L = 1; H = 1
        c = 0.2; r = 0.05
        def size(dim, tag, x, y, z, lc):
            resolution = 0.1
            d = ((x - c)**2 + (y - c)**2)**0.5
            if d < 3 * r:
                return 0.4*resolution
            d = ((x - (1-c))**2 + (y - (1-c))**2)**0.5
            if d < 3 * r:
                return 0.4*resolution
            return resolution
        rectangle = geom.add_rectangle([0, 0, 0], L, H)
        obs1 = geom.add_ball([c, c, 0], r)
        obs2 = geom.add_ball([1-c, 1-c, 0], r)
        geom.boolean_difference(rectangle, [obs1, obs2])
        geom.set_mesh_size_callback(size)
        mesh = geom.generate_mesh(dim=2)
        eps = 0.01 # tolerance
        c1 = [c - r - eps, c + r + eps]
        c2 = [(1-c) - r - eps, (1-c) + r + eps]
        bndDomain = {1: [[-1.0 , -1.0 ], [0.0  , H+0.1]], # bounding box inflow
                     2: [[   L , -1.0 ], [L+1.0, H+0.1]], # bounding box outflow
                     4: [[c1[0], c1[0]], [c1[1], c1[1]]], # bounding box lower left circle
                     5: [[c2[0], c2[0]], [c2[1], c2[1]]], # bounding box upper right circle
                     3: "default"  # top and bottom wall,
                    }
        dgf, _,_ = mesh2DGF((mesh.points, mesh.cells_dict), bndDomain=bndDomain, dim=2)
        domain2d = (reader.dgfString, dgf)
except ImportError: # pygmsh not installed - use a simple cartesian domain
    print("pygmsh module not found using a simple Cartesian domain - ignored")
    from dune.grid import cartesianDomain
    domain2d = cartesianDomain([-0.25,-0.25],[0.25,0.25],[30,30])
gridView2d = leafGridView(domain2d, dimgrid=2)
fig = pyplot.figure()
boundaryFunction( gridView2d).plot(gridLines="white",linewidth=2,figure=fig)
if len( fig.get_axes() ) > 0:
    fig.get_axes()[0].set_facecolor("lightgray")
with pygmsh.geo.Geometry() as geom:
    L,H = 1, 1
    c,r = 0.2, 0.05
    def size(dim, tag, x, y, z, lc):
        resolution = 0.1
        d = ((x - c)**2 + (y - c)**2)**0.5
        if d < 3 * r:
            return 0.4*resolution
        d = ((x - (1-c))**2 + (y - (1-c))**2)**0.5
        if d < 3 * r:
            return 0.4*resolution
        return resolution
    obs1 = geom.add_circle([c,c,0], r)
    obs2 = geom.add_circle([1-c,1-c,0], r)
    points = [
        geom.add_point((0, 0, 0)),
        geom.add_point((L, 0, 0)),
        geom.add_point((L, H, 0)),
        geom.add_point((0, H, 0))
      ]
    rectangle = [
        geom.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
      ]
    rect_loop = geom.add_curve_loop(rectangle)
    surface = geom.add_plane_surface(rect_loop, holes=[obs1.curve_loop,obs2.curve_loop])
    geom.synchronize()
    geom.add_physical([rectangle[0]], "Inflow")                 # 1
    geom.add_physical([rectangle[2]], "Outflow")                # 2
    geom.add_physical([surface], "Fluid")                       # 3 (will be ignored during import)
    geom.add_physical(obs1.curve_loop.curves, "Obstacle1")      # 4
    geom.add_physical(obs2.curve_loop.curves, "Obstacle2")      # 5
    geom.set_mesh_size_callback(size)
    geom.generate_mesh(dim=2, verbose=True, algorithm=8)
    gmsh.model.mesh.recombine()
    with tempfile.NamedTemporaryFile(suffix=".msh") as f:
        gmsh.write(f.name)
        gridView = dune.alugrid.aluGrid((reader.meshio,f.name),defaultBndId=3)
fig = pyplot.figure()
boundaryFunction(gridView).plot(gridLines="white",linewidth=2, figure=fig)
if len( fig.get_axes() ) > 0:
    fig.get_axes()[0].set_facecolor("lightgray")
gridView = dune.alugrid.aluGrid((reader.meshio,"wave_tank.msh"))
boundaryFunction(gridView).plot(gridLines="white", linewidth=1,
                figure=pyplot.figure(figsize=(16,10)),xlim=[-0.05,1.2],ylim=[-0.1,0.6] )
v = [numpy.log(i) for i in range(1,100)]
e = [(i,i+1) for i in range(1,len(v))]
g = onedGrid(constructor={"vertices":v, "simplices":e})
s = solutionSpace(g)
x = SpatialCoordinate(s)
u = s.interpolate(sin(x[0]), name="u")
u.plot()
pyplot.plot(g.tessellate()[0], u.pointData(), '-p')