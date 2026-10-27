from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import dia_matrix

import dune.common
import dune.fem as fem
from dune.grid import reader
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.space import lagrange as solutionSpace
from ufl import TrialFunction, TestFunction, SpatialCoordinate, grad, dot, dx
from dune.ufl import Constant, BoxDirichletBC, DirichletBC
from dune.fem.function import levelFunction, partitionFunction
from dune.fem.scheme import galerkin

T = 3
dt = 0.005
t = 0

domain = (reader.meshio, "wave_tank.msh")
gridView  = leafGridView( domain, dimgrid=2 )
gridView.hierarchicalGrid.loadBalance()
V = solutionSpace(gridView, order=1, storage="numpy")

u    = TrialFunction(V)
v    = TestFunction(V)
x    = SpatialCoordinate(V)

p      = V.interpolate(0,name="p")
phi    = V.interpolate(0,name="phi")

p_in = Constant(0.0, name="g")
bc   = DirichletBC(V, p_in, 1)

op        = galerkin([dot(grad(u),grad(v))*dx==0,bc])
S         = op.linear()
lapPhi    = V.interpolate(1,name="e")
lapPhiVec = lapPhi.as_numpy

lumping = galerkin(u*v*dx(metadata={"quadrature_rule":"lumped"})==0)
Mlumped = lumping.linear()

# 'Mlumped' is a diagonal matrix from which we need the inverse. To make
# this as efficient as possible we store the inverse in a scipy diagonal
# matrix. An issue is that the linear operator 'Mlumped' is additively
# distributed, so we first need to sum up diagonal entries shared over
# multiple processors. The linear operator itself does not provide a
# suitable communication method but we can interpret the diagonal entries
# as a discrete function and use the communication method there:
lumped = V.function(name="lump", dofVector=Mlumped.as_numpy.diagonal())
lumped.communicate()
N = lumped.size
Minv = dia_matrix(([dt/lumped.as_numpy],[0]),shape=(N,N) )

saveInterval = T/20
nextSaveTime = saveInterval
# vtk = gridView.sequencedVTK("wave", pointdata=[p,phi],
#                celldata=[levelFunction(gridView), partitionFunction(gridView)])
# vtk()

tmp = phi.copy()
while t <= T:
    phi.axpy(-dt/2, p)

    S(phi,tmp)
    p.as_numpy[:] += Minv*tmp.as_numpy[:]

    p_in.value = np.sin(2*np.pi*5*t)
    op.setConstraints(p)

    phi.axpy(-dt/2, p)

    if t > nextSaveTime-dt/4:
        # vtk()
        nextSaveTime += saveInterval
        # check for instabilities
        assert np.max(abs(p.as_numpy[:])) < 3

    # update time
    t += dt

# vtk()
fig = plt.figure(figsize=(10,10))
p.plot(gridLines=None, clim=[-0.25,0.25], figure=fig)
fig.savefig(f"wave_mpi{dune.common.comm.rank}.png", dpi=150, bbox_inches="tight")
