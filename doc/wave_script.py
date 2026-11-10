from matplotlib import pyplot as plt
import numpy as np
import dune.fem as fem
from dune.grid import reader
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.space import lagrange as solutionSpace
from ufl import TrialFunction, TestFunction, SpatialCoordinate, grad, dot, dx
from dune.ufl import Constant, BoxDirichletBC, DirichletBC
from dune.fem.scheme import galerkin
from scipy.sparse import dia_matrix

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
pVec   = p.as_numpy
phiVec = phi.as_numpy
p_in = Constant(0.0, name="g")
# bc   = DirichletBC(V, p_in, conditional(x[0]<1e-10,1,0))
# bc   = BoxDirichletBC(V, p_in, [None,None],[0,None], eps=1e-10)
bc   = DirichletBC(V, p_in, 1)
op        = galerkin([dot(grad(u),grad(v))*dx==0,bc])
S         = op.linear().as_numpy
lapPhi    = V.interpolate(1,name="e")
lapPhiVec = lapPhi.as_numpy
M = galerkin(u*v*dx==0).linear().as_numpy
diag = M.sum(axis=1)
lumping = galerkin(u*v*dx==0)
lumped = lapPhi.copy()
lumping(lapPhi,lumped)       # note that lapPhi=1
N = len(lumped.as_numpy)
M = dia_matrix(([dt/lumped.as_numpy],[0]),shape=(N,N) )
S1 = M*S
lumping = galerkin(u*v*dx(metadata={"quadrature_rule":"lumped"})==0)
lumped = lumping.linear().as_numpy.diagonal()
M = dia_matrix(([dt/lumped],[0]),shape=(N,N) )
S2 = M*S
while t <= T:
    # first phi update (two possible approaches doing the same thing...)
    # phiVec[:] -= pVec[:]*dt/2
    phi.axpy(-dt/2, p)
    # multiply p with S and set values on the Dirichlet boundary
    pVec[:] += S2*phiVec[:]
    p_in.value = np.sin(2*np.pi*5*t)
    op.setConstraints(p)
    # second phi update
    phiVec[:] -= pVec[:]*dt/2
    # phi.axpy(-dt/2, p)
    # check for instabilities
    assert np.max(abs(pVec)) < 3
    # update time
    t += dt
p.plot(gridLines=None, clim=[-0.25,0.25], figure=plt.figure(figsize=(10,10)))