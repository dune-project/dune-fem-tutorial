from matplotlib import pyplot
import random
from dune.grid import cartesianDomain, gridFunction
import dune.fem
from dune.fem.plotting import plotPointData as plot
from dune.fem.function import discreteFunction
from ufl import *
import dune.ufl

try:
    import dune.vem
except:
    print("This example needs 'dune.vem' - skipping")
    import sys
    sys.exit(0)
dune.fem.threading.use = 4
order        = 3
polyGrid = dune.vem.polyGrid( cartesianDomain([0,0],[1,1],[30,30]), cubes=True)
testSpaces = [ [0], [order-3,order-2], [order-4] ]
space = dune.vem.vemSpace(polyGrid, order=order, testSpaces=testSpaces)
t     = dune.ufl.Constant(0,"time")
tau   = dune.ufl.Constant(0,"dt")
eps   = dune.ufl.Constant(0,"eps")
df_n  = discreteFunction(space, name="oldSolution") # previous solution
x     = SpatialCoordinate(space)
u     = TrialFunction(space)
v     = TestFunction(space)
H     = lambda v: grad(grad(v))
laplace = lambda v: H(v)[0,0]+H(v)[1,1]
a     = lambda u,v: inner(H(u),H(v))
b     = lambda u,v: inner( grad(u),grad(v) )
W     = lambda v: 1/4*(v**2-1)**2
dW    = lambda v: (v**2-1)*v
equation = ( u*v + tau*eps*eps*a(u,v) + tau*b(dW(u),v) ) * dx == df_n*v * dx
dbc = [dune.ufl.DirichletBC(space, 0, i+1) for i in range(4)]
Eint  = lambda v: eps*eps/2*inner(grad(v),grad(v))+W(v)
biLaplaceCoeff = eps*eps*tau
diffCoeff      = 2*tau
massCoeff      = 1
scheme = dune.vem.vemScheme(
                       [equation, *dbc],
                       solver=("suitesparse","umfpack"),
                       hessStabilization=biLaplaceCoeff,
                       gradStabilization=diffCoeff,
                       massStabilization=massCoeff,
                       boundary="derivative") # only fix the normal derivative = 0
def initial(x):
    h = 0.01
    g0 = lambda x,x0,T: conditional(x-x0<-T/2,0,conditional(x-x0>T/2,0,sin(2*pi/T*(x-x0))**3))
    G  = lambda x,y,x0,y0,T: g0(x,x0,T)*g0(y,y0,T)
    return 0.5*G(x[0],x[1],0.5,0.5,50*h)
initial = dune.fem.space.lagrange(polyGrid,order=1).interpolate(initial(x),name="initial")
df = space.interpolate(initial, name="solution")
t.value = 0
eps.value = 0.05
fig = pyplot.figure(figsize=(30,30))
count = 0
pos = 1
while t.value < 2.15:
    if t.value < 0.6:
        tau.value = 1e-02
    else:
        tau.value = 5e-02
    df_n.assign(df)
    info = scheme.solve(target=df)
    t.value += tau
    count += 1
    if count % 10 == 0:
        df.plot(figure=(fig,330+pos),colorbar=None,clim=[-1,1])
        energy = dune.fem.integrate(Eint(df),order=3)
        print("[",pos,"]",t.value,tau.value,energy,info,flush=True)
        pos += 1