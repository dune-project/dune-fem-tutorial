import ufl
import dune.ufl
import dune.grid
import dune.fem
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import linspace

dimRange   = 1
dt         = 0.25
linearSpiral = True
if linearSpiral:
    spiral_a   = 0.75
    spiral_b   = 0.02
    spiral_eps = 0.02
    spiral_D   = 1./100
    def spiral_h(u,v): return u - v
else:
    spiral_a   = 0.75
    spiral_b   = 0.0006
    spiral_eps = 0.08
    def spiral_h(u,v): return u**3 - v
x = ufl.SpatialCoordinate(dune.ufl.domain(2))
initial_u = ufl.conditional(x[1]>1.25,1,0)
initial_v = ufl.conditional(x[0]<1.25,0.5,0)
gridView = dune.grid.structuredGrid([0,0],[2.5,2.5],[30,30])
space    = dune.fem.space.lagrange( gridView, dimRange=dimRange, order=1 )
uh   = space.interpolate( initial_u, name="u" )
uh_n = uh.copy()
vh   = space.interpolate( initial_v, name="v" )
vh_n = vh.copy()
u   = ufl.TrialFunction(space)
phi = ufl.TestFunction(space)
# right hand side (time derivative part
a_ex = ufl.inner(uh_n, phi) * ufl.dx
# left hand side (heat equation in first variable + backward Euler in time)
a_im = (dt * spiral_D * ufl.inner(ufl.grad(u), ufl.grad(phi)) +
        ufl.inner(u,phi)) * ufl.dx
ustar = (vh_n[0]+spiral_b)/spiral_a
a_ex += ufl.conditional(uh_n[0]<ustar, dt/spiral_eps* u[0]*(1-uh_n[0])*(uh_n[0]-ustar),
                                       dt/spiral_eps*uh_n[0]*(1-u[0]) *(uh_n[0]-ustar) ) * phi[0] * ufl.dx
equation   = a_im == a_ex
ode_update = ufl.as_vector([ vh_n[0] + dt*spiral_h(uh_n[0], vh_n[0]) ])
solverParameters =\
       {"nonlinear.tolerance": 1e-3,
        "nonlinear.verbose": False,
        "linear.tolerance": 1e-5,
        "linear.preconditioning.method": "jacobi",
        "linear.verbose": False}
scheme = dune.fem.scheme.galerkin( equation, space, solver="cg", parameters=solverParameters)
def init():
    data = uh.pointData(1)
    C = plt.tricontourf(triangulation, data[:,0], cmap=plt.cm.rainbow, levels=levels)
    return C.axes.collections
def animate(count):
    global t,stepsize,nextstep
    nextstep += stepsize
    # print(count,t,stepsize,nextstep)
    while t < nextstep:
        uh_n.assign(uh)
        vh_n.assign(vh)
        info = scheme.solve(target=uh)
        vh.interpolate( ode_update )
        # print("Computing solution a t = " + str(t + dt), "iterations: " + str(info["linear_iterations"]) )
        t += dt
    data = uh.pointData(1)
    C = plt.tricontourf(triangulation, data[:,0], cmap=plt.cm.rainbow, levels=levels)
    # gridView.writeVTK("spiral", pointdata=[uh], number=count)
    return C.axes.collections
fig, ax = plt.subplots()
ax.set_xlim(( 0, 2.5))
ax.set_ylim(( 0, 2.5))
triangulation = gridView.triangulation(1)
levels = linspace(-0.1, 1.1, 256)
ax.set_aspect('equal');
t        = 0.
stepsize = 0.5
nextstep = 0.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=20, interval=100, blit=True)
try:
    movie = anim.to_html5_video()
    # save and embed
    anim.save('spiral.gif', writer=animation.ImageMagickWriter(fps=2))
    from IPython.display import HTML, display
    display( HTML(movie) )
except: # ffmpeg probably missing
    try:
        anim.save("spiral.html")
        from IPython.display import IFrame
        IFrame(src='./spiral.html', width=700, height=600)
    except:
        pass