# %% [markdown]
#
# # Alternative Linear Solvers (Scipy and Petsc)
# Here we look at different ways of solving PDEs using external
# packages and python functionality.
# Different linear algebra backends can be accessed by changing setting the
# `storage` parameter during construction of the discrete space. All
# discrete functions and operators/schemes based on this space will then
# use this backend. Available backends are `numpy,istl,petsc`. The default is
# `numpy` which uses simple data structures and linear solvers implemented in
# the `dune-fem` package. The simplicity of the data structure makes it
# possible to use the buffer protocol to seamlessly move between C++ and
# Numpy/Scipy data structures on the python side. A degrees of freedom
# vector (dof vector) can be retrieved from a discrete function over the
# `numpy` space by using the `as_numpy` method. Similar methods are available
# for the other storages, i.e., `as_istl,as_petsc`. The same methods are
# also available to retrieve the underlying matrix structures of linear
# operators.
#
# We will mostly revisit the nonlinear time dependent problem studied at the end of the
# [concepts section](concepts_nb.ipynb) which after discretizing in time had the variational formulation
# \begin{equation}
# \begin{split}
# \int_{\Omega} \frac{u^{n+1}-u^n}{\Delta t} \varphi
# + \frac{1}{2}K(\nabla u^{n+1}) \nabla u^{n+1} \cdot \nabla \varphi \
# + \frac{1}{2}K(\nabla u^n) \nabla u^n \cdot \nabla \varphi v\ dx \\
# - \int_{\Omega} \frac{1}{2}(f(x,t^n)+f(x,t^n+\Delta t) \varphi\ dx
# - \int_{\partial \Omega} \frac{1}{2}(g(x,t^n)+g(x,t^n+\Delta t)) v\ ds
# = 0.
# \end{split}
# \end{equation}
# on a domain $\Omega=[0,1]^2$. We choose $f,g$ so that the exact solution
# is given by
# \begin{align*}
# u(x,t) = e^{-2t}\left(\frac{1}{2}(x^2 + y^2) - \frac{1}{3}(x^3 - y^3)\right) + 1
# \end{align*}
# The following code was described in the [concepts section](concepts_nb.ipynb)

# %%
import numpy, sys, io
import matplotlib.pyplot as plt

from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange as solutionSpace
from dune.fem.scheme import galerkin as solutionScheme
from dune.fem.function import gridFunction
from dune.fem import integrate
from dune.ufl import Constant
from ufl import TestFunction, TrialFunction, SpatialCoordinate, FacetNormal, \
                dx, ds, div, grad, dot, inner, sqrt, exp, sin,\
                conditional

gridView = leafGridView([0, 0], [1, 1], [4, 4])
space = solutionSpace(gridView, order=2)

x = SpatialCoordinate(space)
initial = 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
exact   = lambda t: exp(-2*t)*(initial - 1) + 1

u_h   = space.interpolate(initial, name='u_h')
u_h_n = u_h.copy(name="previous")

u = TrialFunction(space)
v = TestFunction(space)
dt = Constant(0, name="dt")    # time step
t  = Constant(0, name="t")     # current time

abs_du = lambda u: sqrt(inner(grad(u), grad(u)))
K = lambda u: 2/(1 + sqrt(1 + 4*abs_du(u)))
a = ( dot((u - u_h_n)/dt, v) \
    + 0.5*dot(K(u)*grad(u), grad(v)) \
    + 0.5*dot(K(u_h_n)*grad(u_h_n), grad(v)) ) * dx

f = lambda s: -2*exp(-2*s)*(initial - 1) - div( K(exact(s))*grad(exact(s)) )
g = lambda s: K(exact(s))*grad(exact(s))
n = FacetNormal(space)
b = 0.5*(f(t)+f(t+dt))*v*dx + 0.5*dot(g(t)+g(t+dt),n)*v*ds

# %% [markdown]
# When creating a scheme, it is possible to set the linear solver as well as
# parameters for the internal Newton solver and the linear solver
# and preconditioning. See a list of available solvers and preconditioning
# methods at the end of this section.

# %%
scheme = solutionScheme(a == b, solver='cg')

endTime    = 0.25
exact_end  = exact(endTime)
l2error = gridFunction(name="l2error", expr=dot(u_h - exact_end, u_h - exact_end))
h1error = gridFunction(name="h1error", expr=dot(grad(u_h - exact_end), grad(u_h - exact_end)))

# %% [markdown]
# We define a function to evolve the solution from time 0 to the end time.
# The first argument is a class with a `solve` method that moves the
# solution from one time level to the next - i.e., solves the non-linear
# problem for $u^{n+1}$ given $u^n$:

# %%
def evolve(scheme, u_h, u_h_n, endTime):
    time = 0
    while time < (endTime - 1e-6):
        t.value = time
        u_h_n.assign(u_h)
        scheme.solve(target=u_h)
        time += scheme.model.dt

# %% [markdown]
# We can simply use the `galerkinScheme` instance `scheme` in this function to produce the solution
# at the final time. We combine this with a loop to compute the error over
# two grids and estimate the convergence rate:

# %%
dt.value = 0.005

errors = 0,0
loops = 2
for eocLoop in range(loops):
    u_h.interpolate(initial)
    evolve(scheme, u_h, u_h_n, endTime)
    errors_old = errors
    errors = [sqrt(e) for e in integrate([l2error,h1error])]
    if eocLoop == 0:
        eocs = ['-','-']
    else:
        eocs = [ round(numpy.log(e/e_old)/numpy.log(0.5),2) \
                 for e,e_old in zip(errors,errors_old) ]
    print('Forchheimer: step:', eocLoop, ', size:', gridView.size(0))
    print('\t | u_h - u | =', '{:0.5e}'.format(errors[0]), ', eoc =', eocs[0])
    print('\t | grad(uh - u) | =', '{:0.5e}'.format(errors[1]), ', eoc =', eocs[1])
    u_h.plot()
    if eocLoop < loops-1:
        gridView.hierarchicalGrid.globalRefine(1)
        dt.value /= 2

# %% [markdown]
# .. index::
#    pair: Solvers; Scipy
#
# ## Using Scipy
# We implement a simple Newton Krylov solver using a linear solver from
# Scipy. We can use the `as_numpy` method to access the degrees of freedom as
# Numpy vector based on the `python buffer protocol`. So no data is copied
# and changes to the dofs made on the Python side are automatically carried
# over to the C++ side.

# %% [markdown]
# The most important step is accessing the data structures setup on the C++
# side in Python. In this case we would like to use the underlying dof vector from
# a discrete function as numpy arrays and system matrices assembled by the
# schemes and operators as scipy sparse matrices.
# In the [introduction](dune-fempy_nb.ipynb) we already discussed the `as_numpy` method.
# So

# %%
vecu_h = u_h.as_numpy

# %% [markdown]
# provides access to the underlying dof vector without copying. So changes
# to the numpy array `vecu_h` carries over to the discrete function `u_h`.
# Just remember to make changes using `vecu_h[:]` to change the actual
# memory buffer.
#
# A `scheme` describing an operator `L`
# provides a method `linear` which returns an object that
# stores a sparse matrix structure. The object describes the operator
# linearized around zero.
# To linearize around a different value use the `jacobian` method on the `scheme` that linearized
# the the operator `L` around a given grid function `ubar` and fills the
# linear operator structure passed in as second argument. It is also
# possible to pass ``assemble=False`` to the ``linear`` method to avoid an
# the linearization around zero to reduce computational cost:

# %%
linOp = scheme.linear()                  # linearized around 0
linOp = scheme.linear(assemble=False)    # empty (non valid) linear operator
scheme.jacobian(space.zero, linOp)       # linearized around zero

# %% [markdown]
# Here we linearize around zero. But that argument could be any grid
# function. A second version of this method will return an addition
# discrete function `rhs` which equals `-L[ubar]` such that `DL[ubar](u-ubar) - rhs` are the first
# terms in the Taylor expansion of the operator `L`:

# %%
rhs = u_h.copy()
scheme.jacobian(u_h, linOp, rhs)

# %% [markdown]
# One can now easily access the underlying sparse matrix by again using
# `as_numpy` (and again the underlying data buffers are not copied):

# %%
A = linOp.as_numpy
print(type(A))
# plt.spy(A, precision=1e-8, markersize=1)

# %% [markdown]
# Now we have all the ingredients to write a simple Newton solver to solve
# our non-linear time dependent PDE.

# %%
import numpy as np
from scipy.sparse.linalg import spsolve as solver
class Scheme:
  def __init__(self, scheme):
      self.model = scheme.model
      self.jacobian = scheme.linear()

  def solve(self, target):
      # create a copy of target for the residual
      res = target.copy(name="residual")

      # extract numpy vectors from target and res
      sol_coeff = target.as_numpy
      res_coeff = res.as_numpy

      n = 0
      while True:
          scheme(target, res)
          absF = numpy.sqrt( np.dot(res_coeff,res_coeff) )
          if absF < 1e-10:
              break
          scheme.jacobian(target,self.jacobian)
          sol_coeff -= solver(self.jacobian.as_numpy, res_coeff)
          n += 1

scheme_cls = Scheme(scheme)

u_h.interpolate(initial)                # reset u_h to initial
evolve(scheme_cls, u_h, u_h_n, endTime)
error = u_h - exact_end
print("Forchheimer(numpy) size: ", gridView.size(0), "L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
  *[ sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]))

# %% [markdown]
# We can also use a non linear solver from the Scipy package

# %%
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg as solver

class Scheme2:
    def __init__(self, scheme):
        self.scheme = scheme
        self.model = scheme.model
        self.res = u_h.copy(name="residual")

    # non linear function
    def f(self, x_coeff):
        # the following converts a given numpy array
        # into a discrete function over the given space
        x = space.function("tmp", dofVector=x_coeff)
        scheme(x, self.res)
        return self.res.as_numpy

    # class for the derivative DS of S
    class Df(LinearOperator):
        def __init__(self, x_coeff):
            self.shape = (x_coeff.shape[0], x_coeff.shape[0])
            self.dtype = x_coeff.dtype
            x = space.function("tmp", dofVector=x_coeff)
            self.jacobian = scheme.linear()
            self.update(x_coeff,None)
        # reassemble the matrix DF(u) given a DoF vector for u
        def update(self, x_coeff, f):
            x = space.function("tmp", dofVector=x_coeff)
            scheme.jacobian(x, self.jacobian)
        # compute DS(u)^{-1}x for a given DoF vector x
        def _matvec(self, x_coeff):
            return solver(self.jacobian.as_numpy, x_coeff, tol=1e-10, atol=1e-10)[0]

    def solve(self, target):
        sol_coeff = target.as_numpy
        # call the newton krylov solver from scipy
        sol_coeff[:] = newton_krylov(self.f, sol_coeff,
                    verbose=0, f_tol=1e-8,
                    inner_M=self.Df(sol_coeff))

scheme2_cls = Scheme2(scheme)
u_h.interpolate(initial)
evolve(scheme2_cls, u_h, u_h_n, endTime)
error = u_h - exact_end
print("Forchheimer(scipy) size: ", gridView.size(0), "L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
  *[ sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]))

# %% [markdown]
# .. index::
#    triple: Solvers; Dirichlet; Conditions
#
# .. index::
#    pair: Boundary; Linear Solvers
#
# ## Handling Dirichlet boundary conditions
# We look at a simple Poisson problem with Dirichlet BCs
# to show how to use external solvers like the cg method
# from Scipy in this case.
# We solve $-\triangle u=10\chi_\omega$ where $\chi_\omega$ is a characteristic
# function with $\omega=\{x\colon |x|^2<0.6\}$. For the boundary
# we prescribe trivial Neuman at the top and bottom boundaries
# and Dirichlet values $u=-1$ and $u=1$ at the left and right
# boundaries, respectively.
# We will use the CG solver from `scipy.sparse.linalg`.
#
# .. tip:: Since we are not needing to invert the operator
# we will use the `dune.fem.operator.galerkin` class
# to setup the problem. This is similar to `dune.fem.scheme.galerkin`
# we have been using so far but can be used to model
# operators between different spaces.
# See [here](scheme_api.rst) for a summary of the concepts and API for
# operators and schemes.

# %%
from dune.ufl import DirichletBC
from dune.fem.operator import galerkin
from scipy.sparse.linalg import cg as solver
model = ( inner(grad(u), grad(v)) -
          conditional(dot(x,x)<0.6,10.,0.) * v ) * dx
dbcs  = [ DirichletBC(space,-1,1), DirichletBC(space, 1,2) ]
op  = galerkin([model, *dbcs], space)
sol = space.interpolate(0, name="u_h")
rhs = sol.copy()
lin = op.linear()

# %% [markdown]
# So far everything is as before. Dirichlet boundary conditions
# are handled in the matrix through changing all rows
# associated with boundary degrees of freedom to unit rows -
# associated columns are not changed so the matrix will not be symmetric anymore.
# For solving the system we need to modify the right hand side
# and the initial guess for the iterative solver to include
# the boundary values (to counter the missing symmetry).
# We can use the first of the three versions of the
# `setConstraints` methods on the scheme class discussed in the section on
# [more general boundary conditions](boundary_nb.ipynb#Accessing-the-Dirichlet-degrees-of-freedom).

# %%
op.setConstraints(rhs)
op.setConstraints(sol)
rk = sol.copy("residual")
def cb(xk): # a callback to print the residual norm in each step
    x_h = space.function("iterate", dofVector=xk)
    op(x_h,rk)
    print(rk.as_numpy[:].dot(rk.as_numpy[:]), flush=True, end='\t')
sol.as_numpy[:], _ = solver(lin.as_numpy, rhs.as_numpy, x0=sol.as_numpy,
                            callback=cb, tol=1e-10, atol=1e-10)
sol.plot()

# %% [markdown]
# .. index::
#    pair: Solvers; Petsc
#
# ## Using PETSc and Petsc4Py
# The following requires that a PETSc installation was found during the
# configuration of ``dune``. Furthermore some examples make use of the
# Python package ``petsc4py`.

# %%
from dune.common.checkconfiguration import assertCMakeHave, ConfigurationError
try:
    assertCMakeHave("HAVE_PETSC")
    petsc = True
except ConfigurationError:
    print("Dune not configured with petsc - skipping example")
    petsc = False
try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py module not found -- skipping example")
    petsc4py = None

# %% [markdown]
# Switching to a storage based on the PETSc solver package and solving the
# system using the dune-fem bindings can be achieved by using the
# ``storage`` argument to the space constructor

# %%
if petsc:
    spacePetsc = solutionSpace(gridView, order=2, storage='petsc')
    # first we will use the petsc solver available in the `dune-fem` package
    # (using the sor preconditioner)
    schemePetsc = solutionScheme(a == b, space=spacePetsc,
                    parameters={"linear.preconditioning.method":"sor"})
    dt.value = scheme.model.dt
    u_h = spacePetsc.interpolate(initial, name='u_h')
    u_h_n = u_h.copy(name="previous")
    evolve(schemePetsc, u_h, u_h_n, endTime)
    error = u_h - exact_end
    print("Forchheimer(petsc) size: ", gridView.size(0), "L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
      *[ sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]))

# %% [markdown]
# Implementing a Newton Krylov solver using the binding provided by petsc4py
# %%
if petsc4py is not None and petsc:
    class Scheme3:
      def __init__(self, scheme):
          self.model = scheme.model
          self.jacobian = scheme.linear()
          self.ksp = PETSc.KSP()
          self.ksp.create(PETSc.COMM_WORLD)
          # use conjugate gradients method
          self.ksp.setType("cg")
          # and incomplete Cholesky
          self.ksp.getPC().setType("icc")
          self.ksp.setOperators(self.jacobian.as_petsc)
          self.ksp.setFromOptions()
      def solve(self, target):
          res = target.copy(name="residual")
          sol_coeff = target.as_petsc
          res_coeff = res.as_petsc
          n = 0
          while True:
              schemePetsc(target, res)
              absF = numpy.sqrt( res_coeff.dot(res_coeff) )
              if absF < 1e-10:
                  break
              schemePetsc.jacobian(target, self.jacobian)
              self.ksp.solve(res_coeff, res_coeff)
              sol_coeff -= res_coeff
              n += 1

    u_h.interpolate(initial)
    scheme3_cls = Scheme3(schemePetsc)
    evolve(scheme3_cls, u_h, u_h_n, endTime)
    error = u_h - exact_end
    print("Forchheimer(petsc) size: ", gridView.size(0), "L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
      *[ sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]))

# %% [markdown]
# Using the petsc4py bindings for the non linear KSP solvers from PETSc
# %%
if petsc4py is not None and petsc is not None:
    class Scheme4:
        def __init__(self, scheme):
            self.model = scheme.model
            self.res = scheme.space.interpolate([0],name="residual")
            self.scheme = scheme
            self.jacobian = scheme.linear()
            self.snes = PETSc.SNES().create()
            self.snes.setFunction(self.f, self.res.as_petsc.duplicate())
            self.snes.setUseMF(False)
            self.snes.setJacobian(self.Df, self.jacobian.as_petsc, self.jacobian.as_petsc)
            self.snes.getKSP().setType("cg")
            self.snes.setFromOptions()

        def f(self, snes, x, f):
            # setup discrete function using the provide petsc vectors
            inDF = self.scheme.space.function("tmp",dofVector=x)
            outDF = self.scheme.space.function("tmp",dofVector=f)
            self.scheme(inDF,outDF)

        def Df(self, snes, x, m, b):
            inDF = self.scheme.space.function("tmp",dofVector=x)
            self.scheme.jacobian(inDF, self.jacobian)
            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

        def solve(self, target):
            sol_coeff = target.as_petsc
            self.res.clear()
            self.snes.solve(self.res.as_petsc, sol_coeff)

    u_h.interpolate(initial)
    scheme4_cls = Scheme4(schemePetsc)
    evolve(scheme4_cls, u_h, u_h_n, endTime)
    error = u_h - exact_end
    print("Forchheimer(petsc4py) size: ", gridView.size(0), "L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
      *[ sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]))

# %% [markdown]
# .. index::
#    triple: I/O; Logging; Parameters

# ## Accessing and reusing values of parameters
# Sometimes it is necessary to extract which parameters were read and which
# values were used, e.g., for debugging purposes like finding spelling
# in the parameters provided to a scheme.
# Note that this information can only be reliably obtained after usage of
# the scheme, e.g., after calling solve as shown in the example below.
# To add logging to a set of parameters passed to a `scheme` one simply
# needs to add a `logging` key to the parameter dictionary provided to the scheme
# with a tag (string) that is used in the output.
#
# As an example we will solve the simple Laplace equation from the
# introduction but pass some preconditioning parameters to the scheme.

# %%
import dune.fem
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from ufl import (TestFunction, TrialFunction, SpatialCoordinate,
                 dx, grad, inner, dot, sin, cos, pi )
gridView = structuredGrid([0, 0], [1, 1], [200, 200])
space = lagrange(gridView, order=1, storage="istl")
u_h   = space.interpolate(0, name='u_h')
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

f = (8*pi**2+1) * cos(2*pi*x[0])*cos(2*pi*x[1])
a = ( inner(grad(u),grad(v)) + u*v ) * dx
l = f*v * dx
scheme = galerkin( a==l, solver="cg", parameters=
                   {"newton.linear.tolerance": 1e-12,
                    "newton.linear.verbose": True,
                    "newton.linear.preconditioning.method": "amg-ilu",
                    "fem.solver.newton.linear.errormeasure": "relative",
                    "logging": "precon-amg"
                   } )
info = scheme.solve(target=u_h)

# %% [markdown]
# We use the `pprint` (pretty print) module if available to get nicer
# output.

# %%
try:
    from pprint import pprint as _pprint
    pprint = lambda *args,**kwargs: _pprint(*args,**kwargs,width=200,compact=False)
except ImportError:
    pprint = print

pprint(dune.fem.parameter.log())

# %% [markdown]
# Note above that all parameters are printed including some default ones
# used in other parts of the code. If multiple schemes with different
# `logging` parameter strings are used, all would be shown using the `log`
# method as shown above.
# To access only the parameters used in the scheme simply use
# either `dune.fem.parameter.log()["tag"])` or access the parameter log
# through the scheme:

# %%
pprint(scheme.parameterLog())

# %% [markdown]
# One can easily reuse these parameters to construct another scheme by
# converting the result of the above call to a dictionary.
# As an example change the above problem to a PDE with Dirichlet conditions
# but turn of verbose output of the solver.
#
# .. note:: the `logging` parameter has to be set if we want to use the
# `parameterLog` method on the scheme.

# %%
param = dict(scheme.parameterLog()) # this method returns a set of pairs which we can convert to a dictionary
param["logging"] = "Dirichlet" # only needed to use the `parameterLog` method
param["newton.linear.verbose"] = False
scheme2 = galerkin( [a==l,DirichletBC(space,0)], parameters=param )
u_h.clear()
info = scheme2.solve(target=u_h)
pprint(scheme2.parameterLog())

# %% [markdown]
# ### Parameter hints
#
# .. tip:: To get information about available values for some parameters
# (those with string arguments) a possible approach is to provide a non valid
# string, e.g., `"help"`.

# %%
scheme = galerkin( a==l, solver="cg", parameters=
                   {"newton.linear.tolerance": 1e-12,
                    "newton.linear.verbose": True,
                    "newton.linear.preconditioning.method": "help",
                    "fem.solver.newton.linear.errormeasure": "relative",
                    "logging": "precon-amg"
                   } )
try:
    scheme.solve(target=u_h)
except RuntimeError as rte:
    print(rte)

# %% [markdown]
# .. index::
#    triple: Solvers; Available solvers; Parameters
#
# ## Available solvers and parameters
# Upon creation of a discrete function space one also have to specifies the
# storage which is tied to the solver backend.
# As mentioned, different linear algebra backends can be accessed by changing setting the
# `storage` parameter during construction of the discrete space. All
# discrete functions and operators/schemes based on this space will then
# use this backend. Available backends are `numpy,istl,petsc`.
# Note that not all methods which are available in `dune-istl` or `PETSc` have been forwarded
# to be used with `dune-fem`.

# %%
space = solutionSpace(gridView, order=2, storage='numpy')

# %% [markdown]
# Switching is as simple as passing `storage='istl'` or  `storage='petsc'`.
# Here is a summary of the available backends
#
# | Solver    | Description |
# | --------- | ------------------------------------------------------------------- |
# | numpy     | the storage is based on a raw C pointer which can be                |
# |           | directly accessed as a numpy.array using the Python buffer protocol |
# |           | To change the underlying vector of a discrete function 'u_h' use    |
# |           | 'uh.as_numpy[:]'.                                                   |
# |           | As shown in the examples, linear operators return a                 |
# |           | `scipy.sparse.csr_matrix` through the 'as_numpy' property.          |
# | istl      | data is stored in a block vector/matrix from the dune.istl package  |
# |           | Access through 'as_istl'                                            |
# | petsc     | data is stored in a petsc vector/matrix which can also be used with |
# |           | petsc4py on the python side using 'as_petsc'                        |

# %% [markdown]
# .. index:: Solvers; Parameters

# When creating a scheme, there is the possibility to select a linear
# solver for the internal Newton method.
# In addition the behavior of the solver can be customized through a
# parameter dictionary. This allows to set tolerances, verbosity, but also
# which preconditioner to use.
#
# For details see the help available for a scheme:

# %%
help(scheme)
