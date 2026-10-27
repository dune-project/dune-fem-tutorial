import time, io
import numpy
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from ufl import *
import dune.ufl
from dune.generator import algorithm
import dune.geometry as geometry
import dune.fem as fem
from dune.fem.view import geometryGridView as geoGridView
from dune.fem.space import lagrange as solutionSpace
from dune.fem.scheme import galerkin as solutionScheme
from dune.alugrid import aluConformGrid as leafGridView

def plot(ct, ct2):
    fig, ax = plt.subplots()
    plt.loglog(ct[0][:], ct[1][:],'*-', markersize=15, label='hybrid')
    plt.loglog(ct2[0][:], ct2[1][:],'*-', markersize=15, label='python')
    plt.grid(True)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
    yticks = ax.yaxis.get_minor_ticks()
    for t in yticks:
        t.label1.set_visible(False)
    xticks = ax.xaxis.get_minor_ticks()
    for t in xticks:
        t.label1.set_visible(False)
    plt.yticks(numpy.append(ct[1], ct2[1]))
    plt.xticks(ct[0])
    plt.legend(loc="upper left")
    plt.xlabel('Number of Grid Elements (log)',fontsize=20)
    plt.gcf().subplots_adjust(bottom=0.17, left=0.16)
    plt.ylabel('Runtime in s (log)',fontsize=20)
order = 2
R0 = 2.
endTime = 0.1
def calcRadius(surface):
    R,vol = 0, 0
    for e in surface.elements:
        rule = geometry.quadratureRule(e.type, 4)
        for p in rule:
            geo = e.geometry
            weight = geo.volume * p.weight
            R   += geo.toGlobal(p.position).two_norm * weight
            vol += weight
    return R/vol
code = """
template< class Surface >
double calcRadius( const Surface &surface ) {
  double R = 0, vol = 0.;
  for( const auto &entity : elements( surface ) ) {
    const auto& rule = Dune::QuadratureRules<double, 2>::rule(entity.type(), 4);
    for ( const auto &p : rule ) {
      const auto geo = entity.geometry();
      const double weight = geo.volume() * p.weight();
      R   += geo.global(p.position()).two_norm() * weight;
      vol += weight;
    }
  }
  return R/vol;
}
"""
switchCalcRadius = lambda use_cpp,surface: \
             algorithm.load('calcRadius', io.StringIO(code), surface) \
             if use_cpp else calcRadius
def calculate(use_cpp, gridView):
    space = solutionSpace(gridView, dimRange=gridView.dimWorld, order=order)
    u = TrialFunction(space)
    v = TestFunction(space)
    x = SpatialCoordinate(space)
    positions = space.interpolate(x, name="position")
    surface = geoGridView(positions)
    space = solutionSpace(surface, dimRange=surface.dimWorld, order=order)
    solution  = space.interpolate(x, name="solution")
    theta = 0.5   # Crank-Nicolson
    I = Identity(3)
    dt = dune.ufl.Constant(0,"dt")
    a = (inner(u - x, v) + dt * inner(theta*grad(u)
        + (1 - theta)*I, grad(v))) * dx
    scheme = solutionScheme(a == 0, space, solver="cg")
    Rexact = lambda t: numpy.sqrt(R0*R0 - 4.*t)
    radius = switchCalcRadius(use_cpp,surface)
    dt.value = 0.02
    numberOfLoops = 3
    times = numpy.zeros(numberOfLoops)
    errors = numpy.zeros(numberOfLoops)
    totalIterations = numpy.zeros(numberOfLoops, numpy.dtype(numpy.uint32))
    gridSizes = numpy.zeros(numberOfLoops, numpy.dtype(numpy.uint32))
    for i in range(numberOfLoops):
        positions.interpolate(x * (R0/sqrt(dot(x,x))))
        solution.interpolate(x)
        t = 0.
        error = abs(radius(surface)-Rexact(t))
        iterations = 0
        start = time.time()
        while t < endTime:
            info = scheme.solve(target=solution)
            positions.assign(solution)
            iterations += int( info["linear_iterations"] )
            t          += scheme.model.dt
            error       = max(error, abs(radius(surface)-Rexact(t)))
        print("time used:", time.time() - start)
        times[i] = time.time() - start
        errors[i] = error
        totalIterations[i] = iterations
        gridSizes[i] = gridView.size(2)
        if i < numberOfLoops - 1:
            gridView.hierarchicalGrid.globalRefine(1)
            dt.value /= 2
    eocs = numpy.log(errors[0:][:numberOfLoops-1] / errors[1:]) / numpy.log(numpy.sqrt(2))
    try:
        import pandas as pd
        keys = {'size': gridSizes, 'error': errors, "eoc": numpy.insert(eocs, 0, None), 'iterations': totalIterations}
        table = pd.DataFrame(keys, index=range(numberOfLoops),columns=['size', 'error', 'eoc', 'iterations'])
        print(table)
    except ImportError:
        print("pandas module not found so not showing table - ignored")
        pass
    return gridSizes, times
results = []
gridView = leafGridView("sphere.dgf", dimgrid=2, dimworld=3)
results += [calculate(True, gridView)]
gridView = leafGridView("sphere.dgf", dimgrid=2, dimworld=3)
results += [calculate(False, gridView)]
plot(results[0],results[1])