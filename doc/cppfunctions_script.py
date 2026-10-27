import numpy
from matplotlib import pyplot, ticker
from dune.fem.plotting import plotComponents
from dune.grid import structuredGrid as leafGridView
from dune.fem.function import gridFunction
from ufl import SpatialCoordinate
import dune.ufl
from dune.fem import integrate
from dune.common import FieldVector
from dune.fem.space import lagrange as solutionSpace
from ufl import TestFunction, TrialFunction, SpatialCoordinate,\
                dx, ds, grad, inner, sin, pi
from dune.fem.scheme import galerkin as solutionScheme
from dune.ufl import DirichletBC, Constant

gridView = leafGridView([0, 0], [1, 1], [4, 4])
code="""
template <class GridView>
auto aTimesExact(double a) {
  return [a](const auto& en,const auto& xLocal) -> auto {
    auto x = en.geometry().global(xLocal);
    return a*(1./2.*(std::pow(x[0],2)+std::pow(x[1],2)) - 1./3.*(std::pow(x[0],3) - std::pow(x[1],3)) + 1.);
  };
}
"""
exactCpp = gridFunction(code, gridView, name="aTimesExact", order=2, args=[2.])
exactCpp.plot()
x = SpatialCoordinate(dune.ufl.domain(2))
exact = (x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1
exact_gf = gridFunction(exact, gridView, name="ufl", order=1)
print( integrate(abs(exact-exactCpp)) )
code2="""
template <class GridView, class GF>
auto aTimesExact(const GF &gf,Dune::FieldVector<double,1> &a) {
  return [lgf=localFunction(gf),&a] (const auto& en,const auto& xLocal) mutable -> auto {
    lgf.bind(en); // lambda must be mutable so that the non const function can be called
    return a[0]*lgf(xLocal);
  };
}
"""
a = FieldVector([2])
exactCpp2 = gridFunction(code2, gridView, name="aTimesExact", order=2, args=[exact_gf,a])
print( integrate(abs(exactCpp-exactCpp2)) )
a[0] = -5.
exactCpp2.plot()
print( integrate(abs(exactCpp-exactCpp2)) )
vecSpace = solutionSpace(gridView, dimRange=2, order=2)
vec = vecSpace.interpolate([0,0], name='u_h')
uVec,vVec = TrialFunction(vecSpace), TestFunction(vecSpace)
a  = inner(grad(uVec), grad(vVec)) * dx
a += ( ( 10*(1-2*uVec[1])+uVec[0] ) * vVec[0] +\
       ( 10*sin(2*pi*x[1]) + (1-2*uVec[0])**2*uVec[1] ) * vVec[1]
     ) * dx
code="""
template <class GridView>
auto bnd(pybind11::array_t<double> &a, double h) {
  return [ptr=static_cast<double*>(a.request(true).ptr),h](const auto& en,const auto& xLocal) -> auto {
    auto x = en.geometry().global(xLocal);
    return sin(h*M_PI*(ptr[0]*x[0]+ptr[1]*x[1]));
  };
}
"""
coeffVec = numpy.array([5.,-2.])
bndCpp = gridFunction(code, gridView=gridView, name="bnd", order=2, args=[coeffVec,2.])
bcBottom = DirichletBC(vecSpace,[sin(2*pi*(x[0]+x[1])),1],x[1]<1e-10)
bc = DirichletBC(vecSpace,[bndCpp,None])
vecScheme = solutionScheme( [a == 0, bc, bcBottom],
                            parameters={"linear.tolerance": 1e-9} )
vecScheme.solve(target=vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})
coeffVec[:] = numpy.array([1.,1.])
vecScheme.solve(target=vec)
plotComponents(vec, gridLines=None, level=2,
               colorbar={"orientation":"horizontal", "ticks":ticker.MaxNLocator(nbins=4)})