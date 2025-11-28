#include <dune/fem/gridpart/common/entitysearch.hh>
#include <dune/fem/function/common/localcontribution.hh>
#include <dune/fem/function/localfunction/const.hh>
#include <dune/fem/common/bindguard.hh>

// Note: the coefficients of the functional will be ADDED to the storage of the functional
template <class Point, class Functional, class Error>
double pointFunctional(const Point &point, Functional &functional, Error &error)
{
  // first find the entity containing `point`
  typedef typename Functional::DiscreteFunctionSpaceType::GridPartType GridPartType;
  Dune::Fem::EntitySearch<GridPartType> search(functional.space().gridPart());
  const auto &entity = search(point);
  const auto localPoint = entity.geometry().local(point);

  // add the contributions from the basis functions to `functional`
  Dune::Fem::AddLocalContribution< Functional > wLocal( functional );
  {
    auto guard = Dune::Fem::bindGuard( wLocal, entity );
    typename Functional::RangeType one( 1 );
    wLocal.axpy(localPoint, one);
  }

  // in addition we also want to compute the value of `error` at `point`
  auto localError = constLocalFunction(error);
  localError.bind(entity);
  return localError.evaluate(localPoint)[0];
}
