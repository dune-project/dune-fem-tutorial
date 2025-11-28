#include <dune/geometry/quadraturerules.hh>
template< class Surface >
double calcRadius( const Surface &surface )
{
  double R = 0.;
  double vol = 0.;
  for( const auto &entity : elements( surface ) )
  {
    const auto& rule = Dune::QuadratureRules<double, 2>::rule(entity.type(), 4);
    for ( const auto &p : rule )
    {
      const auto geo = entity.geometry();
      const double weight = geo.volume() * p.weight();
      R   += geo.global(p.position()).two_norm() * weight;
      vol += weight;
    }
  }
  return R/vol;
}
