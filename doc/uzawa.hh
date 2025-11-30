#include <dune/common/fvector.hh>
#include <iostream>

template< class MainOp, class GOp, class AInv, class DOp, class MInv, class PInv,
          class Velocity, class Pressure >
double uzawa(double nu, double mu,
             const MainOp &mainOp, const GOp &G, const DOp &D,
             const AInv &Ainv, const MInv &Minv, const PInv &Pinv, Velocity &rhsVelo,
             Velocity &xi, Pressure &rhsPress,
             Pressure &r, Pressure &d, Pressure &precon,
             Velocity &velocity, Pressure &pressure)
{
  const int dimension = 2; // TODO
  Ainv(rhsVelo, velocity);
  D(velocity,rhsPress);
  Minv(rhsPress, r);
  if (nu > 0) {
    precon.clear();
    Pinv(rhsPress, precon);
    r *= mu;
    r.axpy(nu,precon); }
  d.assign(r);
  auto delta = r.scalarProductDofs(rhsPress);
  for (int m=0; m<100; ++m) {
    xi.clear();
    G(d,rhsVelo);
    mainOp.setConstraints(
       Dune::FieldVector<double,dimension>(0),rhsVelo);
    Ainv(rhsVelo, xi);
    D(xi,rhsPress);
    auto rho = delta /
       d.scalarProductDofs(rhsPress);
    pressure.axpy(rho,d);
    velocity.axpy(-rho,xi);
    D(velocity, rhsPress);
    Minv(rhsPress,r);
    if (nu > 0) {
      precon.clear();
      Pinv(rhsPress,precon);
      r *= mu;
      r.axpy(nu,precon); }
    auto oldDelta = delta;
    delta = r.scalarProductDofs(rhsPress);
    if (delta < 1e-9) break;
    auto gamma = delta/oldDelta;
    d *= gamma;
    d += r;
  }
}
