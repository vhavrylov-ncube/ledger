#include "math/linalg/blas/base.hpp"
#include "math/linalg/blas/gemv_t.hpp"
#include "math/linalg/matrix.hpp"
#include "math/linalg/prototype.hpp"
namespace fetch {
namespace math {
namespace linalg {

template <typename S, uint64_t V>
void Blas<S, Signature(_y <= _alpha, _A, _x, _n, _beta, _y, _m),
          Computes(_y = _alpha * T(_A) * _x + _beta * _y), V>::
     operator()(type const &alpha, Matrix<type> const &a, ShapeLessArray<type> const &x, int const &incx,
           type const &beta, ShapeLessArray<type> &y, int const &incy) const
{
  int  jy;
  type temp;
  int  kx;
  int  j;
  int  i;
  int  ky;
  int  leny;
  int  lenx;
  if ((int(a.height()) == 0) || ((int(a.width()) == 0) || ((alpha == 0.0) && (beta == 1.0))))
  {
    return;
  }

  lenx = int(a.height());
  leny = int(a.width());
  if (incx > 0)
  {
    kx = 1;
  }
  else
  {
    kx = 1 + (-(-1 + lenx) * incx);
  }

  if (incy > 0)
  {
    ky = 1;
  }
  else
  {
    ky = 1 + (-(-1 + leny) * incy);
  }

  if (beta != 1.0)
  {
    if (incy == 1)
    {
      if (beta == 0.0)
      {
        for (i = 0; i < leny; ++i)
        {
          y[i] = 0.0;
        }
      }
      else
      {
        for (i = 0; i < leny; ++i)
        {
          y[i] = beta * y[i];
        }
      }
    }
    else
    {
      int iy;
      iy = -1 + ky;
      if (beta == 0.0)
      {
        for (i = 0; i < leny; ++i)
        {
          y[iy] = 0.0;
          iy    = iy + incy;
        }
      }
      else
      {
        for (i = 0; i < leny; ++i)
        {
          y[iy] = beta * y[iy];
          iy    = iy + incy;
        }
      }
    }
  }

  if (alpha == 0.0)
  {
    return;
  }

  jy = -1 + ky;
  if (incx == 1)
  {
    for (j = 0; j < int(a.width()); ++j)
    {
      temp = 0.0;
      for (i = 0; i < int(a.height()); ++i)
      {
        temp = temp + a(i, j) * x[i];
      }

      y[jy] = y[jy] + alpha * temp;
      jy    = jy + incy;
    }
  }
  else
  {
    for (j = 0; j < int(a.width()); ++j)
    {
      int ix;
      temp = 0.0;
      ix   = -1 + kx;
      for (i = 0; i < int(a.height()); ++i)
      {
        temp = temp + a(i, j) * x[ix];
        ix   = ix + incx;
      }

      y[jy] = y[jy] + alpha * temp;
      jy    = jy + incy;
    }
  }

  return;
};

template class Blas<double, Signature(_y <= _alpha, _A, _x, _n, _beta, _y, _m),
                    Computes(_y = _alpha * T(_A) * _x + _beta * _y),
                    platform::Parallelisation::NOT_PARALLEL>;
template class Blas<float, Signature(_y <= _alpha, _A, _x, _n, _beta, _y, _m),
                    Computes(_y = _alpha * T(_A) * _x + _beta * _y),
                    platform::Parallelisation::NOT_PARALLEL>;
template class Blas<double, Signature(_y <= _alpha, _A, _x, _n, _beta, _y, _m),
                    Computes(_y = _alpha * T(_A) * _x + _beta * _y),
                    platform::Parallelisation::THREADING>;
template class Blas<float, Signature(_y <= _alpha, _A, _x, _n, _beta, _y, _m),
                    Computes(_y = _alpha * T(_A) * _x + _beta * _y),
                    platform::Parallelisation::THREADING>;

}  // namespace linalg
}  // namespace math
}  // namespace fetch