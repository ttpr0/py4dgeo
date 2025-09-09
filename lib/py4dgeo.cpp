#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Core>

namespace py4dgeo {
Eigen::Matrix3d
to_covariance_matrix(const EigenCovarianceSetConstRef cov,
                     const IndexType index)
{
  const Eigen::Vector<double, 9> cov_vector = cov.row(index);
  return cov_vector.reshaped<Eigen::RowMajor>(3, 3);
}
}
