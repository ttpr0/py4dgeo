#include <py4dgeo/compute.hpp>

#include <py4dgeo/openmp.hpp>
#include <py4dgeo/py4dgeo.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace py4dgeo {

Epoch
pointcloud_stacking(
  const Epoch& epoch,
  double radius,
  double max_distance)
{
  EigenPointCloud cloud(epoch.cloud.rows(), 3);
  EigenCovarianceSet covs(epoch.cloud.rows(), 9);

  EigenNormalSet orientation = {0.0, 0.0, 1.0};
  EigenNormalSet normals(epoch.cloud.rows(), 3);
  std::vector<double> used_radii;
  py4dgeo::compute_multiscale_directions(epoch, epoch.cloud, {radius}, orientation, normals, used_radii);

  CallbackExceptionVault vault;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (IndexType i = 0; i < epoch.cloud.rows(); ++i) {
    vault.run([&]() {
      Eigen::RowVector3d normal = normals.row(i);

      WorkingSetFinderParameters params{
        epoch, radius, epoch.cloud.row(i), normal, max_distance
      };
      std::vector<IndexType> subset = py4dgeo::cylinder_workingset_finder(params);

      // adjust position to the weighted mean of the subset along the normal direction
      double mean = 0.0;
      double covariance_sum = 0.0;
      for (std::size_t j = 0; j < subset.size(); ++j) {
        Eigen::Matrix3d C = to_covariance_matrix(epoch.covariances.value(), subset[j]);
        Eigen::RowVector3d P = epoch.cloud.row(subset[j]);
        double c = normal * C * normal.transpose();
        mean += static_cast<double>(normal * P.transpose()) / c;
        covariance_sum += 1.0 / c;
      }
      mean /= covariance_sum;
      double diff = mean - static_cast<double>(normal * epoch.cloud.row(i).transpose());
      cloud.row(i) = epoch.cloud.row(i) + diff * normal;
      covs.row(i) = epoch.covariances.value().row(i);
    });
  }

  // Potentially rethrow an exception that occurred in above parallel region
  vault.rethrow();

  return py4dgeo::Epoch(cloud, covs);
}

} // namespace py4dgeo
