#pragma once

#include <py4dgeo/epoch.hpp>
#include <py4dgeo/py4dgeo.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace py4dgeo {

/** @brief The type to use for the spatiotemporal distance array. */
using EigenSpatiotemporalArray =
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenSpatiotemporalArrayRef = Eigen::Ref<EigenSpatiotemporalArray>;
using EigenSpatiotemporalArrayConstRef =
  Eigen::Ref<const EigenSpatiotemporalArray>;

/** @brief The type to use for a TimeSeries */
using EigenTimeSeries = Eigen::Vector<double, Eigen::Dynamic>;
using EigenTimeSeriesRef = Eigen::Ref<EigenTimeSeries>;
using EigenTimeSeriesConstRef = Eigen::Ref<const EigenTimeSeries>;

/** @brief The data object passed to time series distance functions */
struct TimeseriesDistanceFunctionData
{
  EigenTimeSeriesConstRef ts1;
  EigenTimeSeriesConstRef ts2;
  double norm1;
  double norm2;
};

/** @brief The signature to use for a distance function */
using TimeseriesDistanceFunction =
  std::function<double(const TimeseriesDistanceFunctionData&)>;

/** @brief Basic data structure for 4D change object */
struct ObjectByChange
{
  std::unordered_map<IndexType, double> indices_distances;
  IndexType start_epoch;
  IndexType end_epoch;
  double threshold;
};

struct RegionGrowingSeed
{
  IndexType index;
  IndexType start_epoch;
  IndexType end_epoch;
};

struct RegionGrowingAlgorithmData
{
  EigenSpatiotemporalArrayConstRef distances;
  const Epoch& corepoints;
  double radius;
  RegionGrowingSeed seed;
  std::vector<double> thresholds;
  std::size_t min_segments;
  std::size_t max_segments;
};

/** @brief The main region growing algorithm */
ObjectByChange
region_growing(const RegionGrowingAlgorithmData&,
               const TimeseriesDistanceFunction&);

struct FullRegionGrowingAlgorithmData
{
  EigenSpatiotemporalArrayConstRef distances;
  const Epoch& corepoints;
  double radius;
  std::vector<RegionGrowingSeed> seeds;
  std::vector<double> thresholds;
  std::size_t min_segments;
  std::size_t max_segments;
};

/** @brief The main region growing algorithm */
std::vector<std::tuple<int, ObjectByChange>>
full_region_growing(const FullRegionGrowingAlgorithmData&,
                    const TimeseriesDistanceFunction&,
                    int resume_from_seed,
                    int stop_at_seed);

/** @brief The DTW distance measure implementation used in 4DOBC */
double
dtw_distance(const TimeseriesDistanceFunctionData&);

/** @brief Normalized DTW distance measure for 4DOBC */
double
normalized_dtw_distance(const TimeseriesDistanceFunctionData&);

struct ChangePointDetectionData
{
  EigenTimeSeriesConstRef ts;
  IndexType window_width;
  IndexType min_size;
  IndexType jump;
  double penalty;
};

/** @brief Calculate the median of double vector. The function changed the
 * array!*/
double
median_calculation(std::vector<double>&);
double
median_calculation_simp(std::vector<double>&);
/** @brief Calculate the local maxima, which more than "order" values left and
 * right */
std::vector<IndexType>
local_maxima_calculation(std::vector<double>&, IndexType);
/** @brief Calculate cost error */
double cost_L1_error(EigenTimeSeriesConstRef, IndexType, IndexType, IndexType);

/** @brief Calculate signal sum of costs */
double
sum_of_costs(EigenTimeSeriesConstRef, std::vector<IndexType>&, IndexType);

/** @brief Change point detection using sliding window approach, run fit then
 * predict function */
std::vector<IndexType>
change_point_detection(const ChangePointDetectionData&);

/** @brief Compute parameters for change point detection function, return scores
 * array. */
std::vector<double> fit_change_point_detection(EigenTimeSeriesConstRef,
                                               IndexType,
                                               IndexType,
                                               IndexType);

/** @brief Predict change point detection, return the optimal breackpoints, must
 * called after fit function */
std::vector<IndexType>
predict_change_point_detection(EigenTimeSeriesConstRef,
                               std::vector<double>&,
                               IndexType,
                               IndexType,
                               IndexType,
                               double);

struct SeedCandidate
{
  IndexType start_epoch;
  IndexType end_epoch;
};

std::vector<SeedCandidate>
seed_candidate_detection(const EigenTimeSeriesConstRef times,
                         const EigenTimeSeriesConstRef distances,
                         double epsilon,
                         double min_change_magnitude,
                         double max_slope_difference,
                         std::size_t min_period);

std::vector<IndexType>
ramer_douglas_peucker(const EigenTimeSeriesConstRef times,
                      const EigenTimeSeriesConstRef distances,
                      double epsilon);

std::vector<std::vector<int>>
obc_fusion(std::vector<ObjectByChange>& objects,
           double spatial_iou_threshold,
           double temporal_iou_threshold);

/** @brief Applies temporal filtering on the time series of distances */
std::tuple<EigenSpatiotemporalArray, EigenSpatiotemporalArray>
weighted_average_filtering(EigenSpatiotemporalArrayConstRef,
                           EigenSpatiotemporalArrayConstRef,
                           int,
                           bool);

/** @brief Applies robust temporal filtering on the time series of distances */
std::tuple<EigenSpatiotemporalArray, EigenSpatiotemporalArray>
robust_weighted_average_filtering(EigenSpatiotemporalArrayConstRef,
                                  EigenSpatiotemporalArrayConstRef,
                                  int,
                                  bool);

} // namespace py4dgeo
