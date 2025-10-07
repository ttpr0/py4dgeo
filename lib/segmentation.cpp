#include <py4dgeo/segmentation.hpp>

#include <py4dgeo/openmp.hpp>
#include <py4dgeo/searchtree.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace py4dgeo {

void
fixed_threshold_region_growing(
  const RegionGrowingAlgorithmData& data,
  const TimeseriesDistanceFunction& distance_function,
  double threshold,
  ObjectByChange& obj)
{
  // Instantiate a set of candidates to check
  std::unordered_set<IndexType> rejected;
  std::multimap<double, IndexType> candidates_distances;
  std::set<double, std::greater<double>> used_distances;

  // We add the initial seed to the candidates in order to kick off
  // the calculation in the loop below
  candidates_distances.insert({ 0.0, data.seed.index });

  // Create the return object
  obj.start_epoch = data.seed.start_epoch;
  obj.end_epoch = data.seed.end_epoch;
  obj.threshold = threshold;

  // The seed is included in the final object for sure
  obj.indices_distances[data.seed.index] = 0.0;
  used_distances.insert(0.0);

  double residual = threshold;

  // Grow while we have candidates
  while (!candidates_distances.empty()) {
    // Get one element to process and remove it from the candidates
    auto [distance, candidate] = *candidates_distances.begin();
    candidates_distances.erase(candidates_distances.begin());

    // Add neighboring corepoints to list of candidates
    RadiusSearchResult neighbors;
    auto radius_search =
      get_radius_search_function(data.corepoints, data.radius);
    radius_search(data.corepoints.cloud.row(candidate), neighbors);
    for (auto n : neighbors) {
      // Check whether the corepoint is already present among candidates,
      // the final result or the points added at this threshold level.
      // If none of that match, this is a new candidate
      if ((rejected.find(n) == rejected.end()) &&
          (obj.indices_distances.find(n) == obj.indices_distances.end())) {
        // Calculate the distance for this neighbor
        TimeseriesDistanceFunctionData distance_data{
          data.distances(
            data.seed.index,
            Eigen::seq(data.seed.start_epoch, data.seed.end_epoch)),
          data.distances(
            n, Eigen::seq(data.seed.start_epoch, data.seed.end_epoch)),
          data.distances(data.seed.index, 0),
          data.distances(n, 0)
        };
        auto d = distance_function(distance_data);

        // If it is smaller than the threshold, add it to the object (or
        // rather: maybe do so if adaptive thresholding wants you to)
        if (d < threshold) {
          obj.indices_distances[n] = d;
          used_distances.insert(d);
        } else {
          rejected.insert(n);
        }

        // Decide whether this neighbor should also be used as a candidate
        // for further neighbor selection. We do not do this with *all*
        // neighbors added to the grown region, but only with those under the
        // 95th percentile.
        if (d < residual) {
          candidates_distances.insert({ d, n });
        }

        // Update the residual parameter for above criterion
        if (obj.indices_distances.size() >= data.min_segments) {
          auto it = used_distances.begin();
          std::advance(it,
                       static_cast<std::size_t>(0.05 * used_distances.size()));
          residual = *it;
        }
      }
    }
  }
}

ObjectByChange
region_growing(const RegionGrowingAlgorithmData& data,
               const TimeseriesDistanceFunction& distance_function)
{
  // Ensure that we have a sorted list of thresholds
  std::vector<double> sorted_thresholds(data.thresholds);
  std::sort(sorted_thresholds.begin(), sorted_thresholds.end());

  // Run for all threshold levels
  std::vector<ObjectByChange> objects(sorted_thresholds.size());

  // Calculate region growing for all threshold levels. When finished,
  // decide which one to pick.
  CallbackExceptionVault vault;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < sorted_thresholds.size(); ++i)
    vault.run([&]() {
      fixed_threshold_region_growing(
        data, distance_function, sorted_thresholds[i], objects[i]);
    });

  // Potentially rethrow an exception that occurred in above parallel region
  vault.rethrow();

  // Decide which threshold to use
  double last_ratio = 0.5;
  for (std::size_t i = 0; i < objects.size() - 1; ++i) {
    // Apply the maximum threshold
    if (objects[i].indices_distances.size() >= data.max_segments)
      return objects[i];

    double new_ratio =
      static_cast<double>(objects[i].indices_distances.size()) /
      static_cast<double>(objects[i + 1].indices_distances.size());
    if (new_ratio <= last_ratio) {
      // Apply minimum segment threshold
      if (objects[i].indices_distances.size() < data.min_segments)
        return ObjectByChange();

      return objects[i];
    } else {
      last_ratio = new_ratio;
    }
  }

  return objects[objects.size() - 1];
}

std::vector<std::tuple<int, ObjectByChange>>
full_region_growing(const FullRegionGrowingAlgorithmData& data,
                    const TimeseriesDistanceFunction& distance_function,
                    int resume_from_seed,
                    int stop_at_seed)
{
  std::vector<std::tuple<int, ObjectByChange>> objects;
  for (int i = 0; i < data.seeds.size(); ++i) {
    std::cout << "Processing seed " << (i + 1) << " of " << data.seeds.size()
              << "\n";
    auto& seed = data.seeds[i];
    if (i < (resume_from_seed -
             1)) { // resume from index 0 when `resume_from_seed` == 1
      continue;
    }
    if (i >= (stop_at_seed - 1)) { // stop at index 0 when `stop_at_seed` == 1
      break;
    }
    bool found = false;
    for (auto& [_, obj] : objects) {
      if (obj.indices_distances.find(seed.index) !=
            obj.indices_distances.end() &&
          ((obj.end_epoch > seed.start_epoch) &&
           (seed.end_epoch > obj.start_epoch))) {
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    RegionGrowingAlgorithmData rg_data{ data.distances,   data.corepoints,
                                        data.radius,      seed,
                                        data.thresholds,  data.min_segments,
                                        data.max_segments };
    auto objdata = region_growing(rg_data, distance_function);
    if (objdata.indices_distances.size() >= data.min_segments &&
        objdata.indices_distances.size() <= data.max_segments) {
      std::vector<double> distarray;
      distarray.reserve(objdata.indices_distances.size());
      for (const auto& pair : objdata.indices_distances) {
        distarray.push_back(pair.second);
      }
      double mean_distarray =
        std::accumulate(distarray.begin(), distarray.end(), 0.0) /
        distarray.size();
      if (mean_distarray == 0.0) {
        mean_distarray = 1e-10;
      }
      double sq_sum = std::inner_product(
        distarray.begin(), distarray.end(), distarray.begin(), 0.0);
      double stdev =
        std::sqrt(sq_sum / distarray.size() - mean_distarray * mean_distarray);
      double cv = stdev / mean_distarray;
      if (cv <= 0.8) {
        objects.push_back(std::make_tuple(i, objdata));
      }
    }
  }
  return objects;
}

inline double
distance(double x, double y, double norm1, double norm2)
{
  return std::fabs(x - norm1 - (y - norm2));
}

double
dtw_distance(const TimeseriesDistanceFunctionData& data)
{
  // Create an index vector of non-NaN values
  std::vector<IndexType> indices;
  indices.reserve(data.ts1.size());
  for (IndexType i = 0; i < data.ts1.size(); ++i)
    if (!(std::isnan(data.ts1[i]) || (std::isnan(data.ts2[i]))))
      indices.push_back(i);

  // If all values were NaN, our distance is NaN
  if (indices.empty())
    return std::numeric_limits<double>::quiet_NaN();

  const auto n = indices.size();
  std::vector<std::vector<double>> d(n, std::vector<double>(n));

  // Upper left corner
  d[0][0] = distance(
    data.ts1[indices[0]], data.ts2[indices[0]], data.norm1, data.norm2);

  // Upper row and left-most column
  for (std::size_t i = 1; i < n; ++i) {
    d[i][0] =
      distance(
        data.ts1[indices[i]], data.ts2[indices[0]], data.norm1, data.norm2) +
      d[i - 1][0];
    d[0][i] =
      distance(
        data.ts1[indices[0]], data.ts2[indices[i]], data.norm1, data.norm2) +
      d[0][i - 1];
  }

  // Rest of the distance matrix
  for (std::size_t i = 1; i < n; ++i)
    for (std::size_t j = 1; j < n; ++j)
      d[i][j] =
        distance(
          data.ts1[indices[i]], data.ts2[indices[j]], data.norm1, data.norm2) +
        std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);

  return d[n - 1][n - 1];
}

double
normalized_dtw_distance(const TimeseriesDistanceFunctionData& data)
{
  // Calculate Dmax from the first timeseries
  double max_dist = 0.0;
  for (auto entry : data.ts1)
    if (!std::isnan(entry))
      max_dist += std::abs(entry);

  return std::fmin(1.0, 1.0 - (max_dist - dtw_distance(data)) / max_dist);
}

double
median_calculation(std::vector<double>& subsignal)
{                          // function calculate median of the vector
                           // the function change the vector
  if (subsignal.empty()) { // exeption
    throw std::runtime_error{ "Empty signal passed to median calculation" };
  }
  auto n = subsignal.size() / 2;
  std::nth_element(subsignal.begin(), subsignal.begin() + n, subsignal.end());
  double med = subsignal[n];
  if (subsignal.size() % 2 == 0) { // If the set size is even
    auto max_it = std::max_element(subsignal.begin(), subsignal.begin() + n);
    med = (*max_it + med) / 2.0;
  }
  return med;
}

double
median_calculation_simp(std::vector<double>& subsignal)
{
  if (subsignal.empty()) {
    throw std::runtime_error{ "Empty signal passed to median calculation" };
  }

  // Copy elements to a separate container
  std::vector<double> values = subsignal;

  auto n = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + n, values.end());

  return values[values.size() / 2];
}

std::vector<IndexType>
local_maxima_calculation(std::vector<double>& score, IndexType order)
{
  std::vector<IndexType> result;

  if (score.empty()) { // exeption
    throw std::runtime_error{ "The score is empty" };
  }
  if (order < 1) { // exeption
    throw std::runtime_error{
      "Order in local_maxima_calculation function < 1"
    };
  }
  auto n = score.size();
  if (n == 1) { // exeption
    throw std::runtime_error{ "The score contains only one element" };
  }

  std::vector<double>::const_iterator current = score.begin();

  std::vector<double>::const_iterator right_array_left_index =
    score.begin() + 1;
  std::vector<double>::const_iterator right_array_right_index =
    score.begin() + order + 1;

  std::vector<double>::const_iterator left_array_left_index = score.begin();
  std::vector<double>::const_iterator left_array_right_index = score.begin();

  std::vector<double>::const_iterator max_right_array =
    std::max_element(right_array_left_index, right_array_right_index);

  std::vector<double>::const_iterator max_left_array =
    std::max_element(left_array_left_index, left_array_right_index);

  auto distance_range = 0;
  auto max_right_array_num = 0.0;
  auto max_left_array_num = 0.0;
  auto second_max = 0.0;
  IndexType current_distance = std::distance(score.cbegin(), current);

  while (current_distance < n) // check main part of score
  {
    if (left_array_left_index == left_array_right_index) {
      max_left_array_num = 0;
    } else {
      max_left_array =
        std::max_element(left_array_left_index, left_array_right_index);
      max_left_array_num = *max_left_array;
    }

    if (right_array_left_index == right_array_right_index) {
      max_right_array_num = 0;
    } else {
      max_right_array =
        std::max_element(right_array_left_index, right_array_right_index);
      max_right_array_num = *max_right_array;
    }

    if (current_distance < order) {
      distance_range =
        order - std::distance(left_array_left_index, left_array_right_index);
      if (distance_range == order)
        max_left_array_num =
          *(std::max_element((score.cend() - distance_range), score.cend()));
      else
        max_left_array_num = std::max(
          *max_left_array,
          *(std::max_element((score.cend() - distance_range), score.cend())));
    }

    else if (current_distance > n - order - 1) {
      distance_range =
        order - std::distance(right_array_left_index, right_array_right_index);
      if (distance_range == order)
        max_right_array_num =
          *(std::max_element(score.cbegin(), score.cbegin() + distance_range));
      else
        max_right_array_num = std::max(
          *max_right_array,
          *(std::max_element(score.cbegin(), score.cbegin() + distance_range)));
    }

    if (*current > max_left_array_num && *current > max_right_array_num) {
      auto item_index = current - score.cbegin();
      result.push_back(item_index);
      if (current_distance + order + 1 < n) {
        current = current + order + 1;
        current_distance = current_distance + order + 1;
      } else
        break;
    }

    else {
      if (current_distance + 1 < n) {
        current++;
        current_distance++;
      } else
        break;
    }

    if (current == score.end() - 1) {
      right_array_left_index = score.cend();
    } else {
      right_array_left_index = current + 1;
    }

    if (current_distance >= n - order - 1) {
      right_array_right_index = score.cend();
    } else {
      right_array_right_index = current + order + 1;
    }

    left_array_right_index = current;
    if (current_distance < order)
      left_array_left_index = score.begin();
    else
      left_array_left_index = current - order;
  }
  return result;
}

double
cost_L1_error(EigenTimeSeriesConstRef signal,
              IndexType start,
              IndexType end,
              IndexType min_size)
{ // the function calculate error with cost function "l1"

  if (end < start) { // exeption
    throw std::runtime_error{ "End < Start in cost_L1_error function" };
  }

  if (start == end) {
    return 0.0;
  }

  std::vector<double> signal_subvector(signal.begin() + start,
                                       signal.begin() + end);

  double median = median_calculation(signal_subvector);

  double sum_result = std::accumulate(
    signal_subvector.begin(),
    signal_subvector.end(),
    0.0,
    [median](double a, double b) { return a + std::abs(b - median); });

  return sum_result;
}

std::vector<double>
fit_change_point_detection(EigenTimeSeriesConstRef signal,
                           IndexType width,
                           IndexType jump,
                           IndexType min_size)
{
  std::vector<double> score;
  score.reserve(signal.size());
  double gain;
  IndexType half_of_width = width / 2;

  for (int i = 0; i < signal.size(); i += jump) {
    if ((i < half_of_width) || (i >= (signal.size() - half_of_width))) {
      continue;
    }
    IndexType start = i - half_of_width;
    IndexType end = i + half_of_width;
    gain = cost_L1_error(signal, start, end, min_size);
    if (gain < 0) {
      score.push_back(0);
    }
    gain -= cost_L1_error(signal, start, i, min_size) +
            cost_L1_error(signal, i, end, min_size);
    score.push_back(gain);
  }
  return score;
}

double
sum_of_costs(EigenTimeSeriesConstRef signal,
             std::vector<IndexType>& bkps,
             IndexType min_size)
{ // input bkps array should be sorted!!!
  double result = 0.0;
  int start = 0;
  int end = 0;
  for (auto i : bkps) {
    end = i;
    result += cost_L1_error(signal, start, end, min_size);
    start = end;
  }
  return result;
}

std::vector<IndexType>
predict_change_point_detection(EigenTimeSeriesConstRef signal,
                               std::vector<double>& score,
                               IndexType width,
                               IndexType jump,
                               IndexType min_size,
                               double pen)
{
  int n_samples = signal.size();
  std::vector<IndexType> bkps;
  bkps.reserve(n_samples / width);
  int bkp;
  double gain;
  bkps.push_back(n_samples);
  bool stop = false;
  double error = sum_of_costs(signal, bkps, min_size);

  // forcing order to be above one in case jump is too large
  int preorder = std::max(width, 2 * min_size) / (2 * jump);
  int order = std::max(preorder, 1);

  std::vector<int> inds;
  inds.reserve(signal.size()); // todo: I can make it faster

  int half_of_width = width / 2;
  for (int i = 0; i < signal.size(); i += jump) {
    if ((i < half_of_width) || (i >= (signal.size() - half_of_width))) {
      continue;
    } else
      inds.push_back(i);
  }

  std::vector<IndexType> peak_inds_shifted_indx;

  peak_inds_shifted_indx = local_maxima_calculation(score, order);

  std::vector<double> gains;
  std::vector<IndexType> peak_inds_arr;
  std::vector<IndexType> peak_inds;

  for (auto i : peak_inds_shifted_indx) {
    gains.push_back(score[i]);
    peak_inds_arr.push_back(inds[i]);
  }

  std::vector<int> index_vec(peak_inds_arr.size());
  std::iota(index_vec.begin(), index_vec.end(), 0);
  std::sort(index_vec.begin(), index_vec.end(), [&](int a, int b) {
    return gains[a] < gains[b];
  });

  for (auto i : index_vec) {
    peak_inds.push_back(peak_inds_arr[i]);
  }

  while (!stop) {
    stop = true;
    if (peak_inds.size() != 0) {
      bkp = peak_inds.back();
      peak_inds.pop_back();
    } else {
      break;
    }

    if (pen > 0) {
      std::vector<IndexType> temp_bkps = bkps;
      temp_bkps.push_back(bkp);
      sort(temp_bkps.begin(), temp_bkps.end());
      gain = error - sum_of_costs(signal, temp_bkps, min_size);
      if (gain > pen) {
        stop = false;
      }
    }

    if (!stop) {
      bkps.push_back(bkp);
      sort(bkps.begin(), bkps.end());
      error = sum_of_costs(signal, bkps, min_size);
    }
  }
  return bkps;
}

std::vector<IndexType>
change_point_detection(const ChangePointDetectionData& data)
{
  std::vector<IndexType> changepoints;
  std::vector<double> score;
  score.reserve(data.ts.size());
  score = fit_change_point_detection(
    data.ts, data.window_width, data.jump, data.min_size);
  changepoints = predict_change_point_detection(
    data.ts, score, data.window_width, data.jump, data.min_size, data.penalty);

  return changepoints;
}

// compute the y-distance to the line defined by (x1, y1) and (x2, y2)
double
line_distance(const double time,
              const double value,
              const double time_start,
              const double time_end,
              const double value_start,
              const double value_end)
{
  double dx = time_end - time_start;
  double dy = value_end - value_start;
  // Handle case where start and end points are the same
  if (dx == 0 && dy == 0) {
    return (time - time_start) == 0 ? 0.0 : std::abs(value_start);
  }
  double slope = dy / dx;
  double intercept = value_start - slope * time_start;
  double expected_value = slope * time + intercept;
  return std::abs(value - expected_value);
}

std::tuple<double, double>
linear_regression(const EigenTimeSeriesConstRef times,
                  const EigenTimeSeriesConstRef distances,
                  IndexType start_index,
                  IndexType end_index)
{
  double slope, intercept;
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
  size_t n = end_index - start_index + 1;
  if (n < 2) {
    slope = 0.0;
    intercept = distances(start_index);
    return { slope, intercept };
  }
  for (size_t i = start_index; i <= end_index; ++i) {
    sum_x += times(i);
    sum_y += distances(i);
    sum_xy += times(i) * distances(i);
    sum_x2 += times(i) * times(i);
  }
  double denominator = n * sum_x2 - sum_x * sum_x;
  if (std::abs(denominator) <
      1e-9) { // Avoid division by zero if all x are the same
    slope = 0.0;
  } else {
    slope = (n * sum_xy - sum_x * sum_y) / denominator;
  }
  intercept = (sum_y - slope * sum_x) / n;
  return { slope, intercept };
}

void
ramer_douglas_peucker_recursive(const EigenTimeSeriesConstRef times,
                                const EigenTimeSeriesConstRef distances,
                                IndexType start_index,
                                IndexType end_index,
                                double epsilon,
                                std::vector<bool>& markers)
{
  if (end_index <= start_index + 1) {
    return;
  }
  double start_time = times(start_index);
  double end_time = times(end_index);
  double start_value = distances(start_index);
  double end_value = distances(end_index);
  double max_dist = 0.0;
  size_t max_dist_index = 0;
  for (size_t i = start_index + 1; i < end_index; ++i) {
    double dist = line_distance(
      times(i), distances(i), start_time, end_time, start_value, end_value);
    if (dist > max_dist) {
      max_dist = dist;
      max_dist_index = i;
    }
  }
  if (max_dist > epsilon) {
    markers[max_dist_index] = true;
    ramer_douglas_peucker_recursive(
      times, distances, start_index, max_dist_index, epsilon, markers);
    ramer_douglas_peucker_recursive(
      times, distances, max_dist_index, end_index, epsilon, markers);
  }
}

std::vector<IndexType>
ramer_douglas_peucker(const EigenTimeSeriesConstRef times,
                      const EigenTimeSeriesConstRef distances,
                      double epsilon)
{
  if (times.size() < 2) {
    return {};
  }
  std::vector<bool> markers(times.size(), false);
  markers[0] = true;
  markers[times.size() - 1] = true;
  ramer_douglas_peucker_recursive(
    times, distances, 0, times.size() - 1, epsilon, markers);
  std::vector<IndexType> result;
  for (size_t i = 0; i < markers.size(); ++i) {
    if (markers[i]) {
      result.push_back(i);
    }
  }
  return result;
}

std::vector<SeedCandidate>
seed_candidate_detection(const EigenTimeSeriesConstRef times,
                         const EigenTimeSeriesConstRef distances,
                         double epsilon,
                         double min_change_magnitude,
                         std::size_t min_period)
{
  if (times.size() < 2) {
    return {};
  }
  // Simplify time series using RDP to get polygon points
  std::vector<bool> markers(times.size(), false);
  markers[0] = true;
  markers[times.size() - 1] = true;
  ramer_douglas_peucker_recursive(
    times, distances, 0, times.size() - 1, epsilon, markers);
  std::vector<IndexType> polygon_indices;
  for (IndexType i = 0; i < markers.size(); ++i) {
    if (markers[i]) {
      polygon_indices.push_back(i);
    }
  }
  // Iterate through the RDP segments to find significant changes
  std::vector<SeedCandidate> seeds;
  int prev_end_idx = -1;
  double prev_amplitude = std::numeric_limits<double>::quiet_NaN();
  for (size_t i = 0; i < polygon_indices.size() - 1; ++i) {
    IndexType start_idx = polygon_indices[i];
    IndexType end_idx = polygon_indices[i + 1];
    if (end_idx <= start_idx)
      continue;
    // Perform linear regression on the interval
    auto [slope, intercept] =
      linear_regression(times, distances, start_idx, end_idx);
    // Calculate the amplitude of the change from the regression line
    double start_val = slope * times(start_idx) + intercept;
    double end_val = slope * times(end_idx) + intercept;
    double amplitude = end_val - start_val;
    // If amplitude exceeds the threshold, it's a seed candidate
    if (std::abs(amplitude) > min_change_magnitude) {
      // Check if this seed has same direction as previous one
      if (!std::isnan(prev_amplitude) && (amplitude * prev_amplitude > 0) &&
          (prev_end_idx == start_idx)) {
        // Merge with previous seed
        seeds.back().end_epoch = end_idx;
      } else {
        // Add new seed
        seeds.push_back({ start_idx, end_idx });
      }
      prev_amplitude = amplitude;
      prev_end_idx = end_idx;
    } else {
      prev_amplitude = std::numeric_limits<double>::quiet_NaN();
      prev_end_idx = -1;
    }
  }
  // Filter seeds based on min and max segments
  std::vector<SeedCandidate> filtered_seeds;
  for (const auto& seed : seeds) {
    std::size_t segment_length = seed.end_epoch - seed.start_epoch + 1;
    if (segment_length >= min_period) {
      filtered_seeds.push_back(seed);
    }
  }
  return filtered_seeds;
}

std::vector<std::vector<int>>
obc_fusion(std::vector<ObjectByChange>& objects,
           double spatial_iou_threshold,
           double temporal_iou_threshold)
{
  std::vector<std::vector<int>> adj_list(objects.size());
  for (int i = 0; i < objects.size(); i++) {
    for (int j = i + 1; j < objects.size(); j++) {
      // check sign
      int sign1 = objects[i].threshold > 0 ? 1 : -1;
      int sign2 = objects[j].threshold > 0 ? 1 : -1;
      if (sign1 != sign2) {
        continue;
      }
      // check spatial overlap
      std::set<IndexType> points1, points2;
      for (const auto& pair : objects[i].indices_distances) {
        points1.insert(pair.first);
      }
      for (const auto& pair : objects[j].indices_distances) {
        points2.insert(pair.first);
      }
      std::vector<IndexType> intersection_spatial;
      std::set_intersection(points1.begin(),
                            points1.end(),
                            points2.begin(),
                            points2.end(),
                            std::back_inserter(intersection_spatial));
      std::vector<IndexType> union_spatial;
      std::set_union(points1.begin(),
                     points1.end(),
                     points2.begin(),
                     points2.end(),
                     std::back_inserter(union_spatial));
      double iou_spatial =
        union_spatial.empty()
          ? 0.0
          : static_cast<double>(intersection_spatial.size()) /
              union_spatial.size();
      if (iou_spatial < spatial_iou_threshold) {
        continue;
      }
      // check temporal overlap
      std::set<IndexType> epochs1, epochs2;
      for (IndexType e = objects[i].start_epoch; e <= objects[i].end_epoch;
           ++e) {
        epochs1.insert(e);
      }
      for (IndexType e = objects[j].start_epoch; e <= objects[j].end_epoch;
           ++e) {
        epochs2.insert(e);
      }
      std::vector<IndexType> intersection_temporal;
      std::set_intersection(epochs1.begin(),
                            epochs1.end(),
                            epochs2.begin(),
                            epochs2.end(),
                            std::back_inserter(intersection_temporal));
      std::vector<IndexType> union_temporal;
      std::set_union(epochs1.begin(),
                     epochs1.end(),
                     epochs2.begin(),
                     epochs2.end(),
                     std::back_inserter(union_temporal));
      double iou_temporal =
        union_temporal.empty()
          ? 0.0
          : static_cast<double>(intersection_temporal.size()) /
              union_temporal.size();
      if (iou_temporal < temporal_iou_threshold) {
        continue;
      }
      adj_list[i].push_back(j);
      adj_list[j].push_back(i);
    }
  }

  std::function<void(int,
                     const std::vector<std::vector<int>>&,
                     std::vector<bool>&,
                     std::vector<int>&)>
    _dfs_find_component;
  _dfs_find_component =
    [&_dfs_find_component](int node_idx,
                           const std::vector<std::vector<int>>& adj_list,
                           std::vector<bool>& visited,
                           std::vector<int>& component) {
      visited[node_idx] = true;
      component.push_back(node_idx);
      for (int neighbor : adj_list[node_idx]) {
        if (!visited[neighbor]) {
          _dfs_find_component(neighbor, adj_list, visited, component);
        }
      }
    };
  std::vector<bool> visited(objects.size(), false);
  std::vector<std::vector<int>> all_components;
  for (int i = 0; i < objects.size(); ++i) {
    if (!visited[i]) {
      std::vector<int> component_indices;
      _dfs_find_component(i, adj_list, visited, component_indices);
      all_components.push_back(component_indices);
    }
  }

  return all_components;
}

std::tuple<EigenSpatiotemporalArray, EigenSpatiotemporalArray>
weighted_average_filtering(EigenSpatiotemporalArrayConstRef distances,
                           EigenSpatiotemporalArrayConstRef uncertainties,
                           int window_size)
{
  if (distances.rows() != uncertainties.rows() ||
      distances.cols() != uncertainties.cols()) {
    throw std::runtime_error{
      "Distance and uncertainty arrays have different shapes"
    };
  }

  EigenSpatiotemporalArray filtered =
    EigenSpatiotemporalArray::Zero(distances.rows(), distances.cols());
  EigenSpatiotemporalArray filtered_uncertainties =
    EigenSpatiotemporalArray::Zero(distances.rows(), distances.cols());

#pragma omp parallel for
  for (int i = 0; i < distances.rows(); ++i) {
    for (int j = 0; j < distances.cols(); ++j) {
      int start = std::max(0, j - window_size / 2);
      int end =
        std::min(static_cast<int>(distances.cols() - 1), j + window_size / 2);

      double N = 0.0;
      double n = 0.0;
      double q = 0.0;
      int f = -1;
      for (int k = start; k <= end; ++k) {
        double weight = 0.0;
        if (!std::isnan(uncertainties(i, k)) && !std::isnan(distances(i, k)) &&
            uncertainties(i, k) > 0.0) {
          weight = 1.0 / uncertainties(i, k);
          f += 1;
        }
        N += weight;
        n += weight * distances(i, k);
        q += weight * distances(i, k) * distances(i, k);
      }
      double Q = 1 / N;
      double mean = Q * n;
      double s0 = (q - n * mean) / f;
      double C = s0 * Q;
      if (f <= 0) {
        filtered(i, j) = std::numeric_limits<double>::quiet_NaN();
        filtered_uncertainties(i, j) = std::numeric_limits<double>::quiet_NaN();
      } else {
        filtered(i, j) = mean;
        filtered_uncertainties(i, j) = C;
      }
    }
  }

  return { std::move(filtered), std::move(filtered_uncertainties) };
}

std::tuple<EigenSpatiotemporalArray, EigenSpatiotemporalArray>
robust_weighted_average_filtering(
  EigenSpatiotemporalArrayConstRef distances,
  EigenSpatiotemporalArrayConstRef uncertainties,
  int window_size)
{
  if (distances.rows() != uncertainties.rows() ||
      distances.cols() != uncertainties.cols()) {
    throw std::runtime_error{
      "Distance and uncertainty arrays have different shapes"
    };
  }
  if (window_size % 2 == 1) {
    window_size -= 1;
  }

  EigenSpatiotemporalArray filtered =
    EigenSpatiotemporalArray::Zero(distances.rows(), distances.cols());
  EigenSpatiotemporalArray filtered_uncertainties =
    EigenSpatiotemporalArray::Zero(distances.rows(), distances.cols());

  std::vector<bool> inliers(window_size);

#pragma omp parallel for firstprivate(inliers)
  for (int i = 0; i < distances.rows(); ++i) {
    for (int j = 0; j < distances.cols(); ++j) {
      int start = std::max(0, j - window_size / 2);
      int end =
        std::min(static_cast<int>(distances.cols() - 1), j + window_size / 2);
      int num_elements = end - start + 1;

      // Detect inliers using RANSAC
      inliers.assign(inliers.size(), false);
      int max_inliers = 0;
      for (int iter = 0; iter < num_elements * 3; ++iter) {
        int random_index = start + (std::rand() % num_elements);
        if (std::isnan(distances(i, random_index)) ||
            std::isnan(uncertainties(i, random_index)) ||
            uncertainties(i, random_index) <= 0.0) {
          continue;
        }
        double model = distances(i, random_index);
        int current_inliers = 0;
        for (int k = start; k <= end; ++k) {
          if (std::isnan(distances(i, k)) || std::isnan(uncertainties(i, k)) ||
              uncertainties(i, k) <= 0.0) {
            continue;
          }
          double threshold = 1.96 * std::sqrt(uncertainties(i, k));
          if (std::fabs(distances(i, k) - model) > threshold) {
            continue;
          }
          current_inliers += 1;
        }
        if (current_inliers > max_inliers) {
          max_inliers = current_inliers;
          inliers.assign(inliers.size(), false);
          for (int k = start; k <= end; ++k) {
            if (std::isnan(distances(i, k)) ||
                std::isnan(uncertainties(i, k)) || uncertainties(i, k) <= 0.0) {
              continue;
            }
            double threshold = 1.96 * std::sqrt(uncertainties(i, k));
            if (std::fabs(distances(i, k) - model) > threshold) {
              continue;
            }
            inliers[k - start] = true;
          }
        }
        if (max_inliers > (num_elements * 4 / 3)) {
          break;
        }
      }

      // Compute weighted mean
      double N = 0.0;
      double n = 0.0;
      double q = 0.0;
      int f = -1;
      for (int k = start; k <= end; ++k) {
        double weight = 0.0;
        if (!std::isnan(uncertainties(i, k)) && !std::isnan(distances(i, k)) &&
            uncertainties(i, k) > 0.0 && inliers[k - start]) {
          weight = 1.0 / uncertainties(i, k);
          f += 1;
        }
        N += weight;
        n += weight * distances(i, k);
        q += weight * distances(i, k) * distances(i, k);
      }
      double Q = 1 / N;
      double mean = Q * n;
      double s0 = (q - n * mean) / f;
      double C = s0 * Q;
      if (f <= 0) {
        filtered(i, j) = std::numeric_limits<double>::quiet_NaN();
        filtered_uncertainties(i, j) = std::numeric_limits<double>::quiet_NaN();
      } else {
        filtered(i, j) = mean;
        filtered_uncertainties(i, j) = C;
      }
    }
  }

  return { std::move(filtered), std::move(filtered_uncertainties) };
}

std::tuple<EigenSpatiotemporalArray, EigenSpatiotemporalArray>
_robust_weighted_average_filtering(
  EigenSpatiotemporalArrayConstRef distances,
  EigenSpatiotemporalArrayConstRef uncertainties,
  int window_size)
{
  if (distances.rows() != uncertainties.rows() ||
      distances.cols() != uncertainties.cols()) {
    throw std::runtime_error{
      "Distance and uncertainty arrays have different shapes"
    };
  }

  EigenSpatiotemporalArray filtered =
    EigenSpatiotemporalArray::Zero(distances.rows(), distances.cols());
  EigenSpatiotemporalArray filtered_uncertainties =
    EigenSpatiotemporalArray::Zero(distances.rows(), distances.cols());

  std::vector<double> weights(window_size);
#pragma omp parallel for firstprivate(weights)
  for (int i = 0; i < distances.rows(); ++i) {
    for (int j = 0; j < distances.cols(); ++j) {
      int start = std::max(0, j - window_size / 2);
      int end =
        std::min(static_cast<int>(distances.cols() - 1), j + window_size / 2);

      double weight_sum = 0.0;
      for (int k = start; k <= end; ++k) {
        double weight = 0.0;
        if (!std::isnan(uncertainties(i, k)) && !std::isnan(distances(i, k)) &&
            uncertainties(i, k) > 0.0) {
          weight = 1.0 / uncertainties(i, k);
        }
        weights[k - start] = weight;
        weight_sum += weight;
      }
      while (true) {
        int f = -1;
        double mean = 0.0;
        for (int k = start; k <= end; ++k) {
          if (weights[k - start] > 0.0) {
            mean += weights[k - start] * distances(i, k);
            f += 1;
          }
        }
        mean /= weight_sum;
        double s0 = 0.0;
        for (int k = start; k <= end; ++k) {
          if (weights[k - start] > 0.0) {
            s0 += weights[k - start] * std::pow(distances(i, k) - mean, 2);
          }
        }
        s0 = s0 / f;
        double max_w = 0.0;
        int max_w_index = -1;
        for (int k = start; k <= end; ++k) {
          if (weights[k - start] > 0.0) {
            double v = distances(i, k) - mean;
            double w =
              v / std::sqrt(s0 * (1.0 / weights[k - start] - 1.0 / weight_sum));
            if (std::fabs(w) > 2.0 && std::fabs(w) > max_w) {
              max_w = std::fabs(w);
              max_w_index = k - start;
            }
          }
        }
        if (max_w_index == -1) {
          if (f <= 0) {
            filtered(i, j) = std::numeric_limits<double>::quiet_NaN();
            filtered_uncertainties(i, j) =
              std::numeric_limits<double>::quiet_NaN();
          } else {
            filtered(i, j) = mean;
            filtered_uncertainties(i, j) = s0 * (1.0 / weight_sum);
          }
          break;
        } else {
          weight_sum -= weights[max_w_index];
          weights[max_w_index] = 0.0;
        }
      }
    }
  }

  return { std::move(filtered), std::move(filtered_uncertainties) };
}
}
