#include "Tracker.h"

#include <algorithm>
#include <tuple>

namespace {

// State: [cx, cy, w, h, vcx, vcy, vw, vh]; measurement: [cx, cy, w, h].
constexpr int kState = 8;
constexpr int kMeas = 4;

cv::Mat toMeasurement(const cv::Rect2f &r) {
  return cv::Mat(
      cv::Vec4f(r.x + r.width / 2, r.y + r.height / 2, r.width, r.height));
}

cv::Rect2f toRect(const cv::Mat &state) {
  const float cx = state.at<float>(0), cy = state.at<float>(1);
  const float w = std::max(state.at<float>(2), 1.0f);
  const float h = std::max(state.at<float>(3), 1.0f);
  return {cx - w / 2, cy - h / 2, w, h};
}

float iou(const cv::Rect2f &a, const cv::Rect2f &b) {
  const float inter = (a & b).area();
  const float uni = a.area() + b.area() - inter;
  return uni > 0 ? inter / uni : 0.0f;
}

cv::Point2f center(const cv::Rect2f &r) {
  return {r.x + r.width / 2, r.y + r.height / 2};
}

} // namespace

Track Tracker::createTrack(const Detection &det) {
  Track t{nextId_++, det.classId, det.conf, det.box,
          cv::KalmanFilter(kState, kMeas, 0, CV_32F)};

  // x' = x + v each frame.
  cv::setIdentity(t.kf.transitionMatrix);
  for (int i = 0; i < kMeas; ++i)
    t.kf.transitionMatrix.at<float>(i, i + kMeas) = 1.0f;
  cv::setIdentity(t.kf.measurementMatrix); // picks the first 4 state values

  // Noise values follow SORT: trust detections enough that the filter reacts
  // quickly when an object changes direction (otherwise it loses the match).
  cv::setIdentity(t.kf.processNoiseCov, cv::Scalar::all(1.0));
  for (int i = kMeas; i < kState; ++i)
    t.kf.processNoiseCov.at<float>(i, i) = 1e-2f;
  cv::setIdentity(t.kf.measurementNoiseCov, cv::Scalar::all(1.0));
  for (int i = 2; i < kMeas; ++i)
    t.kf.measurementNoiseCov.at<float>(i, i) = 10.0f; // box size is noisier
  cv::setIdentity(t.kf.errorCovPost, cv::Scalar::all(10.0));
  for (int i = kMeas; i < kState; ++i)
    t.kf.errorCovPost.at<float>(i, i) = 1000.0f; // velocity unknown

  t.kf.statePost = cv::Mat::zeros(kState, 1, CV_32F);
  toMeasurement(det.box).copyTo(t.kf.statePost.rowRange(0, kMeas));

  t.confirmed = params_.minHits <= 1;
  t.trail.push_back(center(det.box));
  return t;
}

void Tracker::update(const std::vector<Detection> &detections) {
  // 1. Predict where every existing track should be now.
  for (Track &t : tracks_)
    t.box = toRect(t.kf.predict());

  // 2. Score every same-class (track, detection) pair by overlap.
  std::vector<std::tuple<float, size_t, size_t>>
      pairs; // (iou, track, detection)
  for (size_t ti = 0; ti < tracks_.size(); ++ti)
    for (size_t di = 0; di < detections.size(); ++di) {
      if (tracks_[ti].classId != detections[di].classId)
        continue;
      const float score = iou(tracks_[ti].box, detections[di].box);
      if (score >= params_.iouThreshold)
        pairs.emplace_back(score, ti, di);
    }

  // 3. Greedy matching: best overlaps first, each track/detection used once.
  std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  std::vector<bool> trackUsed(tracks_.size(), false),
      detUsed(detections.size(), false);
  for (const auto &[score, ti, di] : pairs) {
    if (trackUsed[ti] || detUsed[di])
      continue;
    trackUsed[ti] = detUsed[di] = true;

    Track &t = tracks_[ti];
    t.box = toRect(t.kf.correct(toMeasurement(detections[di].box)));
    t.conf = detections[di].conf;
    t.misses = 0;
    if (++t.hits >= params_.minHits)
      t.confirmed = true;
    t.trail.push_back(center(t.box));
    if (t.trail.size() > params_.trailLength)
      t.trail.pop_front();
  }

  // 4. Unmatched tracks age; unmatched detections become new tracks.
  for (size_t ti = 0; ti < tracks_.size(); ++ti)
    if (!trackUsed[ti])
      ++tracks_[ti].misses;
  for (size_t di = 0; di < detections.size(); ++di)
    if (!detUsed[di])
      tracks_.push_back(createTrack(detections[di]));

  // 5. Forget tracks that have been lost too long (or never got confirmed).
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track &t) {
                                 return t.misses > params_.maxMisses ||
                                        (!t.confirmed && t.misses > 0);
                               }),
                tracks_.end());
}
