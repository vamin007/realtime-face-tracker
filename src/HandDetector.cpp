#include "HandDetector.h"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {

constexpr int kInputSize = 192;  // the palm model's square input
constexpr int kKeypoints = 7;

// Hand geometry relative to the palm box size: the hand is about twice as long
// as the palm (fingers included), a bit wider (thumb), and centered half a palm
// toward the fingers.
constexpr float kHandLength = 2.0f;
constexpr float kHandWidth = 1.5f;
constexpr float kHandShift = 0.5f;

// SSD anchors of MediaPipe's 192x192 palm model: a 24x24 grid with 2 anchors per
// cell (stride 8), then a 12x12 grid with 6 per cell (stride 16). 2016 in total.
std::vector<cv::Point2f> palmAnchors() {
    std::vector<cv::Point2f> anchors;
    anchors.reserve(2016);
    for (int y = 0; y < 24; ++y)
        for (int x = 0; x < 24; ++x)
            for (int k = 0; k < 2; ++k) anchors.emplace_back((x + 0.5f) / 24, (y + 0.5f) / 24);
    for (int y = 0; y < 12; ++y)
        for (int x = 0; x < 12; ++x)
            for (int k = 0; k < 6; ++k) anchors.emplace_back((x + 0.5f) / 12, (y + 0.5f) / 12);
    return anchors;
}

}  // namespace

cv::Rect2f handBoxFromPalm(const cv::Rect2f& palm, const std::vector<cv::Point2f>& keypoints) {
    const cv::Point2f center(palm.x + palm.width / 2, palm.y + palm.height / 2);
    const float size = std::max(palm.width, palm.height);
    cv::Point2f dir(0, -1);  // fingers up when the keypoints don't say otherwise
    if (keypoints.size() >= 3) {
        const cv::Point2f v = keypoints[2] - keypoints[0];  // wrist -> middle finger base
        const float len = std::hypot(v.x, v.y);
        if (len > 1e-3f) dir = v / len;
    }
    const cv::Point2f handCenter = center + dir * (kHandShift * size);
    const float angle = std::atan2(dir.y, dir.x) * 180.0f / static_cast<float>(CV_PI);
    const cv::RotatedRect hand(handCenter, cv::Size2f(kHandLength * size, kHandWidth * size),
                               angle);
    return hand.boundingRect2f();
}

HandDetector::HandDetector(const std::string& modelPath, ClassId handClass,
                           DetectorThresholds thresholds)
    : handClass_(handClass), th_(thresholds), anchors_(palmAnchors()) {
    requireModelFile(modelPath);
    net_ = cv::dnn::readNet(modelPath);
    if (net_.empty()) throw std::runtime_error("Failed to load hand model: " + modelPath);
}

std::vector<Detection> HandDetector::detect(const cv::Mat& frame) {
    // Letterbox to 192x192, centered, padded with black.
    const float ratio = std::min(static_cast<float>(kInputSize) / frame.cols,
                                 static_cast<float>(kInputSize) / frame.rows);
    const cv::Size fitted(static_cast<int>(frame.cols * ratio),
                          static_cast<int>(frame.rows * ratio));
    cv::Mat resized;
    cv::resize(frame, resized, fitted);
    const int padX = (kInputSize - fitted.width) / 2;
    const int padY = (kInputSize - fitted.height) / 2;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padY, kInputSize - fitted.height - padY, padX,
                       kInputSize - fitted.width - padX, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // NHWC float RGB blob [1, 192, 192, 3] in 0..1.
    cv::Mat rgb, floatImg;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    const int shape[] = {1, kInputSize, kInputSize, 3};
    net_.setInput(cv::Mat(4, shape, CV_32F, floatImg.data));
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.size() < 2) return {};

    // outputs[0]: per anchor 18 values (box cx, cy, w, h, then 7 keypoint x, y),
    // as offsets in input pixels from the anchor. outputs[1]: per anchor a score logit.
    const cv::Mat raw = outputs[0].reshape(1, static_cast<int>(outputs[0].total() / 18));
    const cv::Mat logits = outputs[1].reshape(1, static_cast<int>(outputs[1].total()));
    const float scale = static_cast<float>(std::max(frame.cols, frame.rows));
    const cv::Point2f padBias(padX / ratio, padY / ratio);
    auto toFrame = [&](float dx, float dy, const cv::Point2f& anchor) {
        return cv::Point2f((dx / kInputSize + anchor.x) * scale,
                           (dy / kInputSize + anchor.y) * scale) -
               padBias;
    };

    std::vector<cv::Rect2d> palms;
    std::vector<float> scores;
    std::vector<int> rows;
    const int n = std::min(logits.rows, static_cast<int>(anchors_.size()));
    for (int i = 0; i < n; ++i) {
        const float score = 1.0f / (1.0f + std::exp(-logits.at<float>(i, 0)));
        if (score < th_.low) continue;
        const float* b = raw.ptr<float>(i);
        const cv::Point2f tl = toFrame(b[0] - b[2] / 2, b[1] - b[3] / 2, anchors_[i]);
        const cv::Point2f br = toFrame(b[0] + b[2] / 2, b[1] + b[3] / 2, anchors_[i]);
        palms.emplace_back(tl.x, tl.y, std::max(1.0f, br.x - tl.x), std::max(1.0f, br.y - tl.y));
        scores.push_back(score);
        rows.push_back(i);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(palms, scores, th_.low, th_.nms, keep);

    const cv::Rect2f bounds(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
    std::vector<Detection> result;
    for (int k : keep) {
        const float* b = raw.ptr<float>(rows[k]);
        Detection d;
        for (int p = 0; p < kKeypoints; ++p)
            d.keypoints.push_back(toFrame(b[4 + 2 * p], b[5 + 2 * p], anchors_[rows[k]]));
        d.box = handBoxFromPalm(cv::Rect2f(palms[k]), d.keypoints) & bounds;
        if (d.box.area() <= 0) continue;
        d.classId = handClass_;
        d.conf = scores[k];
        d.weak = d.conf < th_.conf;
        result.push_back(std::move(d));
    }
    return result;
}
