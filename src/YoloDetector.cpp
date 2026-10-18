#include "YoloDetector.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace {

// YOLO often reports one object under two similar classes (tv and laptop) with
// nearly the same box. Boxes overlapping this much are merged across classes.
constexpr float kCrossClassIou = 0.7f;

}  // namespace

YoloDetector::YoloDetector(const std::string& modelPath, int numClasses,
                           DetectorThresholds thresholds, cv::Size inputSize)
    : numClasses_(numClasses), th_(thresholds), inputSize_(inputSize) {
    requireModelFile(modelPath);
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty()) throw std::runtime_error("Failed to load YOLO model: " + modelPath);
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    // Letterbox: scale to fit, keep the aspect ratio, pad the rest with gray.
    const float scale = std::min(static_cast<float>(inputSize_.width) / frame.cols,
                                 static_cast<float>(inputSize_.height) / frame.rows);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::Mat input(inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(input(cv::Rect(0, 0, resized.cols, resized.rows)));

    net_.setInput(
        cv::dnn::blobFromImage(input, 1.0 / 255.0, inputSize_, cv::Scalar(), true, false));
    const cv::Mat out = net_.forward();
    if (out.size[1] - 4 != numClasses_)
        throw std::runtime_error("YOLO model has " + std::to_string(out.size[1] - 4) +
                                 " classes but coco.names lists " + std::to_string(numClasses_));
    return decodeYolo(out, scale, frame.size(), th_);
}

std::vector<Detection> decodeYolo(const cv::Mat& output, float scale, cv::Size frameSize,
                                  const DetectorThresholds& th) {
    const int attrs = output.size[1];
    const int numBoxes = output.size[2];
    const int numClasses = attrs - 4;
    // [attrs, boxes] -> [boxes, attrs]: one row per candidate box.
    const cv::Mat preds =
        cv::Mat(attrs, numBoxes, CV_32F, const_cast<float*>(output.ptr<float>())).t();

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    for (int i = 0; i < preds.rows; ++i) {
        const float* row = preds.ptr<float>(i);
        const float* best = std::max_element(row + 4, row + 4 + numClasses);
        if (*best < th.low) continue;
        const float cx = row[0] / scale, cy = row[1] / scale;
        const float w = row[2] / scale, h = row[3] / scale;
        boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
        scores.push_back(*best);
        classIds.push_back(static_cast<int>(best - (row + 4)));
    }

    // Duplicates within a class first, then near-identical boxes across classes.
    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(boxes, scores, classIds, th.low, th.nms, keep);
    std::vector<cv::Rect2d> keptBoxes;
    std::vector<float> keptScores;
    for (int k : keep) {
        keptBoxes.push_back(boxes[k]);
        keptScores.push_back(scores[k]);
    }
    std::vector<int> keepAcross;
    cv::dnn::NMSBoxes(keptBoxes, keptScores, th.low, kCrossClassIou, keepAcross);

    const cv::Rect2f bounds(0, 0, static_cast<float>(frameSize.width),
                            static_cast<float>(frameSize.height));
    std::vector<Detection> result;
    for (int j : keepAcross) {
        const int k = keep[j];
        const cv::Rect2f box = cv::Rect2f(boxes[k]) & bounds;
        if (box.area() <= 0) continue;
        Detection d;
        d.box = box;
        d.classId = classIds[k];
        d.conf = scores[k];
        d.weak = scores[k] < th.conf;
        result.push_back(std::move(d));
    }
    return result;
}
