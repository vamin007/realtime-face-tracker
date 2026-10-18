#include "FaceDetector.h"

#include <stdexcept>

FaceDetector::FaceDetector(const std::string& modelPath, ClassId faceClass,
                           DetectorThresholds thresholds)
    : faceClass_(faceClass), th_(thresholds) {
    requireModelFile(modelPath);
    // The input size is set per frame in detect().
    net_ = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320), th_.low, th_.nms);
    if (!net_) throw std::runtime_error("Failed to load face model: " + modelPath);
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& frame) {
    if (frame.size() != inputSize_) {
        net_->setInputSize(frame.size());
        inputSize_ = frame.size();
    }
    cv::Mat faces;  // one row per face: x, y, w, h, 5 landmark (x, y) pairs, score
    net_->detect(frame, faces);

    const cv::Rect2f bounds(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
    std::vector<Detection> result;
    for (int i = 0; i < faces.rows; ++i) {
        const float* f = faces.ptr<float>(i);
        const cv::Rect2f box = cv::Rect2f(f[0], f[1], f[2], f[3]) & bounds;
        if (box.area() <= 0) continue;
        Detection d;
        d.box = box;
        d.classId = faceClass_;
        d.conf = f[14];
        d.weak = d.conf < th_.conf;
        for (int k = 0; k < 5; ++k) d.keypoints.emplace_back(f[4 + 2 * k], f[5 + 2 * k]);
        result.push_back(std::move(d));
    }
    return result;
}
