// FaceDetector: wraps OpenCV's YuNet (FaceDetectorYN), downscaling wide frames for speed and
// mapping boxes and landmarks back to full-frame pixels. See FaceDetector.h for the overview.
#include "FaceDetector.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

FaceDetector::FaceDetector(const std::string& modelPath, ClassId faceClass,
                           DetectorThresholds thresholds, int maxWidth)
    : faceClass_(faceClass), th_(thresholds), maxWidth_(maxWidth) {
    requireModelFile(modelPath);
    // The input size is set per frame in detect().
    // The network itself filters at the *low* score threshold and applies NMS, so weak faces
    // come back too; detect() marks the ones below th_.conf as weak.
    net_ = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320), th_.low, th_.nms);
    if (!net_) throw std::runtime_error("Failed to load face model: " + modelPath);
}

std::string FaceDetector::info() const {
    return (maxWidth_ > 0 ? "up to " + std::to_string(maxWidth_) + " px wide, " : "") +
           std::string("OpenCV CPU");
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& frame) {
    // scale: network pixels per frame pixel (1 when the frame is narrow enough or scaling is off).
    const float scale =
        maxWidth_ > 0 && frame.cols > maxWidth_ ? static_cast<float>(maxWidth_) / frame.cols : 1.0f;
    cv::Mat input = frame;
    // INTER_AREA is OpenCV's resampling mode for shrinking images (it averages source pixels).
    if (scale < 1.0f) cv::resize(frame, input, cv::Size(), scale, scale, cv::INTER_AREA);
    // YuNet must be told the size of the image it will get, but only when that changes.
    if (input.size() != inputSize_) {
        net_->setInputSize(input.size());
        inputSize_ = input.size();
    }
    cv::Mat faces;  // one row per face: x, y, w, h, 5 landmark (x, y) pairs, score
    net_->detect(input, faces);

    // Divide by `scale` to get back to frame pixels, and clip to the frame.
    const cv::Rect2f bounds(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
    std::vector<Detection> result;
    for (int i = 0; i < faces.rows; ++i) {
        const float* f = faces.ptr<float>(i);
        const cv::Rect2f box =
            cv::Rect2f(f[0] / scale, f[1] / scale, f[2] / scale, f[3] / scale) & bounds;
        if (box.area() <= 0) continue;
        Detection d;
        d.box = box;
        d.classId = faceClass_;
        d.conf = f[14];
        d.weak = d.conf < th_.conf;
        // The 5 landmarks, in YuNet's order (see Detection::keypoints).
        for (int k = 0; k < 5; ++k)
            d.keypoints.emplace_back(f[4 + 2 * k] / scale, f[5 + 2 * k] / scale);
        result.push_back(std::move(d));
    }
    return result;
}
