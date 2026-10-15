#include "Detector.h"

#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "FaceDetector.h"
#include "HandDetector.h"

Detector::Detector(const std::string& modelPath, const std::string& namesPath, int inputSize,
                   const std::string& faceModelPath, const std::string& handModelPath)
    : inputSize_(inputSize) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty()) throw std::runtime_error("Failed to load model: " + modelPath);

    std::ifstream in(namesPath);
    if (!in) throw std::runtime_error("Failed to open class names: " + namesPath);
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) names_.push_back(line);

    personClassId_ = classIdFor("person");

    // Ensure 'hand' and 'face' classes exist in the list of known classes.
    if (classIdFor("hand") < 0) names_.push_back("hand");
    handClassId_ = classIdFor("hand");

    if (classIdFor("face") < 0) names_.push_back("face");
    faceClassId_ = classIdFor("face");

    if (!faceModelPath.empty()) {
        faceDetector_ = std::make_unique<FaceDetector>(faceModelPath);
    }
    if (!handModelPath.empty()) {
        handDetector_ = std::make_unique<HandDetector>(handModelPath);
    }
}

Detector::~Detector() = default;

int Detector::classIdFor(const std::string& name) const {
    for (size_t i = 0; i < names_.size(); ++i)
        if (names_[i] == name) return static_cast<int>(i);
    return -1;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame, float confThreshold,
                                        float nmsThreshold, const std::set<int>& allowed) {
    std::vector<Detection> result;
    const cv::Rect2f bounds(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));

    // 1. YOLOv8 general object detection
    if (!net_.empty()) {
        const float scale = std::min(static_cast<float>(inputSize_) / frame.cols,
                                     static_cast<float>(inputSize_) / frame.rows);
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::Mat input(inputSize_, inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(input(cv::Rect(0, 0, resized.cols, resized.rows)));

        cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(inputSize_, inputSize_),
                                              cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat out = net_.forward();

        const int attrs = out.size[1];
        const int numBoxes = out.size[2];
        cv::Mat preds = cv::Mat(attrs, numBoxes, CV_32F, out.ptr<float>()).t();
        const int numClasses = attrs - 4;

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;
        for (int i = 0; i < preds.rows; ++i) {
            const float* row = preds.ptr<float>(i);
            cv::Mat classScores(1, numClasses, CV_32F, const_cast<float*>(row + 4));
            cv::Point maxLoc;
            double maxScore;
            cv::minMaxLoc(classScores, nullptr, &maxScore, nullptr, &maxLoc);
            if (maxScore < confThreshold) continue;

            // When personByFace is active (and the face model loaded), skip YOLO's
            // full-body person box; otherwise fall back to it so people are still tracked.
            if (personByFace_ && faceDetector_ && faceDetector_->isLoaded() &&
                maxLoc.x == personClassId_)
                continue;

            if (!allowed.empty() && !allowed.count(maxLoc.x)) continue;

            const float cx = row[0] / scale, cy = row[1] / scale;
            const float w = row[2] / scale, h = row[3] / scale;
            boxes.emplace_back(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                               static_cast<int>(w), static_cast<int>(h));
            scores.push_back(static_cast<float>(maxScore));
            classIds.push_back(maxLoc.x);
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxesBatched(boxes, scores, classIds, confThreshold, nmsThreshold, keep);

        for (int idx : keep) {
            cv::Rect2f box = cv::Rect2f(boxes[idx]) & bounds;
            if (box.area() <= 0) continue;
            result.push_back({box, classIds[idx], scores[idx]});
        }
    }

    // 2. Face detection: identifies a person strictly through their face.
    const bool allowPerson = allowed.empty() || (personClassId_ >= 0 && allowed.count(personClassId_));
    const bool allowFace = allowed.empty() || (faceClassId_ >= 0 && allowed.count(faceClassId_));
    if ((allowPerson || allowFace) && faceDetector_ && faceDetector_->isLoaded()) {
        const int targetClass = (personByFace_ && allowPerson) ? personClassId_ : faceClassId_;
        auto faceDets = faceDetector_->detect(frame, confThreshold, targetClass);
        for (auto& det : faceDets) {
            if (allowed.empty() || allowed.count(det.classId)) {
                result.push_back(det);
            }
        }
    }

    // 3. Hand detection: identifies hands via palm model.
    const bool allowHand = enableHands_ && (allowed.empty() || (handClassId_ >= 0 && allowed.count(handClassId_)));
    if (allowHand && handDetector_ && handDetector_->isLoaded()) {
        auto handDets = handDetector_->detect(frame, confThreshold, handClassId_);
        for (auto& det : handDets) {
            if (allowed.empty() || allowed.count(det.classId)) {
                result.push_back(det);
            }
        }
    }

    return result;
}
