#include "HandDetector.h"

#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>

HandDetector::HandDetector() = default;

HandDetector::HandDetector(const std::string& modelPath, float scoreThresh, float nmsThresh) {
    load(modelPath, scoreThresh, nmsThresh);
}

std::vector<cv::Point2f> HandDetector::generateAnchors() {
    std::vector<cv::Point2f> anchors;
    anchors.reserve(2016);
    // Stride 8: 24x24 grid, 2 anchors per cell
    for (int y = 0; y < 24; ++y) {
        const float yc = (y + 0.5f) / 24.0f;
        for (int x = 0; x < 24; ++x) {
            const float xc = (x + 0.5f) / 24.0f;
            anchors.emplace_back(xc, yc);
            anchors.emplace_back(xc, yc);
        }
    }
    // Stride 16: 12x12 grid, 6 anchors per cell
    for (int y = 0; y < 12; ++y) {
        const float yc = (y + 0.5f) / 12.0f;
        for (int x = 0; x < 12; ++x) {
            const float xc = (x + 0.5f) / 12.0f;
            for (int k = 0; k < 6; ++k) {
                anchors.emplace_back(xc, yc);
            }
        }
    }
    return anchors;
}

bool HandDetector::load(const std::string& modelPath, float scoreThresh, float nmsThresh) {
    scoreThresh_ = scoreThresh;
    nmsThresh_ = nmsThresh;
    try {
        net_ = cv::dnn::readNet(modelPath);
        if (net_.empty()) return false;
        anchors_ = generateAnchors();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load HandDetector from " << modelPath << ": " << e.what() << '\n';
        return false;
    }
}

std::vector<Detection> HandDetector::detect(const cv::Mat& frame, float confThreshold, int classId) {
    std::vector<Detection> detections;
    if (net_.empty() || frame.empty()) return detections;

    const float thresh = confThreshold > 0.0f ? confThreshold : scoreThresh_;

    // Letterbox to 192x192
    constexpr int kInputSize = 192;
    const float ratio = std::min(static_cast<float>(kInputSize) / frame.cols,
                                 static_cast<float>(kInputSize) / frame.rows);
    const cv::Size ratioSize(static_cast<int>(frame.cols * ratio),
                             static_cast<int>(frame.rows * ratio));
    cv::Mat resized;
    cv::resize(frame, resized, ratioSize);

    const int padW = kInputSize - ratioSize.width;
    const int padH = kInputSize - ratioSize.height;
    const int padX = padW / 2;
    const int padY = padH / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padY, padH - padY, padX, padW - padX,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // NHWC float RGB blob [1, 192, 192, 3]
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    int sz[] = {1, kInputSize, kInputSize, 3};
    cv::Mat blob(4, sz, CV_32F, floatImg.data);

    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    if (outputs.size() < 2) return detections;

    cv::Mat boxes = outputs[0].reshape(1, outputs[0].total() / 18);
    cv::Mat scores = outputs[1].reshape(1, outputs[1].total());

    const float scale = std::max(frame.cols, frame.rows);
    const float padBiasX = padX / ratio;
    const float padBiasY = padY / ratio;

    std::vector<cv::Rect2d> candBoxes;
    std::vector<float> candScores;

    for (int i = 0; i < scores.rows && i < static_cast<int>(anchors_.size()); ++i) {
        const float rawScore = scores.at<float>(i, 0);
        const float score = 1.0f / (1.0f + std::exp(-rawScore));
        if (score < thresh) continue;

        const float* b = boxes.ptr<float>(i);
        const cv::Point2f& anchor = anchors_[i];

        const float cxDelta = b[0] / static_cast<float>(kInputSize);
        const float cyDelta = b[1] / static_cast<float>(kInputSize);
        const float wDelta  = b[2] / static_cast<float>(kInputSize);
        const float hDelta  = b[3] / static_cast<float>(kInputSize);

        const float x1 = ((cxDelta - wDelta / 2.0f + anchor.x) * scale) - padBiasX;
        const float y1 = ((cyDelta - hDelta / 2.0f + anchor.y) * scale) - padBiasY;
        const float x2 = ((cxDelta + wDelta / 2.0f + anchor.x) * scale) - padBiasX;
        const float y2 = ((cyDelta + hDelta / 2.0f + anchor.y) * scale) - padBiasY;

        candBoxes.emplace_back(x1, y1, std::max(1.0f, x2 - x1), std::max(1.0f, y2 - y1));
        candScores.push_back(score);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(candBoxes, candScores, thresh, nmsThresh_, keep);

    const cv::Rect2f bounds(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
    for (int idx : keep) {
        cv::Rect2f box = cv::Rect2f(candBoxes[idx]) & bounds;
        if (box.area() <= 0) continue;
        detections.push_back({box, classId, candScores[idx]});
    }

    return detections;
}
