#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

#include "OrtModel.h"

// A neural network's summary of an image crop, normalized to length 1: crops
// of the same item point in nearly the same direction, so the dot product of
// two embeddings (their cosine similarity) says how alike they are.
using Embedding = std::vector<float>;

float similarity(const Embedding& a, const Embedding& b);

class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    // Names the model; saved labels remember it, so a different model means re-embedding.
    virtual std::string modelId() const = 0;
    // Similarity from which two crops count as the same item with this model.
    virtual float defaultThreshold() const = 0;
    // Embedding of the object in `box` (frame pixels); faces use their 5 landmarks
    // to straighten the face first. Empty if the box is unusable.
    virtual Embedding embed(const cv::Mat& frame, const cv::Rect2f& box,
                            const std::vector<cv::Point2f>& keypoints) = 0;
};

// Faces: SFace (OpenCV's FaceRecognizerSF), trained to tell people apart.
class FaceEmbedder : public IEmbedder {
public:
    explicit FaceEmbedder(const std::string& modelPath);
    std::string modelId() const override { return "sface-2021dec"; }
    // OpenCV's recommended cosine threshold for SFace.
    float defaultThreshold() const override { return 0.363f; }
    Embedding embed(const cv::Mat& frame, const cv::Rect2f& box,
                    const std::vector<cv::Point2f>& keypoints) override;

private:
    cv::Ptr<cv::FaceRecognizerSF> net_;
};

// Objects: DINOv2-small, averaged over its image patches, which is good at
// telling one specific thing (this mug) from other things. On the flashlight
// clip the same phone scored 0.77 typically (0.57 at the 10th percentile) and
// other objects 0.29 (0.57 at the 90th), hence the 0.6 threshold.
class ObjectEmbedder : public IEmbedder {
public:
    ObjectEmbedder(const std::string& modelPath, Backend backend, const std::string& cacheDir);
    std::string modelId() const override { return "dinov2-small-patchmean"; }
    float defaultThreshold() const override { return threshold_; }
    Embedding embed(const cv::Mat& frame, const cv::Rect2f& box,
                    const std::vector<cv::Point2f>& keypoints) override;
    std::string info() const { return model_->backendInfo(); }

private:
    std::unique_ptr<OrtModel> model_;
    float threshold_ = 0.6f;
};

// Mirrors a box and its keypoints left-right within an image `width` wide. Face
// landmarks also swap sides (the right eye becomes the left one).
void mirrorDetection(int width, cv::Rect2f& box, std::vector<cv::Point2f>& keypoints, bool isFace);
