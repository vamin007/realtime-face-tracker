#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

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
    // Loads the model now if it's loaded on first use, so errors show up at startup.
    virtual void load() {}
};

// Faces: SFace (OpenCV's FaceRecognizerSF), trained to tell people apart. The
// model is loaded on first use, so a run without labels never needs it. Not
// thread-safe: load() it before embedding on another thread.
class FaceEmbedder : public IEmbedder {
public:
    static constexpr const char* kModelId = "sface-2021dec";
    static constexpr float kThreshold = 0.363f;  // OpenCV's recommended cosine threshold for SFace

    explicit FaceEmbedder(std::string modelPath) : modelPath_(std::move(modelPath)) {}
    std::string modelId() const override { return kModelId; }
    float defaultThreshold() const override { return kThreshold; }
    Embedding embed(const cv::Mat& frame, const cv::Rect2f& box,
                    const std::vector<cv::Point2f>& keypoints) override;
    void load() override;

private:
    std::string modelPath_;
    cv::Ptr<cv::FaceRecognizerSF> net_;
};

// Mirrors a face box and its landmarks left-right within an image `width` wide.
// The landmarks also swap sides (the right eye becomes the left one).
void mirrorDetection(int width, cv::Rect2f& box, std::vector<cv::Point2f>& keypoints);
