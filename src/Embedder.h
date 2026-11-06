// Face embeddings: turning a face in a camera frame or a photo into a vector of numbers that can
// be compared. IEmbedder is the interface (FaceEmbedder wraps OpenCV's SFace network; tests
// substitute fakes), similarity() compares two embeddings, and mirrorDetection() flips a face box
// and its landmarks so the mirrored copy of a photo or crop can be embedded too.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

// A neural network's summary of an image crop, normalized to length 1: crops
// of the same item point in nearly the same direction, so the dot product of
// two embeddings (their cosine similarity) says how alike they are. SFace gives 128 numbers.
// (Cosine similarity is the cosine of the angle between two vectors: 1 for the same direction,
// 0 for unrelated ones, -1 for opposite.)
using Embedding = std::vector<float>;

// Cosine similarity of two embeddings: their dot product, as both have length 1. Returns 0 for
// embeddings of different sizes.
float similarity(const Embedding& a, const Embedding& b);

// What labels, syncing and recognition need from an embedding model, so none of them depends on
// SFace itself and tests can use fakes.
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

// Faces: SFace (OpenCV's FaceRecognizerSF), trained to tell people apart; it turns a
// 112 x 112 aligned face into 128 numbers. The model is loaded on first use, so a run
// without labels never needs it. Not thread-safe: load() it before embedding on another
// thread.
class FaceEmbedder : public IEmbedder {
public:
    // Saved in every label cache: a different id means the saved embeddings can't be compared with
    // new ones, and every photo is embedded again (see planSync).
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
// The landmarks also swap sides (the right eye becomes the left one). Used to embed the
// left-right flipped copy of a photo (a sync) or of a crop (a photo taken while recognized).
void mirrorDetection(int width, cv::Rect2f& box, std::vector<cv::Point2f>& keypoints);
