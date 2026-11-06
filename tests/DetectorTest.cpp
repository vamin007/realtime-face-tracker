// Tests for the detection stage that need no camera: decodeYolo() on hand-built model outputs
// (box scaling back to frame pixels, NMS of duplicates, weak vs strong detections, reading only
// the person score, clipping), the link between YoloDetector's person id and ClassRegistry's,
// a check of the real YOLO model's class count (skipped when the model file is absent), and
// Detector's list of loaded detectors.
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "ClassRegistry.h"
#include "Detector.h"
#include "YoloDetector.h"

namespace {

// Some of YOLO's other COCO classes.
constexpr int kCup = 41, kChair = 56, kCellPhone = 67;

// A detector that finds nothing, known only by its name.
class NamedDetector : public IDetector {
public:
    explicit NamedDetector(const char* name) : name_(name) {}
    const char* name() const override { return name_; }
    bool canDetect(ClassId) const override { return false; }
    std::vector<Detection> detect(const cv::Mat&) override { return {}; }

private:
    const char* name_;
};

// Builds a fake YOLOv8 output [1, 4 + classes, boxes] from (cx, cy, w, h, class, score).
// Each box has a score for one class only (all its other scores are 0).
struct FakeBox {
    float cx, cy, w, h;
    int cls;
    float score;
};

cv::Mat fakeYoloOutput(int numClasses, const std::vector<FakeBox>& boxes) {
    const int attrs = 4 + numClasses, n = static_cast<int>(boxes.size());
    const int shape[] = {1, attrs, n};
    cv::Mat out(3, shape, CV_32F, cv::Scalar(0));
    float* p = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        p[0 * n + i] = boxes[i].cx;
        p[1 * n + i] = boxes[i].cy;
        p[2 * n + i] = boxes[i].w;
        p[3 * n + i] = boxes[i].h;
        p[(4 + boxes[i].cls) * n + i] = boxes[i].score;
    }
    return out;
}

// The first detection whose weak flag equals `weak`, or null.
const Detection* findWeak(const std::vector<Detection>& dets, bool weak) {
    const auto it =
        std::find_if(dets.begin(), dets.end(), [&](const Detection& d) { return d.weak == weak; });
    return it == dets.end() ? nullptr : &*it;
}

}  // namespace

// The model reports center + size in its input pixels; with the frame shrunk by `scale` into the
// input, dividing by the scale gives the frame's top-left corner + size.
TEST(DecodeYolo, ScalesBoxesBackToFramePixels) {
    const cv::Mat out = fakeYoloOutput(3, {{100, 50, 40, 20, kCocoPerson, 0.9f}});
    const auto dets = decodeYolo(out, 0.5f, {1280, 720}, {0.4f});
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].classId, kCocoPerson);
    EXPECT_FLOAT_EQ(dets[0].box.x, 160);  // (100 - 20) / 0.5
    EXPECT_FLOAT_EQ(dets[0].box.y, 80);   // (50 - 10) / 0.5
    EXPECT_FLOAT_EQ(dets[0].box.width, 80);
    EXPECT_FLOAT_EQ(dets[0].box.height, 40);
    EXPECT_FALSE(dets[0].weak);
}

// NMS drops the lower-scoring of two overlapping boxes; a score between the low and the main
// threshold survives but is flagged weak; below the low threshold it is dropped.
TEST(DecodeYolo, MergesDuplicatesAndMarksWeakDetections) {
    const cv::Mat out =
        fakeYoloOutput(3, {
                              {100, 100, 50, 50, kCocoPerson, 0.90f},  // kept
                              {102, 101, 50, 50, kCocoPerson, 0.80f},  // duplicate
                              {400, 300, 60, 60, kCocoPerson, 0.20f},  // weak, kept
                              {600, 300, 60, 60, kCocoPerson, 0.05f},  // below low threshold
                          });
    const auto dets = decodeYolo(out, 1.0f, {1280, 720}, {0.4f, 0.1f, 0.45f});
    ASSERT_EQ(dets.size(), 2u);
    const Detection* strong = findWeak(dets, false);
    const Detection* weak = findWeak(dets, true);
    ASSERT_NE(strong, nullptr);
    ASSERT_NE(weak, nullptr);
    EXPECT_FLOAT_EQ(strong->conf, 0.90f);
    EXPECT_FLOAT_EQ(weak->conf, 0.20f);
}

// Other classes' boxes are skipped before NMS: they aren't reported, and a more
// confident one on the same spot doesn't hide a person.
TEST(DecodeYolo, ReportsOnlyPeople) {
    const cv::Mat out = fakeYoloOutput(kCocoClasses, {
                                                         {300, 300, 100, 200, kCocoPerson, 0.6f},
                                                         {302, 300, 100, 200, kChair, 0.9f},
                                                         {800, 300, 60, 100, kCellPhone, 0.95f},
                                                         {1000, 300, 60, 60, kCup, 0.9f},
                                                     });
    const auto dets = decodeYolo(out, 1.0f, {1280, 720}, {0.4f});
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].classId, kCocoPerson);
    EXPECT_FLOAT_EQ(dets[0].conf, 0.6f);
    EXPECT_FLOAT_EQ(dets[0].box.x, 250);
}

// A box's person score counts on its own, even where another class scores higher.
TEST(DecodeYolo, APersonScoreCountsWhateverElseTheBoxScores) {
    const int boxes = 1;
    cv::Mat out = fakeYoloOutput(kCocoClasses, {{300, 300, 100, 200, kCocoPerson, 0.5f}});
    out.ptr<float>()[(4 + kChair) * boxes + 0] = 0.9f;  // the same box, as a chair
    const auto dets = decodeYolo(out, 1.0f, {1280, 720}, {0.4f});
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].classId, kCocoPerson);
    EXPECT_FLOAT_EQ(dets[0].conf, 0.5f);
}

// A box hanging over the frame edge is cut to the frame (here at the top-left corner).
TEST(DecodeYolo, ClipsBoxesToTheFrame) {
    const cv::Mat out = fakeYoloOutput(1, {{5, 5, 40, 40, 0, 0.9f}});
    const auto dets = decodeYolo(out, 1.0f, {100, 100}, {0.4f});
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].box.x, 0);
    EXPECT_FLOAT_EQ(dets[0].box.y, 0);
    EXPECT_FLOAT_EQ(dets[0].box.width, 25);
}

// People are reported under the registry's person id.
TEST(YoloDetector, PeopleGetTheRegistrysPersonId) {
    EXPECT_EQ(ClassRegistry().person(), kCocoPerson);
}

// The real model, if it's there: it has COCO's 80 classes, is asked only for
// people, and a model with another number of classes is refused.
TEST(YoloDetector, TheModelHasTheCocoClasses) {
    std::string path = PROJECT_DIR "/models/yolov8n_384x640.onnx";
    if (!std::filesystem::exists(path)) path = PROJECT_DIR "/models/yolov8n.onnx";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << path;
    const std::string cache = PROJECT_DIR "/.cache/coreml";
    const cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar::all(114));
    YoloDetector yolo(path, kCocoClasses, {0.4f}, Backend::Cpu, cache);
    EXPECT_TRUE(yolo.canDetect(kCocoPerson));
    EXPECT_FALSE(yolo.canDetect(kCup));
    EXPECT_NO_THROW(yolo.detect(frame));
    YoloDetector other(path, 3, {0.4f}, Backend::Cpu, cache);
    EXPECT_THROW(other.detect(frame), std::runtime_error);
}

// main() uses the names to see whether the face detector loaded; they come out in the order the
// detectors were added.
TEST(Detector, NamesTheDetectorsThatLoaded) {
    Detector detector;
    EXPECT_TRUE(detector.names().empty());
    detector.add(std::make_unique<NamedDetector>("face"));
    detector.add(std::make_unique<NamedDetector>("yolo"));
    EXPECT_EQ(detector.names(), (std::vector<std::string>{"face", "yolo"}));
}
