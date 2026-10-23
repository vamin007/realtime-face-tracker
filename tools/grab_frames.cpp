// Saves frames of a video as JPEG photos, e.g. to try custom labels with
// photos taken from the reference clips instead of personal pictures.
//
// Usage: grab_frames VIDEO OUTDIR [--every S] [--start S] [--end S] [--count N]
//                    [--crop CLASS] [--prefix NAME]
//   --every S     one frame every S seconds (default 2)
//   --start S     from S seconds into the video (default 0)
//   --end S       until S seconds (default: the end)
//   --count N     at most N photos
//   --crop CLASS  save just the most prominent CLASS (a COCO class such as
//                 "cell phone", face or hand) with a margin around it,
//                 skipping frames without one
//   --prefix NAME file names: NAME_000.jpg, NAME_001.jpg, ... (default: frame)

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "ClassRegistry.h"
#include "Detector.h"
#include "FaceDetector.h"
#include "HandDetector.h"
#include "YoloDetector.h"

namespace fs = std::filesystem;

namespace {

constexpr float kCropMargin = 0.15f;  // of the box size, on each side

void usage() {
    std::cout << "Usage: grab_frames VIDEO OUTDIR [--every S] [--start S] [--end S] [--count N]\n"
                 "                   [--crop CLASS] [--prefix NAME]\n";
}

}  // namespace

int main(int argc, char** argv) try {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (argc < 3) {
        usage();
        return 1;
    }
    const std::string video = argv[1], outDir = argv[2];
    double every = 2, start = 0, end = 1e9;
    long count = 1000000;
    std::string cropClass, prefix = "frame";
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--every") every = std::stod(value);
        else if (arg == "--start") start = std::stod(value);
        else if (arg == "--end") end = std::stod(value);
        else if (arg == "--count") count = std::stol(value);
        else if (arg == "--crop") cropClass = value;
        else if (arg == "--prefix") prefix = value;
        else throw std::runtime_error("Unknown option " + arg);
    }
    if (every <= 0) throw std::runtime_error("--every must be above 0");

    const std::string models = PROJECT_DIR "/models/";
    const ClassRegistry classes = ClassRegistry::fromFile(models + "coco.names");
    Detector detector;
    ClassId crop = -1;
    if (!cropClass.empty()) {
        crop = classes.find(cropClass);
        if (crop < 0 || classes.kind(crop) == ClassKind::Custom)
            throw std::runtime_error("Unknown class '" + cropClass +
                                     "' (crop a class from models/coco.names, face or hand)");
        if (crop == classes.face())
            detector.add(std::make_unique<FaceDetector>(
                models + "face_detection_yunet_2023mar.onnx", classes.face(),
                DetectorThresholds{0.6f, 0.1f, 0.3f}, 640));
        else if (crop == classes.hand())
            detector.add(std::make_unique<HandDetector>(
                models + "palm_detection_mediapipe_2023feb.onnx", classes.hand(),
                DetectorThresholds{0.6f, 0.1f, 0.3f}, Backend::CoreML,
                PROJECT_DIR "/.cache/coreml"));
        else
            detector.add(std::make_unique<YoloDetector>(
                models + "yolov8n_384x640.onnx", classes.cocoCount(), DetectorThresholds{0.4f},
                Backend::CoreML, PROJECT_DIR "/.cache/coreml"));
    }

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) throw std::runtime_error("Can't open " + video);
    fs::create_directories(outDir);
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30;
    double next = start;
    long saved = 0, searched = 0;  // searched: frames the crop class was looked for in
    cv::Mat frame;
    for (long index = 0; saved < count && cap.read(frame); ++index) {
        const double t = index / fps;
        if (t > end) break;
        if (t + 1e-9 < next) continue;
        cv::Mat photo = frame;
        if (crop >= 0) {
            ++searched;
            const Detection* best = nullptr;
            const auto detections = detector.detect(frame, {crop});
            for (const Detection& d : detections)
                if (d.classId == crop && !d.weak &&
                    (!best || d.conf * d.box.area() > best->conf * best->box.area()))
                    best = &d;
            if (!best) continue;  // try the next frame
            const cv::Rect2f& b = best->box;
            const cv::Rect area =
                cv::Rect(cv::Rect2f(b.x - b.width * kCropMargin, b.y - b.height * kCropMargin,
                                    b.width * (1 + 2 * kCropMargin),
                                    b.height * (1 + 2 * kCropMargin))) &
                cv::Rect(0, 0, frame.cols, frame.rows);
            photo = frame(area);
        }
        char name[64];
        std::snprintf(name, sizeof(name), "%s_%03ld.jpg", prefix.c_str(), saved);
        const std::string path = (fs::path(outDir) / name).string();
        if (!cv::imwrite(path, photo)) throw std::runtime_error("Can't write " + path);
        std::printf("%s  (%.1f s, %dx%d)\n", path.c_str(), t, photo.cols, photo.rows);
        ++saved;
        next = t + every;
    }
    std::printf("Saved %ld photo(s) to %s\n", saved, outDir.c_str());
    if (crop >= 0 && saved == 0)
        std::cerr << "Warning: no confident " << cropClass << " in the " << searched
                  << " frame(s) searched\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
