// Saves frames of a video as JPEG photos, e.g. to try face labels with
// photos taken from the reference clips instead of personal pictures.
// A developer tool: it is built as its own executable (grab_frames), isn't part of the tracker,
// and only needs the face detector.
//
// Usage: grab_frames VIDEO OUTDIR [--every S] [--start S] [--end S] [--count N]
//                    [--crop face] [--prefix NAME]
//   --every S     one frame every S seconds (default 2)
//   --start S     from S seconds into the video (default 0)
//   --end S       until S seconds (default: the end)
//   --count N     at most N photos
//   --crop face   save just the most prominent face with a margin around it,
//                 like a portrait, skipping frames without one
//   --prefix NAME file names: NAME_000.jpg, NAME_001.jpg, ... (default: frame)

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "FaceDetector.h"

namespace fs = std::filesystem;

namespace {

constexpr float kCropMargin = 0.15f;  // of the box size, on each side
constexpr ClassId kFace = 0;          // the only class looked for, so any id does

void usage() {
    std::cout << "Usage: grab_frames VIDEO OUTDIR [--every S] [--start S] [--end S] [--count N]\n"
                 "                   [--crop face] [--prefix NAME]\n";
}

}  // namespace

int main(int argc, char** argv) try {
    // Quiet OpenCV's own logging so the output is just the photos saved.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (argc < 3) {
        usage();
        return 1;
    }
    const std::string video = argv[1], outDir = argv[2];
    double every = 2, start = 0, end = 1e9;
    long count = 1000000;
    std::string cropClass, prefix = "frame";
    // Options come as "--name value" pairs after the two positional arguments.
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

    // Labels are faces, so faces are the only thing worth cropping.
    std::unique_ptr<FaceDetector> faces;
    if (!cropClass.empty()) {
        if (cropClass != "face")
            throw std::runtime_error("--crop takes face (labels are faces), not '" + cropClass +
                                     "'");
        // The tracker's YuNet face model, with thresholds conf 0.6, low 0.1 and NMS 0.3
        // (non-maximum suppression: boxes overlapping more than that are duplicates). Frames
        // wider than 640 px are scaled down for detection.
        faces =
            std::make_unique<FaceDetector>(PROJECT_DIR "/models/face_detection_yunet_2023mar.onnx",
                                           kFace, DetectorThresholds{0.6f, 0.1f, 0.3f}, 640);
    }

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) throw std::runtime_error("Can't open " + video);
    fs::create_directories(outDir);
    // A frame's time is its index over the FPS (30 if the file doesn't say).
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30;
    // `next`: the video time (s) at which the next photo is due.
    double next = start;
    long saved = 0, searched = 0;  // searched: frames a face was looked for in
    cv::Mat frame;
    for (long index = 0; saved < count && cap.read(frame); ++index) {
        const double t = index / fps;
        if (t > end) break;
        // Not due yet. (The tiny epsilon absorbs floating-point error in index / fps.)
        if (t + 1e-9 < next) continue;
        cv::Mat photo = frame;
        if (faces) {
            ++searched;
            // The most prominent confident face (confidence times area); weak detections are
            // skipped.
            const Detection* best = nullptr;
            const auto detections = faces->detect(frame);
            for (const Detection& d : detections)
                if (!d.weak && (!best || d.conf * d.box.area() > best->conf * best->box.area()))
                    best = &d;
            if (!best) continue;  // try the next frame
            const cv::Rect2f& b = best->box;
            // The face box grown by kCropMargin on every side, clipped to the frame: a
            // portrait-style photo.
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
    if (faces && saved == 0)
        std::cerr << "Warning: no confident face in the " << searched << " frame(s) searched\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
