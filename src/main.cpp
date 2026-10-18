#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Camera.h"
#include "ClassRegistry.h"
#include "Detector.h"
#include "FaceDetector.h"
#include "HandDetector.h"
#include "Logger.h"
#include "Overlay.h"
#include "Tracker.h"
#include "YoloDetector.h"

namespace {

const std::string kModels = std::string(PROJECT_DIR) + "/models/";

struct ModelPath {
    std::string path;
    bool explicitlySet = false;  // given with a flag: failing to load it is an error
};

struct Options {
    std::string classes;  // comma-separated, empty = all
    float confYolo = 0.4f;
    float confFace = 0.6f;
    float confHand = 0.6f;
    int camera = 0;
    std::string video;  // play a file instead of the camera
    ModelPath yoloModel{kModels + "yolov8n.onnx"};
    ModelPath faceModel{kModels + "face_detection_yunet_2023mar.onnx"};
    ModelPath handModel{kModels + "palm_detection_mediapipe_2023feb.onnx"};
    std::string names = kModels + "coco.names";
    std::string logDir = std::string(PROJECT_DIR) + "/logs";
    bool log = true;
    bool mirror = true;
    bool faces = true;
    bool hands = true;
    bool showTrail = false;
    bool window = true;
    bool listClasses = false;
    // Developer options
    long long maxFrames = 0;     // stop after this many frames; 0 = no limit
    std::string dumpDetections;  // CSV of every raw detection per frame
    std::string saveFrames;      // directory for annotated snapshots
    int saveEvery = 30;
};

void printUsage() {
    std::cout
        << "Usage: tracker [options]\n"
           "  --classes a,b,c     only track these classes (e.g. person,hand,face,cup)\n"
           "  --list-classes      print the classes it knows and exit\n"
           "  --conf N            confidence threshold for all detectors, 0..1\n"
           "  --conf-yolo N       objects (default 0.4)\n"
           "  --conf-face N       faces (default 0.6)\n"
           "  --conf-hand N       hands (default 0.6)\n"
           "  --no-faces          turn off face detection\n"
           "  --no-hands          turn off hand detection\n"
           "  --camera N          camera index (default 0)\n"
           "  --video PATH        read a video file instead of the camera\n"
           "  --no-window         don't show a window (video files only)\n"
           "  --no-mirror         don't flip the camera image horizontally\n"
           "  --show-trail        draw a motion trail behind each track\n"
           "  --no-log            don't write a CSV log\n"
           "  --model PATH        YOLO model (default models/yolov8n.onnx)\n"
           "  --face-model PATH   face model (default models/face_detection_yunet_2023mar.onnx)\n"
           "  --hand-model PATH   hand model (default "
           "models/palm_detection_mediapipe_2023feb.onnx)\n"
           "Developer options:\n"
           "  --max-frames N            stop after N frames\n"
           "  --dump-detections PATH    write every raw detection to a CSV\n"
           "  --save-frames DIR         save annotated frames as JPEGs\n"
           "  --save-every N            with --save-frames: every Nth frame (default 30)\n"
           "Keys: Esc or q to quit.\n";
}

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

float parseFloat(const std::string& flag, const std::string& text, float lo, float hi) {
    size_t used = 0;
    float value = 0;
    try {
        value = std::stof(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size() || value < lo || value > hi) {
        std::ostringstream msg;
        msg << flag << " expects a number from " << lo << " to " << hi << ", got '" << text << "'";
        throw std::runtime_error(msg.str());
    }
    return value;
}

long long parseInt(const std::string& flag, const std::string& text, long long lo, long long hi) {
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size() || value < lo || value > hi) {
        std::ostringstream msg;
        msg << flag << " expects a whole number from " << lo << " to " << hi << ", got '" << text
            << "'";
        throw std::runtime_error(msg.str());
    }
    return value;
}

// Returns false when the program should exit right away (e.g. after --help).
bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--classes") opt.classes = next();
        else if (arg == "--list-classes") opt.listClasses = true;
        else if (arg == "--conf")
            opt.confYolo = opt.confFace = opt.confHand = parseFloat(arg, next(), 0, 1);
        else if (arg == "--conf-yolo") opt.confYolo = parseFloat(arg, next(), 0, 1);
        else if (arg == "--conf-face") opt.confFace = parseFloat(arg, next(), 0, 1);
        else if (arg == "--conf-hand") opt.confHand = parseFloat(arg, next(), 0, 1);
        else if (arg == "--no-faces") opt.faces = false;
        else if (arg == "--no-hands") opt.hands = false;
        else if (arg == "--camera") opt.camera = static_cast<int>(parseInt(arg, next(), 0, 99));
        else if (arg == "--video") opt.video = next();
        else if (arg == "--no-window") opt.window = false;
        else if (arg == "--no-mirror") opt.mirror = false;
        else if (arg == "--show-trail") opt.showTrail = true;
        else if (arg == "--no-log") opt.log = false;
        else if (arg == "--model") opt.yoloModel = {next(), true};
        else if (arg == "--face-model") opt.faceModel = {next(), true};
        else if (arg == "--hand-model") opt.handModel = {next(), true};
        else if (arg == "--max-frames") opt.maxFrames = parseInt(arg, next(), 1, 100000000);
        else if (arg == "--dump-detections") opt.dumpDetections = next();
        else if (arg == "--save-frames") opt.saveFrames = next();
        else if (arg == "--save-every")
            opt.saveEvery = static_cast<int>(parseInt(arg, next(), 1, 100000));
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else throw std::runtime_error("Unknown option: " + arg + " (see --help)");
    }
    if (!opt.window && opt.video.empty() && opt.maxFrames == 0)
        throw std::runtime_error(
            "--no-window needs --video or --max-frames (nothing could stop it)");
    return true;
}

std::set<ClassId> resolveClasses(const std::string& csv, const ClassRegistry& classes) {
    std::set<ClassId> ids;
    std::stringstream ss(csv);
    for (std::string name; std::getline(ss, name, ',');) {
        name = trim(name);
        if (name.empty()) continue;
        const ClassId id = classes.find(name);
        if (id < 0) throw std::runtime_error("Unknown class '" + name + "' (see --list-classes)");
        ids.insert(id);
    }
    return ids;
}

// Adds one detector. A model given with a flag must load; a default one that
// fails is skipped with a warning so the other detectors still run.
template <typename Make>
void addDetector(Detector& detector, const ModelPath& model, Make make) {
    try {
        detector.add(make());
    } catch (const std::exception& e) {
        if (model.explicitlySet) throw;
        std::cerr << "Warning: " << e.what() << " (detector disabled)\n";
    }
}

void dumpDetections(std::ofstream& out, long long frame, double tMs,
                    const std::vector<Detection>& detections, const ClassRegistry& classes) {
    for (const Detection& d : detections) {
        char line[160];
        std::snprintf(line, sizeof(line), "%lld,%.1f,%s,%.4f,%d,%.1f,%.1f,%.1f,%.1f,", frame, tMs,
                      classes.name(d.classId).c_str(), d.conf, d.weak ? 1 : 0, d.box.x, d.box.y,
                      d.box.width, d.box.height);
        out << line;
        for (size_t k = 0; k < d.keypoints.size(); ++k)
            out << (k ? " " : "") << d.keypoints[k].x << ' ' << d.keypoints[k].y;
        out << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) try {
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 0;

    ClassRegistry classes = ClassRegistry::fromFile(opt.names);
    if (opt.listClasses) {
        for (ClassId id = 0; id < classes.size(); ++id) std::cout << classes.name(id) << '\n';
        return 0;
    }
    const std::set<ClassId> allowed = resolveClasses(opt.classes, classes);

    Detector detector;
    addDetector(detector, opt.yoloModel, [&] {
        return std::make_unique<YoloDetector>(opt.yoloModel.path, classes.cocoCount(),
                                              DetectorThresholds{opt.confYolo});
    });
    if (opt.faces)
        addDetector(detector, opt.faceModel, [&] {
            return std::make_unique<FaceDetector>(opt.faceModel.path, classes.face(),
                                                  DetectorThresholds{opt.confFace, 0.1f, 0.3f});
        });
    if (opt.hands)
        addDetector(detector, opt.handModel, [&] {
            return std::make_unique<HandDetector>(opt.handModel.path, classes.hand(),
                                                  DetectorThresholds{opt.confHand, 0.1f, 0.3f});
        });
    if (detector.empty()) throw std::runtime_error("No detector could be loaded");

    Camera camera;
    const bool opened =
        opt.video.empty() ? camera.openDevice(opt.camera, 1280, 720) : camera.openFile(opt.video);
    if (!opened) {
        std::cerr << "Could not open " << (opt.video.empty() ? "camera" : opt.video) << ".\n"
                  << "If this is the camera, allow access for your terminal app in\n"
                  << "System Settings > Privacy & Security > Camera, then try again.\n";
        return 1;
    }
    const double videoFps = camera.fps();

    std::unique_ptr<Logger> logger;
    if (opt.log) {
        logger = std::make_unique<Logger>(opt.logDir);
        std::cout << "Logging to " << logger->path() << '\n';
    }
    std::ofstream dump;
    if (!opt.dumpDetections.empty()) {
        dump.open(opt.dumpDetections);
        if (!dump) throw std::runtime_error("Failed to open " + opt.dumpDetections);
        dump << "frame,t_ms,class,conf,weak,x,y,w,h,keypoints\n";
    }
    if (!opt.saveFrames.empty()) std::filesystem::create_directories(opt.saveFrames);

    Tracker tracker;
    const std::string window = "Object Tracker";
    if (opt.window) cv::namedWindow(window, cv::WINDOW_NORMAL);

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto lastFrame = start;
    double fps = 0.0;
    long long frameIndex = 0;
    cv::Mat frame;

    while ((opt.maxFrames == 0 || frameIndex < opt.maxFrames) && camera.read(frame)) {
        if (opt.mirror && camera.isLive()) cv::flip(frame, frame, 1);
        // Frame time: wall clock for the camera, the frame's position for a video.
        const double tMs =
            camera.isLive()
                ? std::chrono::duration<double, std::milli>(Clock::now() - start).count()
                : frameIndex * 1000.0 / videoFps;

        const auto t0 = Clock::now();
        const auto detections = detector.detect(frame, allowed);
        const double inferenceMs =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (dump.is_open()) dumpDetections(dump, frameIndex, tMs, detections, classes);

        tracker.update(detections);
        if (logger) logger->log(frameIndex, tracker.tracks(), classes);

        // Smoothed FPS so the number doesn't flicker.
        const auto now = Clock::now();
        const double instant = 1.0 / std::chrono::duration<double>(now - lastFrame).count();
        lastFrame = now;
        fps = fps == 0.0 ? instant : 0.9 * fps + 0.1 * instant;

        int visible = 0;
        for (const Track& t : tracker.tracks()) visible += isVisible(t);

        drawTracks(frame, tracker.tracks(), classes, opt.showTrail);
        drawHud(frame, {fps, inferenceMs, visible});
        if (!opt.saveFrames.empty() && frameIndex % opt.saveEvery == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "/frame_%06lld.jpg", frameIndex);
            cv::imwrite(opt.saveFrames + name, frame);
        }
        if (opt.window) {
            cv::imshow(window, frame);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q') break;
        }
        ++frameIndex;
    }

    if (opt.window) cv::destroyAllWindows();
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("Processed %lld frames in %.1f s (%.1f FPS)\n", frameIndex, seconds,
                seconds > 0 ? frameIndex / seconds : 0.0);
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
