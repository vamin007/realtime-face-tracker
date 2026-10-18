#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
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
    std::string video;      // play a file instead of the camera
    bool realtime = false;  // play the video at its own speed, skipping frames when busy
    Backend backend = Backend::CoreML;
    // 640x384 fits a 16:9 frame with little padding; the square model is the fallback.
    ModelPath yoloModel{kModels + "yolov8n_384x640.onnx"};
    ModelPath faceModel{kModels + "face_detection_yunet_2023mar.onnx"};
    ModelPath handModel{kModels + "palm_detection_mediapipe_2023feb.onnx"};
    std::string names = kModels + "coco.names";
    std::string logDir = std::string(PROJECT_DIR) + "/logs";
    std::string cacheDir = std::string(PROJECT_DIR) + "/.cache/coreml";
    bool log = true;
    bool mirror = true;
    bool faces = true;
    bool hands = true;
    bool showTrail = false;
    bool window = true;
    bool listClasses = false;
    bool targeting = true;
    TargetParams target;
    // Developer options
    long long maxFrames = 0;     // stop after this many frames; 0 = no limit
    std::string dumpDetections;  // CSV of every raw detection per frame
    std::string saveFrames;      // directory for annotated snapshots
    int saveEvery = 30;
    std::string dumpLight;  // CSV of each face's brightness per frame
    bool parallel = true;   // run the detectors side by side
    int faceWidth = 640;    // face detection works on frames scaled down to this width
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
           "  --no-targeting      don't mark a brightly lit face as TARGET\n"
           "  --target-ratio N    a lit face is N times as bright as the room (default 1.0)\n"
           "  --target-rise N     ... and N times brighter than it normally is (default 1.35)\n"
           "  --camera N          camera index (default 0)\n"
           "  --video PATH        read a video file instead of the camera\n"
           "  --realtime          play the video at its own speed, like a camera\n"
           "  --no-window         don't show a window (video files only)\n"
           "  --no-mirror         don't flip the camera image horizontally\n"
           "  --show-trail        draw a motion trail behind each track\n"
           "  --no-log            don't write a CSV log\n"
           "  --backend B         coreml (GPU / Neural Engine, default) or cpu\n"
           "  --model PATH        YOLO model (default models/yolov8n_384x640.onnx)\n"
           "  --face-model PATH   face model (default models/face_detection_yunet_2023mar.onnx)\n"
           "  --hand-model PATH   hand model (default "
           "models/palm_detection_mediapipe_2023feb.onnx)\n"
           "Developer options:\n"
           "  --max-frames N            stop after N frames\n"
           "  --dump-detections PATH    write every raw detection to a CSV\n"
           "  --save-frames DIR         save annotated frames as JPEGs\n"
           "  --save-every N            with --save-frames: every Nth frame (default 30)\n"
           "  --sequential              run the detectors one after another\n"
           "  --face-width N            width faces are detected at (default 640, 0 = full)\n"
           "  --dump-light PATH         write each face's brightness per frame to a CSV\n"
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
        else if (arg == "--no-targeting") opt.targeting = false;
        else if (arg == "--target-ratio") opt.target.minRatio = parseFloat(arg, next(), 0.5f, 100);
        else if (arg == "--target-rise") opt.target.minRise = parseFloat(arg, next(), 1, 100);
        else if (arg == "--dump-light") opt.dumpLight = next();
        else if (arg == "--camera") opt.camera = static_cast<int>(parseInt(arg, next(), 0, 99));
        else if (arg == "--video") opt.video = next();
        else if (arg == "--realtime") opt.realtime = true;
        else if (arg == "--backend") {
            const std::string b = next();
            if (b == "coreml") opt.backend = Backend::CoreML;
            else if (b == "cpu") opt.backend = Backend::Cpu;
            else throw std::runtime_error("--backend expects coreml or cpu, got '" + b + "'");
        } else if (arg == "--no-window") opt.window = false;
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
        else if (arg == "--sequential") opt.parallel = false;
        else if (arg == "--face-width")
            opt.faceWidth = static_cast<int>(parseInt(arg, next(), 0, 10000));
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

void dumpLight(std::ofstream& out, long long frame, double tMs, const TargetSelector& targeting) {
    for (const auto& [id, f] : targeting.faces()) {
        if (f.lastUpdate * 1000.0 < tMs - 1) continue;  // not measured this frame
        char line[160];
        std::snprintf(line, sizeof(line), "%lld,%.1f,%d,%.1f,%.1f,%.3f,%.3f,%.3f,%d,%d\n", frame,
                      tMs, id, f.faceLuma, targeting.background(), f.ratio, f.normalRatio, f.rise(),
                      f.litSince >= 0 ? 1 : 0, targeting.target().trackId == id ? 1 : 0);
        out << line;
    }
}

}  // namespace

int main(int argc, char** argv) try {
    // OpenCV's DNN module prints a warning about compute targets on every launch.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 0;

    ClassRegistry classes = ClassRegistry::fromFile(opt.names);
    if (opt.listClasses) {
        for (ClassId id = 0; id < classes.size(); ++id) std::cout << classes.name(id) << '\n';
        return 0;
    }
    const std::set<ClassId> allowed = resolveClasses(opt.classes, classes);

    // Fall back to the square YOLO model if the 16:9 one hasn't been exported.
    if (!opt.yoloModel.explicitlySet && !std::filesystem::exists(opt.yoloModel.path))
        opt.yoloModel.path = kModels + "yolov8n.onnx";

    Detector detector;
    detector.setParallel(opt.parallel);
    addDetector(detector, opt.yoloModel, [&] {
        return std::make_unique<YoloDetector>(opt.yoloModel.path, classes.cocoCount(),
                                              DetectorThresholds{opt.confYolo}, opt.backend,
                                              opt.cacheDir);
    });
    if (opt.faces)
        addDetector(detector, opt.faceModel, [&] {
            return std::make_unique<FaceDetector>(opt.faceModel.path, classes.face(),
                                                  DetectorThresholds{opt.confFace, 0.1f, 0.3f},
                                                  opt.faceWidth);
        });
    if (opt.hands)
        addDetector(detector, opt.handModel, [&] {
            return std::make_unique<HandDetector>(opt.handModel.path, classes.hand(),
                                                  DetectorThresholds{opt.confHand, 0.1f, 0.3f},
                                                  opt.backend, opt.cacheDir);
        });
    if (detector.empty()) throw std::runtime_error("No detector could be loaded");
    std::cout << "Detectors (" << (opt.parallel ? "side by side" : "one after another") << "):\n";
    for (const std::string& line : detector.summary()) std::cout << "  " << line << '\n';

    Camera camera;
    const bool opened = opt.video.empty() ? camera.openDevice(opt.camera, 1280, 720)
                                          : camera.openFile(opt.video, opt.realtime);
    if (!opened) {
        std::cerr << "Could not open " << (opt.video.empty() ? "camera" : opt.video) << ".\n"
                  << "If this is the camera, allow access for your terminal app in\n"
                  << "System Settings > Privacy & Security > Camera, then try again.\n";
        return 1;
    }
    std::printf("%s: %dx%d at %.0f FPS%s\n", camera.isLive() ? "Camera" : "Video",
                camera.frameSize().width, camera.frameSize().height, camera.fps(),
                opt.realtime ? " (realtime)" : "");

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

    Tracker tracker(classes);
    tracker.setFrameSize(camera.frameSize());
    TargetSelector targeting(classes, opt.target);
    std::ofstream lightDump;
    if (!opt.dumpLight.empty()) {
        lightDump.open(opt.dumpLight);
        if (!lightDump) throw std::runtime_error("Failed to open " + opt.dumpLight);
        lightDump << "frame,t_ms,id,face_luma,background,ratio,normal,rise,lit,target\n";
    }
    const std::string window = "Object Tracker";
    if (opt.window) cv::namedWindow(window, cv::WINDOW_NORMAL);

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto lastFrame = start;
    double fps = 0.0;
    long long frameIndex = 0;
    cv::Mat frame;
    double tMs = 0;
    // Running totals for the summary at the end.
    double detectMsTotal = 0;
    std::map<std::string, double> detectorMsTotal;

    while ((opt.maxFrames == 0 || frameIndex < opt.maxFrames) && camera.read(frame, tMs)) {
        if (opt.mirror && camera.isLive()) cv::flip(frame, frame, 1);

        const auto t0 = Clock::now();
        const auto detections = detector.detect(frame, allowed);
        detectMsTotal += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        for (const auto& t : detector.timings()) detectorMsTotal[t.name] += t.ms;
        if (dump.is_open()) dumpDetections(dump, frameIndex, tMs, detections, classes);

        tracker.update(detections, tMs / 1000.0);
        // Brightness is measured here, before anything is drawn on the frame.
        if (opt.targeting) targeting.update(frame, tracker.tracks(), tMs / 1000.0);
        if (lightDump.is_open()) dumpLight(lightDump, frameIndex, tMs, targeting);
        if (logger)
            logger->log(frameIndex, tMs, tracker.tracks(), classes,
                        opt.targeting ? &targeting : nullptr);

        // Smoothed FPS so the number doesn't flicker.
        const auto now = Clock::now();
        const double instant = 1.0 / std::chrono::duration<double>(now - lastFrame).count();
        lastFrame = now;
        fps = fps == 0.0 ? instant : 0.9 * fps + 0.1 * instant;

        int visible = 0;
        for (const Track& t : tracker.tracks()) visible += isVisible(t);

        drawTracks(frame, tracker.tracks(), classes, opt.showTrail, targeting.target().trackId);
        drawTarget(frame, targeting.target());
        drawHud(frame, {fps, detector.timings(), visible});
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
    if (frameIndex > 0) {
        std::printf("Detection per frame: %.1f ms (", detectMsTotal / frameIndex);
        const char* sep = "";
        for (const auto& [name, ms] : detectorMsTotal) {
            std::printf("%s%s %.1f", sep, name.c_str(), ms / frameIndex);
            sep = ", ";
        }
        std::printf(")\n");
    }
    if (camera.dropped() > 0)
        std::printf("Skipped %lld frames that arrived while busy\n", camera.dropped());
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
