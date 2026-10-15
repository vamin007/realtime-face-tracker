#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include <sstream>
#include <string>

#include "Camera.h"
#include "Detector.h"
#include "Logger.h"
#include "Overlay.h"
#include "Tracker.h"

namespace {

struct Options {
    std::string classes;  // comma-separated, empty = all
    float conf = 0.4f;
    int camera = 0;
    std::string video;    // play a file instead of the camera
    std::string model = std::string(PROJECT_DIR) + "/models/yolov8n.onnx";
    std::string names = std::string(PROJECT_DIR) + "/models/coco.names";
    std::string faceModel = std::string(PROJECT_DIR) + "/models/face_detection_yunet_2023mar.onnx";
    std::string handModel = std::string(PROJECT_DIR) + "/models/palm_detection_mediapipe_2023feb.onnx";
    std::string logDir = std::string(PROJECT_DIR) + "/logs";
    bool log = true;
    bool mirror = true;
    bool personByFace = true;
    bool enableHands = true;
    bool showTrail = false;
};

void printUsage() {
    std::cout << "Usage: tracker [options]\n"
                 "  --classes a,b,c       only track these categories (e.g. person,hand,cup)\n"
                 "  --conf N              detection confidence threshold, 0..1 (default 0.4)\n"
                 "  --camera N            camera index (default 0)\n"
                 "  --video PATH          use a video file instead of the camera\n"
                 "  --model PATH          YOLO ONNX model (default models/yolov8n.onnx)\n"
                 "  --face-model PATH     Face detector ONNX model (default models/face_detection_yunet_2023mar.onnx)\n"
                 "  --hand-model PATH     Hand detector ONNX model (default models/palm_detection_mediapipe_2023feb.onnx)\n"
                 "  --no-person-by-face   identify person via full-body box rather than face only\n"
                 "  --person-by-face      identify person only through a face (default: on)\n"
                 "  --no-hands            disable hand detection\n"
                 "  --enable-hands        enable hand detection (default: on)\n"
                 "  --show-trail          draw motion trail dot/line behind tracks (default: off)\n"
                 "  --no-trail            hide the motion trail (default)\n"
                 "  --no-log              don't write a CSV log\n"
                 "  --no-mirror           don't flip the camera image horizontally\n"
                 "  --list-classes        print the categories the model knows and exit\n"
                 "Keys: ESC or q to quit.\n";
}

bool parseArgs(int argc, char** argv, Options& opt, bool& listClasses) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--classes") opt.classes = next();
        else if (arg == "--conf") opt.conf = std::stof(next());
        else if (arg == "--camera") opt.camera = std::stoi(next());
        else if (arg == "--video") opt.video = next();
        else if (arg == "--model") opt.model = next();
        else if (arg == "--face-model") opt.faceModel = next();
        else if (arg == "--hand-model") opt.handModel = next();
        else if (arg == "--no-person-by-face") opt.personByFace = false;
        else if (arg == "--person-by-face") opt.personByFace = true;
        else if (arg == "--no-hands") opt.enableHands = false;
        else if (arg == "--enable-hands") opt.enableHands = true;
        else if (arg == "--show-trail") opt.showTrail = true;
        else if (arg == "--no-trail") opt.showTrail = false;
        else if (arg == "--no-log") opt.log = false;
        else if (arg == "--no-mirror") opt.mirror = false;
        else if (arg == "--list-classes") listClasses = true;
        else if (arg == "--help" || arg == "-h") { printUsage(); return false; }
        else throw std::runtime_error("Unknown option: " + arg);
    }
    return true;
}

std::set<int> resolveClasses(const std::string& csv, const Detector& detector) {
    std::set<int> ids;
    std::stringstream ss(csv);
    for (std::string name; std::getline(ss, name, ',');) {
        if (name.empty()) continue;
        const int id = detector.classIdFor(name);
        if (id < 0) throw std::runtime_error("Unknown class '" + name + "' (see --list-classes)");
        ids.insert(id);
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) try {
    Options opt;
    bool listClasses = false;
    if (!parseArgs(argc, argv, opt, listClasses)) return 0;

    Detector detector(opt.model, opt.names, 640, opt.faceModel, opt.handModel);
    detector.setPersonByFace(opt.personByFace);
    detector.setEnableHands(opt.enableHands);

    if (listClasses) {
        for (const auto& n : detector.classNames()) std::cout << n << '\n';
        return 0;
    }
    const std::set<int> allowed = resolveClasses(opt.classes, detector);

    Camera camera;
    const bool opened = opt.video.empty() ? camera.openDevice(opt.camera, 1280, 720)
                                          : camera.openFile(opt.video);
    if (!opened) {
        std::cerr << "Could not open " << (opt.video.empty() ? "camera" : opt.video) << ".\n"
                  << "If this is the camera, allow access for your terminal app in\n"
                  << "System Settings > Privacy & Security > Camera, then try again.\n";
        return 1;
    }

    std::unique_ptr<Logger> logger;
    if (opt.log) {
        logger = std::make_unique<Logger>(opt.logDir);
        std::cout << "Logging to " << logger->path() << '\n';
    }

    Tracker tracker;
    const std::string window = "Object Tracker";
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    using Clock = std::chrono::steady_clock;
    auto lastFrame = Clock::now();
    double fps = 0.0;
    long long frameIndex = 0;
    cv::Mat frame;

    while (camera.read(frame)) {
        if (opt.mirror && camera.isLive()) cv::flip(frame, frame, 1);

        const auto t0 = Clock::now();
        const auto detections = detector.detect(frame, opt.conf, 0.45f, allowed);
        const double inferenceMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        tracker.update(detections);
        if (logger) logger->log(frameIndex, tracker.tracks(), detector.classNames());

        // Smoothed FPS so the number doesn't flicker.
        const auto now = Clock::now();
        const double instant = 1.0 / std::chrono::duration<double>(now - lastFrame).count();
        lastFrame = now;
        fps = fps == 0.0 ? instant : 0.9 * fps + 0.1 * instant;

        int visible = 0;
        for (const Track& t : tracker.tracks()) visible += isVisible(t);

        drawTracks(frame, tracker.tracks(), detector.classNames(), opt.showTrail);
        drawHud(frame, {fps, inferenceMs, visible});
        cv::imshow(window, frame);

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
        ++frameIndex;
    }

    cv::destroyAllWindows();
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
