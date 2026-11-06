#include "Options.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

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

}  // namespace

Options defaultOptions(const std::string& projectDir) {
    const std::string models = projectDir + "/models/";
    Options opt;
    // 640x384 fits a 16:9 frame with little padding (main falls back to the
    // square model if this one hasn't been exported).
    opt.yoloModel = {models + "yolov8n_384x640.onnx"};
    opt.faceModel = {models + "face_detection_yunet_2023mar.onnx"};
    opt.faceIdModel = {models + "face_recognition_sface_2021dec.onnx"};
    opt.logDir = projectDir + "/logs";
    opt.cacheDir = projectDir + "/.cache/coreml";
    opt.labelsDir = projectDir + "/labels";
    return opt;
}

void printUsage() {
    std::cout
        << "Usage: tracker [options]\n"
           "  --classes a,b,c       only track these: person, face, labels (e.g. face)\n"
           "  --list-classes        print the classes it knows and exit\n"
           "  --camera N            camera index (default 0)\n"
           "  --video PATH          read a video file instead of the camera\n"
           "  --realtime            play the video at its own speed, like a camera\n"
           "  --no-window           don't show a window (video files only)\n"
           "  --no-mirror           don't flip the camera image horizontally\n"
           "  --show-trail          draw a motion trail behind each track\n"
           "  --debug               show confidence, laser and recognition details and timings\n"
           "  --record PATH         save the raw camera feed as a video (e.g. clips/new.mov)\n"
           "  --no-log              don't write a CSV log\n"
           "\nDetection:\n"
           "  --conf N              confidence threshold for all detectors, 0..1\n"
           "  --conf-yolo N         people (default 0.4)\n"
           "  --conf-face N         faces (default 0.6)\n"
           "  --no-faces            turn off face detection\n"
           "  --backend B           coreml (GPU / Neural Engine, default) or cpu\n"
           "\nTARGET (a face a red laser dot is on):\n"
           "  --no-targeting        turn it off\n"
           "  --laser-redness N     how much more a dot's red must rise above the skin's than its\n"
           "                        green and blue, 0..255 (default 35)\n"
           "\nCustom labels (a specific person, recognized by their face, taught from photos):\n"
           "  Each label is a folder of photos, labels/NAME/; new and changed photos are\n"
           "  looked at when the tracker starts.\n"
           "  --enroll NAME PHOTO...  teach a person's face (e.g. --enroll alex a.jpg b.jpg "
           "c.jpg)\n"
           "  --add-photos NAME PHOTO...  add photos to a label\n"
           "  --sync-labels         look at new and changed photos now, check the labels, exit\n"
           "  --list-labels         list the labels and exit\n"
           "  --rename-label OLD NEW  rename a label\n"
           "  --remove-label NAME   move a label to labels/.removed/ (its photos are kept)\n"
           "  --no-labels           don't recognize labels\n"
           "  --labels-dir DIR      where labels are kept (default labels/)\n"
           "\nModels:\n"
           "  --model PATH          YOLO, for people (default models/yolov8n_384x640.onnx)\n"
           "  --face-model PATH     face detector (default "
           "models/face_detection_yunet_2023mar.onnx)\n"
           "\nDeveloper options:\n"
           "  --max-frames N            stop after N frames\n"
           "  --dump-detections PATH    write every raw detection to a CSV\n"
           "  --dump-laser PATH         write each face's laser search result per frame to a CSV\n"
           "  --dump-labels PATH        write each recognition result (label scores) to a CSV\n"
           "  --save-frames DIR         save annotated frames as JPEGs\n"
           "  --save-every N            with --save-frames: every Nth frame (default 30)\n"
           "  --sequential              run the detectors one after another\n"
           "  --face-width N            width faces are detected at (default 640, 0 = full)\n"
           "\nKeys: p pause/resume, t trails on/off, d debug view on/off, h help, q or Esc quit.\n";
}

bool parseArgs(const std::vector<std::string>& args, Options& opt) {
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) throw std::runtime_error("Missing value for " + arg);
            return args[++i];
        };
        if (arg == "--classes") opt.classes = next();
        else if (arg == "--list-classes") opt.listClasses = true;
        else if (arg == "--camera") opt.camera = static_cast<int>(parseInt(arg, next(), 0, 99));
        else if (arg == "--video") opt.video = next();
        else if (arg == "--realtime") opt.realtime = true;
        else if (arg == "--no-window") opt.window = false;
        else if (arg == "--no-mirror") opt.mirror = false;
        else if (arg == "--show-trail") opt.showTrail = true;
        else if (arg == "--debug") opt.debug = true;
        else if (arg == "--record") opt.record = next();
        else if (arg == "--no-log") opt.log = false;
        else if (arg == "--conf") opt.confYolo = opt.confFace = parseFloat(arg, next(), 0, 1);
        else if (arg == "--conf-yolo") opt.confYolo = parseFloat(arg, next(), 0, 1);
        else if (arg == "--conf-face") opt.confFace = parseFloat(arg, next(), 0, 1);
        else if (arg == "--no-faces") opt.faces = false;
        else if (arg == "--backend") {
            const std::string b = next();
            if (b == "coreml") opt.backend = Backend::CoreML;
            else if (b == "cpu") opt.backend = Backend::Cpu;
            else throw std::runtime_error("--backend expects coreml or cpu, got '" + b + "'");
        } else if (arg == "--no-targeting") opt.targeting = false;
        else if (arg == "--laser-redness") opt.target.minRedGain = parseFloat(arg, next(), 0, 255);
        else if (arg == "--enroll") opt.enrollName = next();
        else if (arg == "--add-photos") {
            opt.enrollName = next();
            opt.addPhotos = true;
        } else if (arg == "--sync-labels") opt.syncLabels = true;
        else if (arg == "--list-labels") opt.listLabels = true;
        else if (arg == "--remove-label") opt.removeLabel = next();
        else if (arg == "--rename-label") {
            opt.renameLabel = next();
            opt.renameTo = next();
        } else if (arg == "--no-labels") opt.labels = false;
        else if (arg == "--labels-dir") opt.labelsDir = next();
        else if (arg == "--model") opt.yoloModel = {next(), true};
        else if (arg == "--face-model") opt.faceModel = {next(), true};
        else if (arg == "--max-frames") opt.maxFrames = parseInt(arg, next(), 1, 100000000);
        else if (arg == "--dump-detections") opt.dumpDetections = next();
        else if (arg == "--dump-laser") opt.dumpLaser = next();
        else if (arg == "--dump-labels") opt.dumpLabels = next();
        else if (arg == "--save-frames") opt.saveFrames = next();
        else if (arg == "--save-every")
            opt.saveEvery = static_cast<int>(parseInt(arg, next(), 1, 100000));
        else if (arg == "--sequential") opt.parallel = false;
        else if (arg == "--face-width")
            opt.faceWidth = static_cast<int>(parseInt(arg, next(), 0, 10000));
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else if (!arg.empty() && arg[0] != '-') {
            opt.enrollPhotos.push_back(arg);  // photos for --enroll
        } else {
            throw std::runtime_error("Unknown option: " + arg + " (see --help)");
        }
    }
    if (!opt.enrollPhotos.empty() && opt.enrollName.empty())
        throw std::runtime_error("Unexpected argument '" + opt.enrollPhotos.front() +
                                 "' (photos go with --enroll NAME)");
    if (!opt.enrollName.empty() && opt.enrollPhotos.empty())
        throw std::runtime_error((opt.addPhotos ? "--add-photos " : "--enroll ") + opt.enrollName +
                                 " needs at least one photo");
    if (!opt.window && opt.video.empty() && opt.maxFrames == 0)
        throw std::runtime_error(
            "--no-window needs --video or --max-frames (nothing could stop it)");
    return true;
}
