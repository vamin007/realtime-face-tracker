// The entry point and the per-frame loop that wires the whole pipeline together:
//
//   Camera            a capture thread that keeps only the newest frame
//   -> Detector       YOLO (people, GPU / Neural Engine) and YuNet (faces, CPU), side by side
//   -> Tracker        one Kalman filter per track, Hungarian matching of detections to tracks
//   -> TargetSelector marks the face a red laser dot is on as the TARGET
//   -> Recognizer     names faces taught from photos (embeddings computed on a worker thread)
//   -> Overlay        draws boxes and tags in the window; optionally records or saves frames
//      Logger         writes one CSV row per visible track per frame
//
// main() first runs any one-off command (--list-labels, --enroll, --sync-labels, ...), which
// exits without tracking. Otherwise it builds the pipeline, runs the loop until the video ends,
// the user quits (q / Esc) or presses Ctrl-C, and prints a per-stage latency summary.
#include <atomic>
#include <chrono>
#include <csignal>
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
#include "Enrollment.h"
#include "FaceDetector.h"
#include "Labels.h"
#include "Latency.h"
#include "Logger.h"
#include "Options.h"
#include "Overlay.h"
#include "Recognizer.h"
#include "Reinforcement.h"
#include "Targeting.h"
#include "Tracker.h"
#include "YoloDetector.h"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> gStop{false};  // set by Ctrl-C; the loop ends cleanly and the log is complete
// The SIGINT handler only sets the flag (an atomic, so that is safe inside a signal handler);
// the loop checks it once per frame.
void onInterrupt(int) { gStop = true; }

// Strips leading and trailing spaces and tabs (for the entries of --classes).
std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    return s.substr(first, s.find_last_not_of(" \t") - first + 1);
}

// What --classes asked for. `detect` is passed to Detector::detect (so detectors that can't
// produce a requested class don't run); `labelOnly` and `labels` then hide the tracks that
// weren't asked for.
struct ClassFilter {
    std::set<ClassId> detect;     // classes to detect; empty = all
    std::set<ClassId> labelOnly;  // base classes requested only through a label
    std::set<int> labels;         // requested labels (indices into the library)

    // Asking for "me" detects faces but shows only the face recognized as me.
    // So: a track whose base class wasn't restricted is always shown; a restricted one is
    // shown only when the recognizer has given it one of the requested labels.
    bool shows(const Track& t, const ClassRegistry& classes, const Recognizer* recognizer) const {
        if (!labelOnly.count(classes.base(t.classId))) return true;
        const int label = recognizer ? recognizer->labelOf(t.id) : -1;
        return label >= 0 && labels.count(label);
    }
};

// Parses the --classes list (e.g. "person,face" or "face,alex") against the registry. Each name
// is a built-in class (detected and shown) or a custom label (detects its base class, shows only
// faces recognized as that label). Throws on an unknown name. An empty list means everything.
ClassFilter resolveClasses(const std::string& csv, const ClassRegistry& classes,
                           const LabelLibrary& library) {
    ClassFilter filter;
    std::set<ClassId> explicitClasses;
    std::stringstream ss(csv);
    for (std::string name; std::getline(ss, name, ',');) {
        name = trim(name);
        if (name.empty()) continue;
        const ClassId id = classes.find(name);
        if (id < 0) {
            std::string known;  // "person, face"
            for (ClassId c = 0; c < classes.size(); ++c)
                if (classes.kind(c) != ClassKind::Custom)
                    known += (known.empty() ? "" : ", ") + classes.name(c);
            throw std::runtime_error("Unknown class or label '" + name + "' (classes: " + known +
                                     "; labels: see --list-labels)");
        }
        if (classes.kind(id) == ClassKind::Custom) {
            filter.detect.insert(classes.base(id));
            filter.labelOnly.insert(classes.base(id));
            filter.labels.insert(library.find(name));
        } else {
            filter.detect.insert(id);
            explicitClasses.insert(id);
        }
    }
    // "--classes face,alex" asks for all faces as well, so a label doesn't restrict them.
    for (ClassId id : explicitClasses) filter.labelOnly.erase(id);
    return filter;
}

// Adds one detector. A model given with a flag must load; a default one that
// fails is skipped with a warning so the other detectors still run.
// `make` is a factory returning the new detector; it throws if the model can't be loaded
// (a template parameter so each call can pass its own lambda).
template <typename Make>
void addDetector(Detector& detector, const ModelPath& model, Make make) {
    try {
        detector.add(make());
    } catch (const std::exception& e) {
        if (model.explicitlySet) throw;
        std::cerr << "Warning: " << e.what() << " (detector disabled)\n";
    }
}

// Creates the YOLO (people) detector and, unless --no-faces, the face detector, and adds them to
// `detector`. Throws if none could be loaded. The face detector's thresholds are
// {confidence, low = 0.1, NMS = 0.3}; YOLO keeps the default low (0.1) and NMS (0.45).
void buildDetectors(Detector& detector, const Options& opt, const ClassRegistry& classes) {
    detector.setParallel(opt.parallel);
    addDetector(detector, opt.yoloModel, [&] {
        return std::make_unique<YoloDetector>(opt.yoloModel.path, kCocoClasses,
                                              DetectorThresholds{opt.confYolo}, opt.backend,
                                              opt.cacheDir);
    });
    if (opt.faces)
        addDetector(detector, opt.faceModel, [&] {
            return std::make_unique<FaceDetector>(opt.faceModel.path, classes.face(),
                                                  DetectorThresholds{opt.confFace, 0.1f, 0.3f},
                                                  opt.faceWidth);
        });
    if (detector.empty()) throw std::runtime_error("No detector could be loaded");
}

// "the yolo detector", "the yolo and face detectors".
std::string detectorNames(const std::vector<std::string>& names) {
    std::string text = "the ";
    for (size_t i = 0; i < names.size(); ++i)
        text += (i == 0 ? "" : i + 1 < names.size() ? ", " : " and ") + names[i];
    return text + (names.size() == 1 ? " detector" : " detectors");
}

// Makes each saved label a class (so --classes and the overlay know its name).
// Every label is a refinement of "face". One whose name can't be registered (say, it clashes
// with an existing class) is reported and left out; the rest still work.
void registerLabels(const LabelLibrary& library, ClassRegistry& classes) {
    for (const CustomLabel& label : library.labels()) {
        try {
            classes.addCustom(label.name, classes.face());
        } catch (const std::exception& e) {
            std::cerr << "Warning: label " << label.name << ": " << e.what() << '\n';
        }
    }
}

// --dump-detections: appends one CSV row per raw detection of this frame (before tracking), in
// the columns main() wrote as the header. keypoints is "x y x y ..." in frame pixels, empty for
// people. tools/replay.cpp reads these files to re-run the tracker without the neural networks.
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

// Faces with the laser dot on them now, or the TARGET: they look unlike
// themselves, so recognition doesn't photograph them for their label.
// Returns track ids.
std::set<int> litFaces(const TargetSelector& targeting) {
    std::set<int> lit;
    for (const auto& [id, f] : targeting.faces())
        if (f.onSince >= 0) lit.insert(id);
    if (targeting.target().active()) lit.insert(targeting.target().trackId);
    return lit;
}

// "Face network: off: needs other faces (0 of 20 seen); ...".
// The face network is a small neural network trained on the labels' face embeddings (see
// FaceNet.h); it stays off until there is enough data, and recognition then uses the labels'
// photos alone.
void printFaceNet(const std::string& status) {
    std::cout << "Face network: " << status;
    if (status.rfind("on", 0) != 0) std::cout << "; recognition uses the photos alone";
    std::cout << '\n';
}

// --dump-laser: appends one CSV row per face the laser search looked at this frame, with what it
// found (dot position, size and color measures; see LaserDot), and whether that face is the
// TARGET. A developer aid for checking the laser search on recorded clips.
void dumpLaser(std::ofstream& out, long long frame, double tMs, const TargetSelector& targeting) {
    for (const auto& [id, f] : targeting.faces()) {
        if (!f.checked) continue;  // not looked at this frame
        const LaserDot& d = f.dot;
        char line[160];
        std::snprintf(line, sizeof(line), "%lld,%.1f,%d,%d,%.1f,%.1f,%d,%.1f,%.0f,%.2f,%d\n", frame,
                      tMs, id, d.found ? 1 : 0, d.center.x, d.center.y, d.area, d.redness, d.glow,
                      d.skinAround, targeting.target().trackId == id ? 1 : 0);
        out << line;
    }
}

}  // namespace

// The function-try-block turns any exception thrown below (bad option, missing model, camera
// failure, ...) into an "Error: ..." line and exit code 1.
int main(int argc, char** argv) try {
    // OpenCV's DNN module prints a warning about compute targets on every launch.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    // PROJECT_DIR is the source folder, baked in by CMake, so models/, logs/ and labels/ are
    // found wherever the program is started from.
    Options opt = defaultOptions(PROJECT_DIR);
    if (!parseArgs(std::vector<std::string>(argv + 1, argv + argc), opt)) return 0;

    ClassRegistry classes;  // person and face; labels are added to it below
    LabelLibrary library(opt.labelsDir);
    library.load();  // label folders (ones made by an older version are converted)

    // One-off commands that need no detectors or camera: do the job and exit.
    if (opt.listClasses) {
        registerLabels(library, classes);
        for (ClassId id = 0; id < classes.size(); ++id) std::cout << classes.name(id) << '\n';
        return 0;
    }
    if (opt.listLabels) {
        listLabels(library);
        if (!library.labels().empty()) printFaceNet(faceNetStatus(library, FaceEmbedder::kModelId));
        return 0;
    }
    if (!opt.removeLabel.empty()) {
        removeLabel(opt.removeLabel, library);
        return 0;
    }
    if (!opt.renameLabel.empty()) {
        renameLabel(opt.renameLabel, opt.renameTo, library, classes);
        return 0;
    }

    // Fall back to the square YOLO model if the 16:9 one hasn't been exported.
    if (!opt.yoloModel.explicitlySet && !fs::exists(opt.yoloModel.path))
        opt.yoloModel.path = fs::path(opt.yoloModel.path).parent_path() / "yolov8n.onnx";

    // Label photos are searched for faces, whatever this run tracks.
    const bool labelCommand = !opt.enrollName.empty() || opt.syncLabels;
    if (labelCommand) opt.faces = true;
    Detector detector;
    buildDetectors(detector, opt, classes);
    // Without the face detector, a sync would record photos as showing no face,
    // for good: the label commands stop here instead, before anything is copied
    // or synced. (The other detectors aren't needed.)
    const std::vector<std::string> missing = missingForSync(detector.names());
    if (labelCommand && !missing.empty())
        throw std::runtime_error("Can't look at label photos: " + detectorNames(missing) +
                                 " didn't load (see the warning above). A sync needs it, or "
                                 "photos would be recorded as showing no face; nothing was "
                                 "changed.");
    // SFace: turns an aligned face crop into an embedding (a vector of numbers) such that two
    // photos of the same person score a high similarity. Labels store embeddings of their photos.
    FaceEmbedder faceEmbedder(opt.faceIdModel.path);  // loaded on first use
    // The label commands that do need the face detector (to find faces in photos):
    if (!opt.enrollName.empty()) {
        enroll(opt.enrollName, opt.enrollPhotos, detector, classes, faceEmbedder, library,
               opt.addPhotos);
        return 0;
    }
    if (opt.syncLabels) {
        syncLabels(library, detector, classes, faceEmbedder, true);
        if (!library.labels().empty()) printFaceNet(trainFaceNet(library, faceEmbedder.modelId()));
        return 0;
    }
    // Photos added to label folders since the last run are looked at now, for
    // faces (so not when the face detector is turned off or didn't load: the
    // labels then stay as the last sync left them).
    bool synced = false;
    if (opt.labels && library.needsSync()) {
        if (missing.empty()) {
            syncLabels(library, detector, classes, faceEmbedder, false);
            synced = true;
        } else if (opt.faces) {
            std::cout << "Label photos changed, but weren't looked at: " << detectorNames(missing)
                      << " didn't load (see the warning above). Labels not updated.\n";
        } else {
            std::cout << "Label photos changed, but weren't looked at: that needs "
                      << detectorNames(missing)
                      << ", turned off for this run. Labels not updated; run --sync-labels to "
                         "look at them.\n";
        }
    }
    // Folders that can't be labels (a bad name or label.yml, an earlier version's
    // object label): a sync can't fix them, so they're only reported (a sync
    // reports them itself).
    if (opt.labels && !synced)
        for (const LabelFolder& f : library.folders())
            if (!f.problem.empty())
                std::cout << "Label '" << f.name << "' can't be used: " << f.problem << '\n';
    // From here on this is a tracking run: the labels become classes (after the sync, so new
    // labels are included), and --classes is resolved against them.
    registerLabels(library, classes);

    std::cout << "Detectors (" << (opt.parallel ? "side by side" : "one after another") << "):\n";
    for (const std::string& line : detector.summary()) std::cout << "  " << line << '\n';
    const ClassFilter filter = resolveClasses(opt.classes, classes, library);

    // Recognition exists only when there are labels to recognize. The Recognizer embeds face
    // crops on a worker thread and gives tracks their label names; the Reinforcer takes
    // automatic photos of recognized faces and keeps other faces for the face network.
    // The reinforcer outlives the recognizer, which hands it the last faces it kept.
    std::unique_ptr<Reinforcer> reinforcer;
    std::unique_ptr<Recognizer> recognizer;
    if (opt.labels && !library.labels().empty()) {
        reinforcer = std::make_unique<Reinforcer>(library.dir(),
                                                  faceNetData(library, faceEmbedder.modelId()));
        RecognizerParams params;
        params.autoPhotos = opt.autoPhotos;
        recognizer =
            std::make_unique<Recognizer>(library, classes, faceEmbedder, params, reinforcer.get());
        std::cout << "Labels (faces):";
        for (const CustomLabel& label : library.labels()) std::cout << ' ' << label.name;
        std::cout << '\n';
        printFaceNet(reinforcer->status());
        if (opt.autoPhotos)
            std::cout
                << "Auto-photos: on (a recognized face, now and then, into its label's folder "
                   "as auto_*.jpg; --no-auto-photos turns them off)\n";
    }

    // The input: a camera (asked for 1280x720; the size it actually gives is used from here on)
    // or a video file. Both are read through the same Camera interface.
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

    // Optional outputs: the CSV run log and the developer dumps (their header rows are
    // written here, the rows in the loop).
    std::unique_ptr<Logger> logger;
    if (opt.log) {
        logger = std::make_unique<Logger>(opt.logDir);
        std::cout << "Logging to " << logger->path() << '\n';
    }
    std::ofstream dump, laserDump;
    if (!opt.dumpDetections.empty()) {
        dump.open(opt.dumpDetections);
        if (!dump) throw std::runtime_error("Failed to open " + opt.dumpDetections);
        dump << "frame,t_ms,class,conf,weak,x,y,w,h,keypoints\n";
    }
    if (!opt.dumpLaser.empty()) {
        laserDump.open(opt.dumpLaser);
        if (!laserDump) throw std::runtime_error("Failed to open " + opt.dumpLaser);
        laserDump << "frame,t_ms,id,dot,x,y,area,redness,glow,skin_around,target\n";
    }
    std::ofstream labelDump;
    if (!opt.dumpLabels.empty() && recognizer) {
        labelDump.open(opt.dumpLabels);
        if (!labelDump) throw std::runtime_error("Failed to open " + opt.dumpLabels);
        labelDump << "t,track,class,best_label,score,runner_up,match,assigned,net_label,net_prob,"
                     "sharpness,photo\n";
        recognizer->setDebugStream(&labelDump);
    }
    if (!opt.saveFrames.empty()) fs::create_directories(opt.saveFrames);

    // The per-frame stages after detection. The tracker needs the frame size to tell a box cut
    // off by the frame edge (an object leaving) from one that merely disappeared.
    Tracker tracker(classes);
    tracker.setFrameSize(camera.frameSize());
    TargetSelector targeting(classes, opt.target);
    const std::string window = "Object Tracker";
    if (opt.window) cv::namedWindow(window, cv::WINDOW_NORMAL);
    std::signal(SIGINT, onInterrupt);

    // Loop state. steady_clock never jumps (unlike the wall clock), so it is the right clock
    // for measuring durations.
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto lastFrame = start;
    double fps = 0.0;
    long long frameIndex = 0;
    cv::Mat frame;
    double tMs = 0;
    // What the overlay shows; the t / d / h keys change it while running. The target and
    // recognizer pointers are null when those features are off.
    DrawOptions draw;
    draw.trails = opt.showTrail;
    draw.debug = opt.debug;
    draw.targeting = opt.targeting ? &targeting : nullptr;
    draw.recognizer = recognizer.get();
    Overlay overlay;
    cv::Mat raw;  // the frame before drawing, to redraw it while paused
    double drawMs = 0;
    auto uptime = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
    // Running totals for the summary at the end. `latency` keeps every frame's time per stage
    // for the mean and 95th percentile.
    double detectMsTotal = 0, drawMsTotal = 0;
    std::map<std::string, double> detectorMsTotal;
    LatencyStats latency;
    // Milliseconds elapsed since `t`.
    auto msSince = [](Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    // The tracks --classes lets through, refilled each frame (only when a label filter is set).
    std::vector<Track> filtered;

    // --record: writes the raw camera feed to a video file, to replay later with --video.
    cv::VideoWriter recorder;
    if (!opt.record.empty()) {
        // H.264 where the system offers it (AVFoundation on macOS), else MPEG-4.
        for (const char* codec : {"avc1", "mp4v"}) {
            recorder.open(opt.record,
                          cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]),
                          camera.fps(), camera.frameSize());
            if (recorder.isOpened()) break;
        }
        if (!recorder.isOpened()) throw std::runtime_error("Can't record to " + opt.record);
        std::cout << "Recording the raw feed to " << opt.record << '\n';
    }

    // The main loop, one iteration per frame, in pipeline order: read -> (record, mirror) ->
    // detect -> track -> target + recognize -> filter -> log -> draw -> save / show -> keys.
    // camera.read() blocks until a new frame is ready, so the loop runs at the speed of the
    // slower of the camera and the processing; it ends when the file runs out, --max-frames is
    // reached, Ctrl-C is pressed or the user quits.
    while (!gStop && (opt.maxFrames == 0 || frameIndex < opt.maxFrames) &&
           camera.read(frame, tMs)) {
        const auto frameStart = Clock::now();
        const double ageMs = camera.frameAgeMs();  // waiting in the camera before processing
        latency.add("capture age", ageMs);
        // Recorded as the camera delivers it (not mirrored), like a QuickTime recording,
        // so it plays back with --video exactly as it was seen.
        if (recorder.isOpened()) recorder.write(frame);
        // Flipping in place is fine: a threaded Camera gives each frame its own buffer and never
        // writes to it again.
        if (opt.mirror && camera.isLive()) cv::flip(frame, frame, 1);

        // Stage 1, detection: people and faces for this frame, in frame pixels.
        const auto t0 = Clock::now();
        const auto detections = detector.detect(frame, filter.detect);
        const double detectMs = msSince(t0);
        detectMsTotal += detectMs;
        latency.add("detect", detectMs);
        for (const auto& t : detector.timings()) detectorMsTotal[t.name] += t.ms;
        if (dump.is_open()) dumpDetections(dump, frameIndex, tMs, detections, classes);

        // Stage 2, tracking: match detections to tracks. The tracker works in seconds, so the
        // frame's time (ms) is converted, and its Kalman filters advance by real elapsed time.
        const auto t1 = Clock::now();
        tracker.update(detections, tMs / 1000.0);
        latency.add("track", msSince(t1));
        // Stage 3, laser targeting and face recognition, both reading the clean frame.
        const auto t2 = Clock::now();
        // Targeting looks for the laser's colors and recognition copies crops:
        // both happen before anything is drawn on the frame. Targeting goes
        // first, so recognition knows which faces the laser is on.
        if (opt.targeting) targeting.update(frame, tracker.tracks(), tMs / 1000.0);
        if (recognizer)
            recognizer->update(frame, tracker.tracks(), tMs / 1000.0,
                               opt.targeting ? litFaces(targeting) : std::set<int>{});
        if (laserDump.is_open()) dumpLaser(laserDump, frameIndex, tMs, targeting);
        latency.add("labels+target", msSince(t2));

        // `shown`: the tracks that are drawn and logged. Normally all of them; with a label in
        // --classes, only those that pass the filter (see ClassFilter::shows).
        const std::vector<Track>* shown = &tracker.tracks();
        if (!filter.labelOnly.empty()) {
            filtered.clear();
            for (const Track& t : tracker.tracks())
                if (filter.shows(t, classes, recognizer.get())) filtered.push_back(t);
            shown = &filtered;
        }
        if (logger)
            logger->log(frameIndex, tMs, *shown, classes, opt.targeting ? &targeting : nullptr,
                        recognizer.get());

        // Smoothed FPS so the number doesn't flicker (an exponential moving average: each
        // frame counts for 10%).
        const auto now = Clock::now();
        const double instant = 1.0 / std::chrono::duration<double>(now - lastFrame).count();
        lastFrame = now;
        fps = fps == 0.0 ? instant : 0.9 * fps + 0.1 * instant;

        // Stage 4, drawing. drawMs shown on screen is the previous frame's (this one isn't
        // drawn yet). The clean frame is kept in `raw` first so pause can redraw it.
        FrameInfo info{fps, detector.timings(), drawMs, uptime()};
        info.recording = recorder.isOpened();
        if (opt.window) frame.copyTo(raw);
        const auto d0 = Clock::now();
        overlay.draw(frame, *shown, classes, draw, info);
        drawMs = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
        drawMsTotal += drawMs;
        latency.add("draw", drawMs);
        // --save-frames: every Nth annotated frame as a numbered JPEG.
        if (!opt.saveFrames.empty() && frameIndex % opt.saveEvery == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "/frame_%06lld.jpg", frameIndex);
            cv::imwrite(opt.saveFrames + name, frame);
        }
        double toScreen = -1;  // measured before any pause, so pausing isn't counted as lag
        if (opt.window) {
            const auto t4 = Clock::now();
            cv::imshow(window, frame);
            // waitKey(1) is also what lets the window repaint and collects a key press (-1 if
            // none); it waits at most 1 ms.
            int key = cv::waitKey(1);
            latency.add("display", msSince(t4));
            toScreen = ageMs + msSince(frameStart);
            // Pause: keep redrawing the clean frame (so view toggles show up at once) until p,
            // q, Esc (27) or Ctrl-C. Detection and tracking don't run meanwhile.
            if (key == 'p') {  // pause on this frame until p again; t, d and h still work
                info.paused = true;
                do {
                    applyViewKey(key, draw);
                    raw.copyTo(frame);
                    info.uptime = uptime();
                    overlay.draw(frame, *shown, classes, draw, info);
                    cv::imshow(window, frame);
                    key = cv::waitKey(50);
                } while (key != 'p' && key != 'q' && key != 27 && !gStop);
            }
            applyViewKey(key, draw);
            if (key == 27 || key == 'q') break;
        }
        // End-to-end latency: from when the camera captured the frame to when it was on screen
        // (or, with no window, when the loop was done with it).
        latency.add("capture to screen", toScreen >= 0 ? toScreen : ageMs + msSince(frameStart));
        ++frameIndex;
    }

    // After the loop: close the window and print the run summary.
    if (opt.window) cv::destroyAllWindows();
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::printf("Processed %lld frames in %.1f s (%.1f FPS)\n", frameIndex, seconds,
                seconds > 0 ? frameIndex / seconds : 0.0);
    if (frameIndex > 0) {
        // Detection per frame is the whole detect() call (about the slower detector when they
        // run in parallel); the brackets give each detector's own mean.
        std::printf("Detection per frame: %.1f ms (", detectMsTotal / frameIndex);
        const char* sep = "";
        for (const auto& [name, ms] : detectorMsTotal) {
            std::printf("%s%s %.1f", sep, name.c_str(), ms / frameIndex);
            sep = ", ";
        }
        std::printf(")\n");
        std::printf("Drawing per frame: %.2f ms\n", drawMsTotal / frameIndex);
        std::printf("Latency per frame, ms (mean/95th percentile): %s\n",
                    latency.summary().c_str());
    }
    // Frames the camera thread overwrote before the loop picked them up (see Camera).
    if (camera.dropped() > 0)
        std::printf("Skipped %lld frames that arrived while busy\n", camera.dropped());
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
