// The command line, as data. main() parses argv into an Options once at startup and then reads
// it everywhere: which detectors to build, whether to open the camera or a video, which
// overlay features are on, and which one-off command (label management) to run instead of
// tracking. Keeping parsing in its own unit (Options.cpp) means it can be tested without
// starting the pipeline (tests/OptionsTest.cpp).
#pragma once
#include <string>
#include <vector>

#include "OrtModel.h"   // Backend
#include "Targeting.h"  // TargetParams

// A model file location, remembering whether the user named it. A default model that fails to
// load only disables that detector, with a warning (see addDetector in main.cpp).
struct ModelPath {
    std::string path;
    bool explicitlySet = false;  // given with a flag: failing to load it is an error
};

// Everything the command line can set.
struct Options {
    std::string classes;  // comma-separated, empty = all; may include custom labels
    // Confidence thresholds for people (YOLO) and faces: a detection at or above its threshold
    // can start a track (weaker ones may only keep a track going).
    float confYolo = 0.4f;
    float confFace = 0.6f;
    int camera = 0;
    std::string video;      // play a file instead of the camera
    bool realtime = false;  // play the video at its own speed, skipping frames when busy
    // Where YOLO runs: CoreML (GPU / Neural Engine, falling back to the CPU) or plain CPU.
    Backend backend = Backend::CoreML;
    ModelPath yoloModel, faceModel;  // detectors
    ModelPath faceIdModel;           // face recognition (custom labels)
    // Where run logs, CoreML's compiled-model cache and label folders (photos) live.
    std::string logDir, cacheDir, labelsDir;
    // log: write the CSV run log. mirror: flip the live camera image horizontally, like a
    // mirror (never applied to video files). faces: run the face detector.
    bool log = true;
    bool mirror = true;
    bool faces = true;
    bool labels = true;      // recognize custom labels
    bool autoPhotos = true;  // photograph recognized faces for their label (and keep other faces)
    bool showTrail = false;
    bool debug = false;  // show laser and recognition details on the boxes
    std::string record;  // save the raw camera feed to this video file
    bool window = true;
    // targeting: look for the laser dot and mark the TARGET face. target holds its tuning
    // (--laser-redness changes target.minRedGain).
    bool targeting = true;
    TargetParams target;

    // One-off commands instead of tracking.
    bool listClasses = false;
    bool listLabels = false;
    bool syncLabels = false;  // look at new and changed label photos, report, exit
    std::string removeLabel;
    std::string renameLabel, renameTo;
    std::string enrollName;  // --enroll NAME photo... (or --add-photos NAME photo...)
    bool addPhotos = false;  // --add-photos: the label must exist already
    std::vector<std::string> enrollPhotos;

    // Developer options
    long long maxFrames = 0;     // stop after this many frames; 0 = no limit
    std::string dumpDetections;  // CSV of every raw detection per frame
    std::string dumpLaser;       // CSV of each face's laser search per frame
    std::string dumpLabels;      // CSV of each recognition result
    std::string saveFrames;      // directory for annotated snapshots
    int saveEvery = 30;
    bool parallel = true;  // run the detectors side by side
    int faceWidth = 640;   // face detection works on frames scaled down to this width
};

// Defaults, with model, log and label folders inside projectDir.
// (main passes the compile-time PROJECT_DIR, so the program finds models/ whatever the
// working directory is.)
Options defaultOptions(const std::string& projectDir);

// Reads the command line (without the program name) into opt. Returns false
// after --help, once the usage is printed. Throws std::runtime_error with a
// readable message for bad input.
// Options not mentioned keep the values already in opt. Besides checking each flag, it
// rejects combinations that can't work (photos without --enroll, --no-window with a camera
// and no frame limit).
bool parseArgs(const std::vector<std::string>& args, Options& opt);

// Prints the --help text to stdout.
void printUsage();
