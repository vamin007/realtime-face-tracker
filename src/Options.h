#pragma once
#include <string>
#include <vector>

#include "OrtModel.h"   // Backend
#include "Targeting.h"  // TargetParams

struct ModelPath {
    std::string path;
    bool explicitlySet = false;  // given with a flag: failing to load it is an error
};

// Everything the command line can set.
struct Options {
    std::string classes;  // comma-separated, empty = all; may include custom labels
    float confYolo = 0.4f;
    float confFace = 0.6f;
    int camera = 0;
    std::string video;      // play a file instead of the camera
    bool realtime = false;  // play the video at its own speed, skipping frames when busy
    Backend backend = Backend::CoreML;
    ModelPath yoloModel, faceModel;  // detectors
    ModelPath faceIdModel;           // face recognition (custom labels)
    std::string logDir, cacheDir, labelsDir;
    bool log = true;
    bool mirror = true;
    bool faces = true;
    bool labels = true;  // recognize custom labels
    bool showTrail = false;
    bool debug = false;  // show brightness and recognition details on the boxes
    std::string record;  // save the raw camera feed to this video file
    bool window = true;
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
Options defaultOptions(const std::string& projectDir);

// Reads the command line (without the program name) into opt. Returns false
// after --help, once the usage is printed. Throws std::runtime_error with a
// readable message for bad input.
bool parseArgs(const std::vector<std::string>& args, Options& opt);

void printUsage();
