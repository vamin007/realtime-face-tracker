# Build definition for the object tracker (C++17, OpenCV, ONNX Runtime; macOS on Apple Silicon).
# Targets:
#   tracker_core    every source file except main(), built once as a static library
#   tracker         the app (src/main.cpp)
#   tracker_replay  developer tool: re-runs the tracker on saved detections
#   grab_frames     developer tool: saves video frames as photos
#   tracker_tests   the GoogleTest unit tests (only if GoogleTest is installed)
cmake_minimum_required(VERSION 3.20)
project(object_tracker CXX)

# C++17, with no fallback to an older standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # build/compile_commands.json, for editor tooling

# Default to an optimized build; inference is much slower in Debug.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies: OpenCV (image handling, the face models, drawing, the window) and ONNX Runtime
# (runs the YOLO model, on CoreML when it can).
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)  # brew install onnxruntime

# Everything except main(), so the tests link the same code the app runs.
add_library(tracker_core STATIC
    # Frames in, detections out: camera or video, the YOLO and face detectors, class ids.
    src/Camera.cpp
    src/ClassRegistry.cpp
    src/Detector.cpp
    src/OrtModel.cpp
    src/YoloDetector.cpp
    src/FaceDetector.cpp
    # Tracking: Hungarian matching of detections to tracks, a Kalman filter per track.
    src/Assignment.cpp
    src/KalmanBox.cpp
    src/Tracker.cpp
    # Laser TARGET: finds the red dot on a face.
    src/Targeting.cpp
    # Custom labels and face recognition: embeddings, the face network, label photos and their
    # sync, the background recognizer, automatic photos, and the --enroll command.
    src/Embedder.cpp
    src/FaceNet.cpp
    src/Labels.cpp
    src/Recognizer.cpp
    src/Reinforcement.cpp
    src/Enrollment.cpp
    # Command line, drawing the overlay (and placing its tags), and the CSV log.
    src/Options.cpp
    src/Overlay.cpp
    src/TagLayout.cpp
    src/Logger.cpp
)
# PUBLIC: whatever links tracker_core (the app, the tools, the tests) can #include "..." from
# src/ and gets OpenCV and ONNX Runtime too.
target_include_directories(tracker_core PUBLIC src)
target_link_libraries(tracker_core PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

# The app itself: just main(), everything else comes from tracker_core.
add_executable(tracker src/main.cpp)
target_link_libraries(tracker PRIVATE tracker_core)

# Lets the program find models/ and logs/ no matter where it's run from.
target_compile_definitions(tracker PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Developer tool: re-runs the tracker on saved detections (see tools/replay.cpp).
add_executable(tracker_replay tools/replay.cpp)
target_link_libraries(tracker_replay PRIVATE tracker_core)
target_compile_definitions(tracker_replay PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Developer tool: saves video frames as photos, for trying custom labels (see tools/grab_frames.cpp).
add_executable(grab_frames tools/grab_frames.cpp)
target_link_libraries(grab_frames PRIVATE tracker_core)
target_compile_definitions(grab_frames PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Unit tests: build with the rest, run with `ctest --test-dir build`.
# Optional: without GoogleTest the rest still builds.
find_package(GTest)
if(GTest_FOUND)
    enable_testing()
    add_executable(tracker_tests
        tests/AssignmentTest.cpp
        tests/ClassRegistryTest.cpp
        tests/DetectorTest.cpp
        tests/EmbedderTest.cpp
        tests/FaceNetTest.cpp
        tests/KalmanBoxTest.cpp
        tests/LabelsTest.cpp
        tests/LabelSyncTest.cpp
        tests/OptionsTest.cpp
        tests/OrtModelTest.cpp
        tests/OverlayTest.cpp
        tests/ReinforcementTest.cpp
        tests/TagLayoutTest.cpp
        tests/TargetingTest.cpp
        tests/TrackerTest.cpp
    )
    # gtest_main supplies main(). PROJECT_DIR lets the tests that use the real models and the
    # cache folder find them.
    target_link_libraries(tracker_tests PRIVATE tracker_core GTest::gtest_main)
    target_compile_definitions(tracker_tests PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")
    # Registers every TEST() with ctest as its own test, so failures are reported one by one.
    include(GoogleTest)
    gtest_discover_tests(tracker_tests)
else()
    message(STATUS "GoogleTest not found (brew install googletest); skipping tests")
endif()
