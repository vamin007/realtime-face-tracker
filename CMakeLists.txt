cmake_minimum_required(VERSION 3.20)
project(object_tracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # build/compile_commands.json, for editor tooling

# Default to an optimized build; inference is much slower in Debug.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)  # brew install onnxruntime

# Everything except main(), so the tests link the same code the app runs.
add_library(tracker_core STATIC
    src/Camera.cpp
    src/ClassRegistry.cpp
    src/Detector.cpp
    src/OrtModel.cpp
    src/YoloDetector.cpp
    src/FaceDetector.cpp
    src/HandDetector.cpp
    src/Assignment.cpp
    src/KalmanBox.cpp
    src/Tracker.cpp
    src/Targeting.cpp
    src/Embedder.cpp
    src/Labels.cpp
    src/Recognizer.cpp
    src/Enrollment.cpp
    src/Options.cpp
    src/Overlay.cpp
    src/TagLayout.cpp
    src/Logger.cpp
)
target_include_directories(tracker_core PUBLIC src)
target_link_libraries(tracker_core PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)

add_executable(tracker src/main.cpp)
target_link_libraries(tracker PRIVATE tracker_core)

# Lets the program find models/ and logs/ no matter where it's run from.
target_compile_definitions(tracker PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Developer tool: re-runs the tracker on saved detections (see tools/replay.cpp).
add_executable(tracker_replay tools/replay.cpp)
target_link_libraries(tracker_replay PRIVATE tracker_core)
target_compile_definitions(tracker_replay PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Unit tests: build with the rest, run with `ctest --test-dir build`.
find_package(GTest)
if(GTest_FOUND)
    enable_testing()
    add_executable(tracker_tests
        tests/AssignmentTest.cpp
        tests/ClassRegistryTest.cpp
        tests/DetectorTest.cpp
        tests/EmbedderTest.cpp
        tests/KalmanBoxTest.cpp
        tests/LabelsTest.cpp
        tests/OptionsTest.cpp
        tests/OrtModelTest.cpp
        tests/OverlayTest.cpp
        tests/TagLayoutTest.cpp
        tests/TargetingTest.cpp
        tests/TrackerTest.cpp
    )
    target_link_libraries(tracker_tests PRIVATE tracker_core GTest::gtest_main)
    target_compile_definitions(tracker_tests PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")
    include(GoogleTest)
    gtest_discover_tests(tracker_tests)
else()
    message(STATUS "GoogleTest not found (brew install googletest); skipping tests")
endif()
