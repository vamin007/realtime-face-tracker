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

# Everything except main(), so the tests link the same code the app runs.
add_library(tracker_core STATIC
    src/Camera.cpp
    src/Detector.cpp
    src/FaceDetector.cpp
    src/HandDetector.cpp
    src/Tracker.cpp
    src/Overlay.cpp
    src/Logger.cpp
)
target_include_directories(tracker_core PUBLIC src)
target_link_libraries(tracker_core PUBLIC ${OpenCV_LIBS})

add_executable(tracker src/main.cpp)
target_link_libraries(tracker PRIVATE tracker_core)

# Lets the program find models/ and logs/ no matter where it's run from.
target_compile_definitions(tracker PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")

# Unit tests: build with the rest, run with `ctest --test-dir build`.
find_package(GTest)
if(GTest_FOUND)
    enable_testing()
    add_executable(tracker_tests
        tests/TrackerTest.cpp
    )
    target_link_libraries(tracker_tests PRIVATE tracker_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(tracker_tests)
else()
    message(STATUS "GoogleTest not found (brew install googletest); skipping tests")
endif()
