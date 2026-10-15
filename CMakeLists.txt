cmake_minimum_required(VERSION 3.20)
project(object_tracker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build; inference is much slower in Debug.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(tracker
    src/main.cpp
    src/Camera.cpp
    src/Detector.cpp
    src/FaceDetector.cpp
    src/HandDetector.cpp
    src/Tracker.cpp
    src/Overlay.cpp
    src/Logger.cpp
)
target_link_libraries(tracker PRIVATE ${OpenCV_LIBS})

# Lets the program find models/ and logs/ no matter where it's run from.
target_compile_definitions(tracker PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")
