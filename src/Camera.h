// The first stage of the pipeline (camera thread -> detectors -> tracker -> targeting/recognition
// -> overlay/log): frames from the MacBook camera or from a video file. The key idea is that a
// live camera is read on its own thread which keeps only the NEWEST frame. If detection and
// tracking take longer than one frame interval, older frames are overwritten instead of queuing
// up, so the overlay always shows what the camera sees now (low latency) at the cost of skipped
// frames, which are counted. Video files can be read frame by frame (every frame processed,
// so runs are repeatable) or paced like a camera (--realtime).
#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>

// Frames from the camera or a video file.
// A camera is read on a background thread that always keeps the newest frame,
// so slow processing skips stale frames instead of falling behind. A video file
// is read frame by frame, or, with realtime playback, paced like a camera.
// Threading: every public function is for the one processing thread. The capture thread is an
// internal detail; it hands frames over under mutex_.
class Camera {
public:
    Camera() = default;
    // Stops and joins the capture thread.
    ~Camera();
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    // Opens camera `index` through AVFoundation (macOS's camera API), asking for width x height
    // (the camera may give another size; see frameSize()), and starts the capture thread.
    // Returns false if it can't be opened (no such camera, or the app lacks camera permission).
    bool openDevice(int index, int width, int height);
    // Opens a video file. realtime=false: no thread, every frame is decoded as read() asks for
    // it. realtime=true: a capture thread delivers frames at the file's own frame rate, like a
    // camera, and drops frames when processing is slow. Returns false if it can't be opened.
    bool openFile(const std::string& path, bool realtime = false);

    // Next frame and its time in ms (camera: since opening; file: the frame's
    // position in the video). Returns false at the end of a file.
    // Blocks (threaded sources) until a frame newer than the last one returned is available.
    // The returned frame is a BGR image the caller may draw on: the capture thread never
    // writes to it again.
    bool read(cv::Mat& frame, double& timeMs);

    // true for a camera, false for a file.
    bool isLive() const { return live_; }
    // Size in pixels as the source actually delivers it.
    cv::Size frameSize() const { return size_; }
    // The source's frame rate; 30 if it doesn't report one.
    double fps() const { return fps_; }
    long long dropped() const;  // frames skipped because processing was still busy
    // How long ago the frame last returned by read() was captured (realtime
    // playback: when it was due), in ms. It grows while processing waits.
    // (Without a capture thread: since read() was called.)
    double frameAgeMs() const;

private:
    using Clock = std::chrono::steady_clock;
    // Body of the capture thread: grabs frames and publishes the newest into latest_.
    void captureLoop();
    // Reads one frame from cap_ and stamps it (live: ms since open; file: its position in the
    // video). Returns false at the end of the file or on a camera error.
    bool grab(cv::Mat& frame, double& timeMs);

    cv::VideoCapture cap_;
    // live_: a camera, not a file. threaded_: a capture thread is running (a camera, or a file
    // played in realtime).
    bool live_ = false;
    bool threaded_ = false;
    double fps_ = 30.0;
    cv::Size size_;
    // Frames grabbed so far; a file frame's timestamp and due time are derived from it.
    long long frameCount_ = 0;
    std::chrono::steady_clock::time_point start_;

    // The members below up to dropped_ are the hand-off between the capture thread (which
    // writes them) and the processing thread (read()); mutex_ protects them. changed_ is
    // signaled on a new frame, the end of the file, or a stop request. readAt_ is only used by
    // the processing thread.
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    cv::Mat latest_;
    double latestMs_ = 0;
    Clock::time_point latestAt_;  // when latest_ was captured
    Clock::time_point readAt_;    // ... and the frame last handed out by read()
    bool fresh_ = false;  // latest_ hasn't been handed out yet
    // ended_: the file ended (or the camera stopped delivering). stop_: the destructor asks the
    // capture thread to exit.
    bool ended_ = false;
    bool stop_ = false;
    long long dropped_ = 0;
};
