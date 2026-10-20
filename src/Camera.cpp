#include "Camera.h"

namespace {

double fpsOrDefault(double fps) { return fps > 0 ? fps : 30.0; }

}  // namespace

Camera::~Camera() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    changed_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool Camera::openDevice(int index, int width, int height) {
    // AVFoundation is the native macOS camera API.
    if (!cap_.open(index, cv::CAP_AVFOUNDATION)) return false;
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    live_ = true;
    fps_ = fpsOrDefault(cap_.get(cv::CAP_PROP_FPS));
    size_ = cv::Size(static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                     static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    threaded_ = true;
    start_ = std::chrono::steady_clock::now();
    thread_ = std::thread(&Camera::captureLoop, this);
    return true;
}

bool Camera::openFile(const std::string& path, bool realtime) {
    if (!cap_.open(path)) return false;
    live_ = false;
    fps_ = fpsOrDefault(cap_.get(cv::CAP_PROP_FPS));
    size_ = cv::Size(static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                     static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (realtime) {
        threaded_ = true;
        start_ = std::chrono::steady_clock::now();
        thread_ = std::thread(&Camera::captureLoop, this);
    }
    return true;
}

bool Camera::grab(cv::Mat& frame, double& timeMs) {
    if (!cap_.read(frame) || frame.empty()) return false;
    timeMs =
        live_ ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                    .count()
              : frameCount_ * 1000.0 / fps_;
    ++frameCount_;
    return true;
}

void Camera::captureLoop() {
    for (;;) {
        Clock::time_point due;
        if (!live_) {
            // Realtime playback: wait until this frame is due, as a camera would deliver it.
            due = start_ +
                  std::chrono::duration_cast<Clock::duration>(
                      std::chrono::duration<double, std::milli>(frameCount_ * 1000.0 / fps_));
            std::unique_lock<std::mutex> lock(mutex_);
            if (changed_.wait_until(lock, due, [this] { return stop_; })) return;
        }
        cv::Mat frame;  // a new buffer each time, so the frame handed out is never overwritten
        double timeMs = 0;
        const bool ok = grab(frame, timeMs);
        const Clock::time_point capturedAt = live_ ? Clock::now() : due;

        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        if (!ok) {
            ended_ = true;
            changed_.notify_all();
            return;
        }
        if (fresh_) ++dropped_;
        latest_ = frame;
        latestMs_ = timeMs;
        latestAt_ = capturedAt;
        fresh_ = true;
        changed_.notify_all();
    }
}

bool Camera::read(cv::Mat& frame, double& timeMs) {
    if (!threaded_) {
        readAt_ = Clock::now();
        return grab(frame, timeMs);
    }
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return fresh_ || ended_ || stop_; });
    if (!fresh_) return false;
    frame = latest_;
    timeMs = latestMs_;
    readAt_ = latestAt_;
    fresh_ = false;
    return true;
}

double Camera::frameAgeMs() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - readAt_).count();
}

long long Camera::dropped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
}
