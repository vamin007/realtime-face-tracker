// Camera: the capture thread and the hand-off to the processing thread. See Camera.h for the
// overview. The hand-off is a one-frame "mailbox" (latest_) plus a flag (fresh_): the capture
// thread overwrites it with every new frame, read() takes it and waits if it is empty.
#include "Camera.h"

namespace {

// Some sources (webcams, odd files) report 0 or a negative frame rate.
double fpsOrDefault(double fps) { return fps > 0 ? fps : 30.0; }

}  // namespace

Camera::~Camera() {
    // Ask the capture thread to stop and wake it if it is waiting for the next frame's due
    // time. (A thread blocked inside cap_.read() finishes that read first.)
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
    // The size is read back rather than assumed: the camera may not offer the one requested.
    size_ = cv::Size(static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                     static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    threaded_ = true;
    // Frame timestamps count from here.
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
    // Without realtime there is no thread: read() decodes straight from the file.
    if (realtime) {
        threaded_ = true;
        start_ = std::chrono::steady_clock::now();
        thread_ = std::thread(&Camera::captureLoop, this);
    }
    return true;
}

bool Camera::grab(cv::Mat& frame, double& timeMs) {
    if (!cap_.read(frame) || frame.empty()) return false;
    // A camera is stamped with the wall clock; a file with the frame's position in the video
    // (frame number / fps), so a file's times are the same however fast it is processed.
    timeMs =
        live_ ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                    .count()
              : frameCount_ * 1000.0 / fps_;
    ++frameCount_;
    return true;
}

// Runs on the capture thread until the file ends, a read fails or the destructor asks it to stop.
// Grabbing happens outside the lock, so a slow camera read never blocks read() or dropped().
void Camera::captureLoop() {
    for (;;) {
        Clock::time_point due;
        if (!live_) {
            // Realtime playback: wait until this frame is due, as a camera would deliver it.
            // (The wait is interruptible: stop_ ends it at once.)
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
            // End of the file (or a camera error): wake read() so it can report it. A frame
            // already waiting in latest_ is still delivered first.
            ended_ = true;
            changed_.notify_all();
            return;
        }
        // The previous frame was never picked up: this is where "keep only the newest frame"
        // happens. It is overwritten below and counted as dropped.
        if (fresh_) ++dropped_;
        latest_ = frame;
        latestMs_ = timeMs;
        latestAt_ = capturedAt;
        fresh_ = true;
        changed_.notify_all();
    }
}

bool Camera::read(cv::Mat& frame, double& timeMs) {
    // Plain file: decode on the calling thread; the frame's "age" counts from now.
    if (!threaded_) {
        readAt_ = Clock::now();
        return grab(frame, timeMs);
    }
    // Threaded: sleep until the capture thread has published a frame we haven't taken yet.
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return fresh_ || ended_ || stop_; });
    if (!fresh_) return false;
    // `frame` shares latest_'s pixel buffer (no copy). That is safe because the capture thread
    // always grabs into a new buffer, never into this one.
    frame = latest_;
    timeMs = latestMs_;
    readAt_ = latestAt_;
    fresh_ = false;
    return true;
}

// Used for the "capture age" and "capture to screen" latency figures.
double Camera::frameAgeMs() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - readAt_).count();
}

long long Camera::dropped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
}
