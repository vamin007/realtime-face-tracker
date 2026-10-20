#pragma once
#include <opencv2/core.hpp>

// Kalman filter for one box with a constant-velocity model in real time units.
// State: center x, y, width, height and their velocities (pixels, pixels/second).
// Unlike cv::KalmanFilter it is a plain value type: copying it copies the state.
class KalmanBox {
public:
    struct Params {
        // Noise levels relative to the box size (the square root of its area), so
        // small and large objects behave alike.
        float measurePos = 0.05f;   // detection jitter of the center
        float measureSize = 0.10f;  // detection jitter of width/height
        float accelPos = 20.0f;     // how fast the center's velocity can change (sizes/s²)
        float accelSize = 5.0f;     // how fast width/height growth can change (sizes/s²)
        float initialSpeed = 5.0f;  // uncertainty of a new box's velocity (sizes/s)
        // Following fast motion. A constant-velocity filter smooths jitter but trails
        // behind an object that speeds up or turns (a waved hand). Where the filtered
        // box still differs from the detection by more than snapStart times the
        // detection jitter above, it is pulled toward the detection, all the way at
        // snapFull. Smaller differences are jitter and stay smoothed.
        float snapStart = 1.0f;
        float snapFull = 2.0f;
        // Width and height snap only at this many times those differences: hand boxes
        // change size a lot from frame to frame, and following that looked jumpy.
        float snapSizeFactor = 2.0f;
    };

    KalmanBox() = default;
    KalmanBox(const cv::Rect2f& box, const Params& params);

    void predict(double dt);              // advance the state by dt seconds
    // Blends in a measured box. noiseScale: how much noisier this detection is
    // than the jitter in Params (a weak detection): it counts for less and must
    // differ more before the box snaps to it.
    void correct(const cv::Rect2f& box, float noiseScale = 1.0f);
    void dampVelocity(float factor);      // scale the velocity (used while unseen)
    void stopResizing();                  // zero the width/height velocity

    cv::Rect2f box() const;
    cv::Point2f velocity() const { return {x_(4), x_(5)}; }

private:
    using Vec8 = cv::Matx<float, 8, 1>;
    using Mat8 = cv::Matx<float, 8, 8>;

    float size() const;

    Params p_;
    Vec8 x_ = Vec8::zeros();
    Mat8 P_ = Mat8::zeros();
};
