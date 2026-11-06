// Per-track motion filter: smooths a detector's jittery boxes and predicts where an object is
// between detections. One KalmanBox lives inside each Track (see Tracker.h).
#pragma once
#include <opencv2/core.hpp>

// Kalman filter for one box with a constant-velocity model in real time units.
// State: center x, y, width, height and their velocities (pixels, pixels/second).
// Unlike cv::KalmanFilter it is a plain value type: copying it copies the state.
//
// A Kalman filter keeps a best guess of the state together with how uncertain that guess is (the
// covariance matrix P), and alternates two steps:
//   predict: move the state forward by dt seconds assuming constant velocity. The uncertainty
//            grows by the process noise, the allowance for the object not really moving at
//            constant velocity (it accelerates, turns).
//   correct: pull the state toward a measured box (a detection). How far depends on the Kalman
//            gain, which compares the state's uncertainty with the measurement noise (how much a
//            detector's box jitters even when the object is still).
// The transition and process-noise matrices are rebuilt from the real dt of every frame, so the
// speed estimate does not depend on the frame rate.
class KalmanBox {
public:
    struct Params {
        // Noise levels relative to the box size (the square root of its area), so
        // small and large objects behave alike. The measure* values are the measurement
        // noise (jitter of a detection around the true box); the accel* values are the
        // process noise (standard deviation of the random acceleration that makes the
        // real motion differ from constant velocity).
        float measurePos = 0.05f;   // detection jitter of the center
        float measureSize = 0.10f;  // detection jitter of width/height
        float accelPos = 20.0f;     // how fast the center's velocity can change (sizes/s²)
        float accelSize = 5.0f;     // how fast width/height growth can change (sizes/s²)
        float initialSpeed = 5.0f;  // uncertainty of a new box's velocity (sizes/s)
        // Following fast motion. A constant-velocity filter smooths jitter but trails
        // behind an object that speeds up or turns (a head turned quickly). Where the filtered
        // box still differs from the detection by more than snapStart times the
        // detection jitter above, it is pulled toward the detection, all the way at
        // snapFull. Smaller differences are jitter and stay smoothed.
        // Between the two the pull grows linearly from 0 to 1 (an abrupt switch would look jumpy).
        float snapStart = 1.0f;
        float snapFull = 2.0f;
        // Width and height snap only at this many times those differences: boxes can
        // change size a lot from frame to frame (hands did, when they were tracked), and
        // following that looked jumpy.
        float snapSizeFactor = 2.0f;
    };

    // A default-constructed filter is empty (all zeros); assign a real one before use.
    KalmanBox() = default;
    // Starts at `box` (frame pixels) with zero velocity and a wide velocity uncertainty
    // (initialSpeed), since a single box says nothing about speed.
    KalmanBox(const cv::Rect2f& box, const Params& params);

    // dt <= 0 does nothing. The uncertainty grows with every predict, so after a long gap the
    // next correct() trusts the detection more.
    void predict(double dt);              // advance the state by dt seconds
    // Blends in a measured box. noiseScale: how much noisier this detection is
    // than the jitter in Params (a weak detection): it counts for less and must
    // differ more before the box snaps to it. box is in the same frame pixels as the
    // constructor's.
    void correct(const cv::Rect2f& box, float noiseScale = 1.0f);
    // Multiplies all four velocities (x, y, width, height) by factor; the Tracker uses it to slow a
    // track that is unseen, so its prediction waits near where the object was lost.
    void dampVelocity(float factor);      // scale the velocity (used while unseen)
    // The Tracker calls this for a box cut off by the frame edge: it shrinks because the object
    // is leaving, which is no trend to extrapolate.
    void stopResizing();                  // zero the width/height velocity

    // The current estimate: top-left corner, width and height, in frame pixels.
    cv::Rect2f box() const;
    // Velocity of the box center, pixels per second.
    cv::Point2f velocity() const { return {x_(4), x_(5)}; }

private:
    using Vec8 = cv::Matx<float, 8, 1>;
    using Mat8 = cv::Matx<float, 8, 8>;

    // The box's scale, the square root of its area (each side at least 1 px). The noise levels in
    // Params are multiples of it.
    float size() const;

    Params p_;
    // x_: center x, y, width, height, then their velocities. P_: the 8x8 covariance of x_.
    Vec8 x_ = Vec8::zeros();
    Mat8 P_ = Mat8::zeros();
};
