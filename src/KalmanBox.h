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
    };

    KalmanBox() = default;
    KalmanBox(const cv::Rect2f& box, const Params& params);

    void predict(double dt);              // advance the state by dt seconds
    void correct(const cv::Rect2f& box);  // blend in a measured box
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
