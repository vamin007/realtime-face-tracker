#include "KalmanBox.h"

#include <algorithm>
#include <cmath>

namespace {

float sq(float v) { return v * v; }

// 0 up to start, 1 from full on, linear in between.
float ramp(float v, float start, float full) {
    if (full <= start) return v > start ? 1.0f : 0.0f;
    return std::clamp((v - start) / (full - start), 0.0f, 1.0f);
}

}  // namespace

KalmanBox::KalmanBox(const cv::Rect2f& box, const Params& params) : p_(params) {
    x_(0) = box.x + box.width / 2;
    x_(1) = box.y + box.height / 2;
    x_(2) = box.width;
    x_(3) = box.height;
    const float s = size();
    for (int i = 0; i < 2; ++i) P_(i, i) = sq(p_.measurePos * s);
    for (int i = 2; i < 4; ++i) P_(i, i) = sq(p_.measureSize * s);
    for (int i = 4; i < 8; ++i) P_(i, i) = sq(p_.initialSpeed * s);
}

float KalmanBox::size() const { return std::sqrt(std::max(x_(2), 1.0f) * std::max(x_(3), 1.0f)); }

void KalmanBox::predict(double dtSeconds) {
    const float dt = static_cast<float>(std::max(dtSeconds, 0.0));
    if (dt <= 0) return;
    Mat8 F = Mat8::eye();
    for (int i = 0; i < 4; ++i) F(i, i + 4) = dt;  // position += velocity * dt
    x_ = F * x_;

    // Random changes in velocity ("white-noise acceleration"): for each value and
    // its velocity, Q = a² [[dt⁴/4, dt³/2], [dt³/2, dt²]].
    Mat8 Q = Mat8::zeros();
    const float s = size();
    for (int i = 0; i < 4; ++i) {
        const float a2 = sq((i < 2 ? p_.accelPos : p_.accelSize) * s);
        Q(i, i) = a2 * sq(dt) * sq(dt) / 4;
        Q(i, i + 4) = Q(i + 4, i) = a2 * dt * sq(dt) / 2;
        Q(i + 4, i + 4) = a2 * sq(dt);
    }
    P_ = F * P_ * F.t() + Q;
    x_(2) = std::max(x_(2), 1.0f);
    x_(3) = std::max(x_(3), 1.0f);
}

void KalmanBox::correct(const cv::Rect2f& box, float noiseScale) {
    const cv::Matx<float, 4, 1> z(box.x + box.width / 2, box.y + box.height / 2, box.width,
                                  box.height);
    // The measurement picks the first four state values, so H*P*H' is P's top-left
    // block and P*H' is its first four columns.
    const float s = size();
    cv::Matx<float, 4, 4> S;
    cv::Matx<float, 8, 4> PHt;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c) PHt(r, c) = P_(r, c);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) S(r, c) = P_(r, c);
    const float measurePos = p_.measurePos * noiseScale, measureSize = p_.measureSize * noiseScale;
    for (int i = 0; i < 2; ++i) S(i, i) += sq(measurePos * s);
    for (int i = 2; i < 4; ++i) S(i, i) += sq(measureSize * s);

    const cv::Matx<float, 8, 4> K = PHt * S.inv(cv::DECOMP_CHOLESKY);
    cv::Matx<float, 4, 1> innovation;
    for (int i = 0; i < 4; ++i) innovation(i) = z(i) - x_(i);
    x_ += K * innovation;

    // P = (I - K H) P
    Mat8 KH = Mat8::zeros();
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c) KH(r, c) = K(r, c);
    P_ = (Mat8::eye() - KH) * P_;

    // Still far from the detection: the object moved in a way the model didn't
    // expect, so follow the detection instead of trailing behind it.
    const float centerOff = std::hypot(z(0) - x_(0), z(1) - x_(1)) / (measurePos * s);
    const float pullCenter = ramp(centerOff, p_.snapStart, p_.snapFull);
    for (int i = 0; i < 2; ++i) x_(i) += pullCenter * (z(i) - x_(i));
    for (int i = 2; i < 4; ++i) {
        const float off = std::abs(z(i) - x_(i)) / (measureSize * std::max(x_(i), 1.0f));
        x_(i) += ramp(off / p_.snapSizeFactor, p_.snapStart, p_.snapFull) * (z(i) - x_(i));
    }
    x_(2) = std::max(x_(2), 1.0f);
    x_(3) = std::max(x_(3), 1.0f);
}

void KalmanBox::dampVelocity(float factor) {
    for (int i = 4; i < 8; ++i) x_(i) *= factor;
}

void KalmanBox::stopResizing() {
    x_(6) = 0;
    x_(7) = 0;
}

cv::Rect2f KalmanBox::box() const { return {x_(0) - x_(2) / 2, x_(1) - x_(3) / 2, x_(2), x_(3)}; }
