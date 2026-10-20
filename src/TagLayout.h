#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <vector>

// Places each box's tag (the plate with its name) next to the box: inside the
// frame and, while there is room, not on top of another tag or a reserved area
// such as the HUD.
class TagLayout {
public:
    struct Item {
        int id;        // stable from frame to frame (the track id): a tag keeps its spot
        cv::Rect box;  // the object's box
        cv::Size size;
    };

    // Frames a tag stays in a spot it was pushed to before it may move back to a
    // better one (0.5 s at 30 FPS).
    static constexpr int kSettleFrames = 15;

    // Call once per frame. Items earlier in the list get the better spots.
    // Returns one rect per item, in the same order.
    std::vector<cv::Rect> place(const std::vector<Item>& items, cv::Size frame,
                                const std::vector<cv::Rect>& reserved = {});

private:
    struct Placed {
        int spot;    // index into the candidate spots, best first
        int frames;  // frames it has been there
    };
    std::map<int, Placed> placed_;  // by id, last frame
};
