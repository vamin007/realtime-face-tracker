// Tag placement for the overlay. A tag is the small plate with an object's name ("face #2",
// "TARGET"). Drawing each at a fixed spot would make tags cover each other and the boxes in a
// crowd, so this class picks a spot for every tag each frame: greedily, in priority order, from a
// fixed list of candidate positions around the box, avoiding tags already placed and reserved
// areas. It remembers the previous frame's choices so tags don't jump around.
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
    // frame: the frame's size in pixels; spots are moved to lie inside it. reserved: rects tags
    // should stay off (the HUD panel, the "TARGET LOCKED" badge). When no spot is free, the
    // one that overlaps the least is used, so a tag is always placed.
    std::vector<cv::Rect> place(const std::vector<Item>& items, cv::Size frame,
                                const std::vector<cv::Rect>& reserved = {});

private:
    // What was chosen for an item last frame (its spot and how many frames in a row it has been
    // there), kept so its tag doesn't hop around.
    struct Placed {
        int spot;    // index into the candidate spots, best first
        int frames;  // frames it has been there
    };
    std::map<int, Placed> placed_;  // by id, last frame
};
