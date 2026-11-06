// Tag placement: candidate spots around a box, and the greedy choice among them with a little
// memory (a tag pushed aside waits before returning) so tags stay steady while objects move.
#include "TagLayout.h"

#include <algorithm>
#include <climits>

namespace {

// Spots to try, best first: on top of the box's top-left corner (attached like
// a tab), just inside that corner (a box at the top of the frame), the same at
// the top-right corner, then the bottom corners, then further out above and
// below for crowded scenes. The top edge comes first so that a tag pushed
// aside stays close to where it was. Each spot is moved inside the frame.
std::vector<cv::Rect> candidates(const cv::Rect& box, cv::Size tag, cv::Size frame) {
    const int left = box.x, right = box.x + box.width - tag.width;
    const int above = box.y - tag.height, below = box.y + box.height;
    std::vector<cv::Point> spots = {{left, above},  {left, box.y},
                                    {right, above}, {right, box.y},
                                    {left, below},  {left, below - tag.height},
                                    {right, below}, {right, below - tag.height}};
    const int step = tag.height + 2;
    for (int k = 1; k <= 3; ++k) {
        spots.push_back({left, above - k * step});
        spots.push_back({left, below + k * step});
    }
    std::vector<cv::Rect> out;
    for (cv::Point p : spots) {
        p.x = std::clamp(p.x, 0, std::max(0, frame.width - tag.width));
        p.y = std::clamp(p.y, 0, std::max(0, frame.height - tag.height));
        out.emplace_back(p, tag);
    }
    return out;
}

// Total area of r that lies on top of the taken rects.
int overlap(const cv::Rect& r, const std::vector<cv::Rect>& taken) {
    int area = 0;
    for (const cv::Rect& t : taken) area += (r & t).area();
    return area;
}

// r enlarged by margin pixels on every side.
cv::Rect grown(const cv::Rect& r, int margin) {
    return {r.x - margin, r.y - margin, r.width + 2 * margin, r.height + 2 * margin};
}

}  // namespace

std::vector<cv::Rect> TagLayout::place(const std::vector<Item>& items, cv::Size frame,
                                       const std::vector<cv::Rect>& reserved) {
    // taken: every rect a new tag must stay off, starting with the reserved areas and growing as
    // tags are placed. placed: this frame's choices, which become next frame's memory.
    std::vector<cv::Rect> taken = reserved, out;
    std::map<int, Placed> placed;
    for (const Item& item : items) {
        const std::vector<cv::Rect> spots = candidates(item.box, item.size, frame);
        const int count = static_cast<int>(spots.size());
        // Spot i is free if it overlaps no taken rect, even after growing it by margin.
        auto isFree = [&](int i, int margin) {
            return overlap(grown(spots[i], margin), taken) == 0;
        };
        // A tag that was pushed to a worse spot moves back to a better one only
        // after it has settled where it is, and only if that one has room to
        // spare, so it doesn't hop back and forth while two objects pass each other.
        // (Hysteresis: being forced out of a spot is immediate, going back is delayed and needs
        // room to spare. Different conditions each way keep a tag from flickering between spots.)
        // Until it has settled it tries its current spot and worse ones first.
        const auto last = placed_.find(item.id);
        const int kept = last != placed_.end() ? last->second.spot : 0;
        const bool settled = last == placed_.end() || last->second.frames >= kSettleFrames;
        const int margin = item.size.height / 2;
        // best: the index of the chosen spot (spots are ordered best first).
        int best = -1;
        for (int i = settled ? 0 : kept; i < count && best < 0; ++i)
            if (isFree(i, i < kept ? margin : 0)) best = i;
        for (int i = 0; i < kept && best < 0; ++i)  // all worse spots are taken
            if (isFree(i, 0)) best = i;
        if (best < 0) {  // no free spot: the least covered one
            int least = INT_MAX;
            for (int i = 0; i < count; ++i)
                if (const int area = overlap(spots[i], taken); area < least) {
                    least = area;
                    best = i;
                }
        }
        out.push_back(spots[best]);
        taken.push_back(spots[best]);
        // Count the frames in a row at this spot; a move restarts the count.
        const bool stayed = last != placed_.end() && kept == best;
        placed[item.id] = {best, stayed ? std::min(last->second.frames + 1, kSettleFrames) : 0};
    }
    placed_ = std::move(placed);
    return out;
}
