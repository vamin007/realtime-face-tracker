#include <gtest/gtest.h>

#include "TagLayout.h"

namespace {

const cv::Size kFrame(1280, 720);
const cv::Size kTag(80, 24);

bool inside(const cv::Rect& r, cv::Size frame) { return (r & cv::Rect({0, 0}, frame)) == r; }

}  // namespace

TEST(TagLayout, ATagSitsOnTopOfItsBoxTopLeftCorner) {
    TagLayout layout;
    const auto spots = layout.place({{1, {400, 300, 200, 150}, kTag}}, kFrame);
    ASSERT_EQ(spots.size(), 1u);
    EXPECT_EQ(spots[0], cv::Rect(400, 276, 80, 24));
}

TEST(TagLayout, TagsStayInsideTheFrame) {
    TagLayout layout;
    const std::vector<TagLayout::Item> items = {
        {1, {0, 0, 300, 200}, kTag},           // top-left corner: no room above
        {2, {1250, 300, 30, 100}, {120, 24}},  // right edge, tag wider than the box
        {3, {500, 690, 100, 30}, kTag},        // bottom edge
        {4, {-50, 400, 100, 100}, kTag},       // partly outside
        {5, {0, 0, 1280, 720}, {2000, 800}},   // a tag bigger than the frame
    };
    const auto spots = layout.place(items, kFrame);
    ASSERT_EQ(spots.size(), items.size());
    for (size_t i = 0; i + 1 < spots.size(); ++i) EXPECT_TRUE(inside(spots[i], kFrame)) << i;
    EXPECT_EQ(spots.back().tl(), cv::Point(0, 0));
}

TEST(TagLayout, TagsOfNearbyBoxesDontOverlap) {
    // A person's box, with a face and two hands near its top-left corner.
    TagLayout layout;
    const std::vector<TagLayout::Item> items = {{1, {300, 150, 700, 570}, {100, 24}},
                                                {2, {300, 155, 200, 250}, {80, 24}},
                                                {3, {320, 160, 150, 150}, {80, 24}},
                                                {4, {310, 150, 120, 100}, {80, 24}}};
    const auto spots = layout.place(items, kFrame);
    for (size_t i = 0; i < spots.size(); ++i)
        for (size_t j = i + 1; j < spots.size(); ++j)
            EXPECT_TRUE((spots[i] & spots[j]).empty()) << i << " and " << j;
}

TEST(TagLayout, ReservedAreasAreAvoided) {
    TagLayout layout;
    const cv::Rect hud(12, 12, 300, 70);
    const auto spots = layout.place({{1, {20, 90, 200, 200}, kTag}}, kFrame, {hud});
    EXPECT_TRUE((spots[0] & hud).empty());
    EXPECT_TRUE(inside(spots[0], kFrame));
}

TEST(TagLayout, EarlierItemsGetTheBetterSpot) {
    TagLayout layout;
    const cv::Rect box(400, 300, 200, 150);
    const auto spots = layout.place({{7, box, kTag}, {3, box, kTag}}, kFrame);
    EXPECT_EQ(spots[0], cv::Rect(400, 276, 80, 24));
    EXPECT_TRUE((spots[0] & spots[1]).empty());
}

TEST(TagLayout, APushedTagSettlesBeforeMovingBack) {
    TagLayout layout;
    const TagLayout::Item item{2, {400, 300, 200, 150}, kTag};
    const cv::Rect home(400, 276, 80, 24);
    // Another object's tag takes this tag's spot...
    const TagLayout::Item other{1, {390, 290, 100, 100}, kTag};
    const cv::Rect pushed = layout.place({other, item}, kFrame)[1];
    EXPECT_NE(pushed, home);
    // ...then that object leaves: the tag stays put for a while, then goes home.
    for (int i = 0; i < TagLayout::kSettleFrames; ++i)
        EXPECT_EQ(layout.place({item}, kFrame)[0], pushed) << i;
    EXPECT_EQ(layout.place({item}, kFrame)[0], home);
}

TEST(TagLayout, ATagMovesBackOnlyWithRoomToSpare) {
    TagLayout layout;
    const TagLayout::Item item{2, {400, 300, 200, 150}, kTag};
    const cv::Rect home(400, 276, 80, 24);
    const cv::Rect pushed = layout.place({{1, {390, 290, 100, 100}, kTag}, item}, kFrame)[1];
    // The other tag moves to just left of home: home is free, but only barely.
    const TagLayout::Item close{1, {316, 300, 50, 50}, kTag};
    for (int i = 0; i < 2 * TagLayout::kSettleFrames; ++i) {
        const auto spots = layout.place({close, item}, kFrame);
        ASSERT_EQ(spots[0], cv::Rect(316, 276, 80, 24));
        EXPECT_EQ(spots[1], pushed) << i;
    }
    EXPECT_EQ(layout.place({item}, kFrame)[0], home);  // once it's gone
}

TEST(TagLayout, AForcedMoveIsImmediate) {
    // Overlaps aren't allowed to wait: a tag whose spot is taken moves at once.
    TagLayout layout;
    const TagLayout::Item item{2, {400, 300, 200, 150}, kTag};
    const cv::Rect home(400, 276, 80, 24);
    EXPECT_EQ(layout.place({item}, kFrame)[0], home);
    const auto spots = layout.place({{1, {390, 290, 100, 100}, kTag}, item}, kFrame);
    EXPECT_TRUE((spots[0] & spots[1]).empty());
}
