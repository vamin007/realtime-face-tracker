#include <gtest/gtest.h>

#include <stdexcept>

#include "ClassRegistry.h"

TEST(ClassRegistry, TheTrackersClassesArePersonHandAndFace) {
    const ClassRegistry classes;
    EXPECT_EQ(classes.size(), 3);
    EXPECT_EQ(classes.person(), 0);
    EXPECT_EQ(classes.hand(), 1);
    EXPECT_EQ(classes.face(), 2);
    EXPECT_EQ(classes.find("person"), classes.person());
    EXPECT_EQ(classes.find("hand"), classes.hand());
    EXPECT_EQ(classes.find("face"), classes.face());
    EXPECT_EQ(classes.kind(classes.person()), ClassKind::Object);
    EXPECT_EQ(classes.kind(classes.hand()), ClassKind::Hand);
    EXPECT_EQ(classes.kind(classes.face()), ClassKind::Face);
    // YOLO's other COCO classes aren't among them.
    EXPECT_EQ(classes.find("cup"), -1);
    EXPECT_EQ(classes.find("cell phone"), -1);
}

TEST(ClassRegistry, InvalidIdsAreSafe) {
    const ClassRegistry classes;
    EXPECT_EQ(classes.name(-1), "unknown");
    EXPECT_EQ(classes.name(99), "unknown");
    EXPECT_EQ(classes.find("unicorn"), -1);
}

TEST(ClassRegistry, CustomLabelRefinesItsBaseClass) {
    ClassRegistry classes;
    const ClassId face = classes.face();
    const ClassId me = classes.addCustom("me", face);
    EXPECT_EQ(classes.name(me), "me");
    EXPECT_EQ(classes.kind(me), ClassKind::Custom);
    EXPECT_EQ(classes.base(me), face);
    EXPECT_EQ(classes.base(face), face);
    EXPECT_EQ(classes.find("me"), me);
    EXPECT_THROW(classes.addCustom("me", face), std::invalid_argument);    // duplicate
    EXPECT_THROW(classes.addCustom("hand", face), std::invalid_argument);  // a built-in name
    EXPECT_THROW(classes.addCustom("me2", me), std::invalid_argument);     // custom base
    EXPECT_THROW(classes.addCustom("me3", 999), std::invalid_argument);    // no such class
}
