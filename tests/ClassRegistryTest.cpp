// Tests for ClassRegistry: the two built-in classes and their ids, safe handling of invalid ids
// and names, and custom labels (a recognized person's name layered on "face").
#include <gtest/gtest.h>

#include <stdexcept>

#include "ClassRegistry.h"

// person must be id 0 (YOLO's person index) and face id 1; nothing else is registered by default.
TEST(ClassRegistry, TheTrackersClassesArePersonAndFace) {
    const ClassRegistry classes;
    EXPECT_EQ(classes.size(), 2);
    EXPECT_EQ(classes.person(), 0);
    EXPECT_EQ(classes.face(), 1);
    EXPECT_EQ(classes.find("person"), classes.person());
    EXPECT_EQ(classes.find("face"), classes.face());
    EXPECT_EQ(classes.kind(classes.person()), ClassKind::Object);
    EXPECT_EQ(classes.kind(classes.face()), ClassKind::Face);
    // Hands aren't tracked, and YOLO's other COCO classes aren't among them.
    EXPECT_EQ(classes.find("hand"), -1);
    EXPECT_EQ(classes.find("cup"), -1);
    EXPECT_EQ(classes.find("cell phone"), -1);
}

// Lookups never read out of range; they return "unknown" / -1.
TEST(ClassRegistry, InvalidIdsAreSafe) {
    const ClassRegistry classes;
    EXPECT_EQ(classes.name(-1), "unknown");
    EXPECT_EQ(classes.name(99), "unknown");
    EXPECT_EQ(classes.find("unicorn"), -1);
}

// A custom label has its own id and name but reports "face" as its base class; names must be
// unique, and a label can only sit on a valid built-in class (not on another label).
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
    EXPECT_THROW(classes.addCustom("person", face), std::invalid_argument);  // a built-in name
    EXPECT_THROW(classes.addCustom("me2", me), std::invalid_argument);     // custom base
    EXPECT_THROW(classes.addCustom("me3", 999), std::invalid_argument);    // no such class
}
