#include <gtest/gtest.h>

#include <stdexcept>

#include "ClassRegistry.h"

TEST(ClassRegistry, CocoClassesThenHandThenFace) {
    const ClassRegistry classes({"person", "bicycle", "cup"});
    EXPECT_EQ(classes.cocoCount(), 3);
    EXPECT_EQ(classes.person(), 0);
    EXPECT_EQ(classes.find("cup"), 2);
    EXPECT_EQ(classes.hand(), 3);
    EXPECT_EQ(classes.face(), 4);
    EXPECT_EQ(classes.kind(classes.find("cup")), ClassKind::Object);
    EXPECT_EQ(classes.kind(classes.hand()), ClassKind::Hand);
    EXPECT_EQ(classes.kind(classes.face()), ClassKind::Face);
}

TEST(ClassRegistry, OldNamesFileWithHandAndFaceStillWorks) {
    const ClassRegistry classes({"person", "cup", "hand", "face"});
    EXPECT_EQ(classes.cocoCount(), 2);
    EXPECT_EQ(classes.hand(), 2);
    EXPECT_EQ(classes.face(), 3);
    EXPECT_EQ(classes.size(), 4);
}

TEST(ClassRegistry, InvalidIdsAreSafe) {
    const ClassRegistry classes({"person"});
    EXPECT_EQ(classes.name(-1), "unknown");
    EXPECT_EQ(classes.name(99), "unknown");
    EXPECT_EQ(classes.find("unicorn"), -1);
}

TEST(ClassRegistry, CustomLabelRefinesItsBaseClass) {
    ClassRegistry classes({"person", "cup"});
    const ClassId cup = classes.find("cup");
    const ClassId mug = classes.addCustom("my_mug", cup);
    EXPECT_EQ(classes.name(mug), "my_mug");
    EXPECT_EQ(classes.kind(mug), ClassKind::Custom);
    EXPECT_EQ(classes.base(mug), cup);
    EXPECT_EQ(classes.base(cup), cup);
    EXPECT_THROW(classes.addCustom("my_mug", cup), std::invalid_argument);  // duplicate
    EXPECT_THROW(classes.addCustom("mug2", mug), std::invalid_argument);    // custom base
    EXPECT_THROW(classes.addCustom("mug3", 999), std::invalid_argument);    // no such class
}

TEST(ClassRegistry, ProjectNamesFileHasTheEightyCocoClasses) {
    const ClassRegistry classes = ClassRegistry::fromFile(PROJECT_DIR "/models/coco.names");
    EXPECT_EQ(classes.cocoCount(), 80);
    EXPECT_EQ(classes.person(), 0);
    EXPECT_EQ(classes.find("toothbrush"), 79);
    EXPECT_EQ(classes.hand(), 80);
    EXPECT_EQ(classes.face(), 81);
}
