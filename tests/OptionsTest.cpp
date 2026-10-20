#include <gtest/gtest.h>

#include <stdexcept>

#include "Options.h"

namespace {

Options parse(const std::vector<std::string>& args) {
    Options opt = defaultOptions("/project");
    parseArgs(args, opt);
    return opt;
}

}  // namespace

TEST(Options, Defaults) {
    const Options opt = parse({});
    EXPECT_EQ(opt.yoloModel.path, "/project/models/yolov8n_384x640.onnx");
    EXPECT_FALSE(opt.yoloModel.explicitlySet);
    EXPECT_EQ(opt.labelsDir, "/project/labels");
    EXPECT_TRUE(opt.targeting);
    EXPECT_TRUE(opt.labels);
    EXPECT_FLOAT_EQ(opt.confFace, 0.6f);
}

TEST(Options, NumbersAreChecked) {
    EXPECT_FLOAT_EQ(parse({"--conf-hand", "0.5"}).confHand, 0.5f);
    EXPECT_THROW(parse({"--conf", "abc"}), std::runtime_error);
    EXPECT_THROW(parse({"--conf", "1.5"}), std::runtime_error);
    EXPECT_THROW(parse({"--conf", "0.5x"}), std::runtime_error);
    EXPECT_THROW(parse({"--camera"}), std::runtime_error);  // missing value
    const Options all = parse({"--conf", "0.3"});
    EXPECT_FLOAT_EQ(all.confYolo, 0.3f);
    EXPECT_FLOAT_EQ(all.confFace, 0.3f);
    EXPECT_FLOAT_EQ(all.confHand, 0.3f);
}

TEST(Options, ErrorMessagesSayWhatWasExpected) {
    try {
        parse({"--conf", "abc"});
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("--conf expects a number from 0 to 1"),
                  std::string::npos)
            << e.what();
    }
}

TEST(Options, EnrollTakesANameAndPhotos) {
    const Options opt = parse({"--enroll", "my_mug", "a.jpg", "b.jpg", "--class", "cup"});
    EXPECT_EQ(opt.enrollName, "my_mug");
    EXPECT_EQ(opt.enrollClass, "cup");
    EXPECT_EQ(opt.enrollPhotos, (std::vector<std::string>{"a.jpg", "b.jpg"}));
    EXPECT_THROW(parse({"--enroll", "my_mug"}), std::runtime_error);  // no photos
    EXPECT_THROW(parse({"a.jpg"}), std::runtime_error);               // photos without --enroll
    EXPECT_THROW(parse({"--class", "cup"}), std::runtime_error);      // --class without --enroll
}

TEST(Options, LabelCommands) {
    Options opt = parse({"--add-photos", "my_mug", "c.jpg"});
    EXPECT_EQ(opt.enrollName, "my_mug");
    EXPECT_TRUE(opt.addPhotos);
    EXPECT_EQ(opt.enrollPhotos, std::vector<std::string>{"c.jpg"});
    EXPECT_THROW(parse({"--add-photos", "my_mug"}), std::runtime_error);  // no photos
    EXPECT_THROW(parse({"--add-photos", "my_mug", "c.jpg", "--class", "cup"}), std::runtime_error);
    EXPECT_TRUE(parse({"--sync-labels"}).syncLabels);
    opt = parse({"--rename-label", "mug", "my_mug"});
    EXPECT_EQ(opt.renameLabel, "mug");
    EXPECT_EQ(opt.renameTo, "my_mug");
    EXPECT_THROW(parse({"--rename-label", "mug"}), std::runtime_error);  // no new name
}

TEST(Options, ModelFlagsMarkTheModelAsRequired) {
    const Options opt = parse({"--model", "other.onnx"});
    EXPECT_EQ(opt.yoloModel.path, "other.onnx");
    EXPECT_TRUE(opt.yoloModel.explicitlySet);
}

TEST(Options, MisuseIsRejected) {
    EXPECT_THROW(parse({"--frobnicate"}), std::runtime_error);
    EXPECT_THROW(parse({"--no-window"}), std::runtime_error);  // camera with no way to stop
    EXPECT_NO_THROW(parse({"--no-window", "--video", "clip.mov"}));
    EXPECT_THROW(parse({"--backend", "gpu"}), std::runtime_error);
}
