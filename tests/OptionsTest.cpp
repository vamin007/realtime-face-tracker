// Tests for command-line parsing (Options.cpp): the defaults, number checking and error
// messages, the label commands (--enroll, --add-photos, --sync-labels, --rename-label), model
// flags, rejected misuse, and options that were removed with features that no longer exist.
#include <gtest/gtest.h>

#include <stdexcept>

#include "Options.h"

namespace {

// Parses `args` (without the program name) on top of the defaults for a fake project folder.
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

// Bad numbers (text, out of range, trailing characters) and a flag missing its value are errors;
// --conf sets both detectors' thresholds at once.
TEST(Options, NumbersAreChecked) {
    EXPECT_FLOAT_EQ(parse({"--conf-face", "0.5"}).confFace, 0.5f);
    EXPECT_THROW(parse({"--conf", "abc"}), std::runtime_error);
    EXPECT_THROW(parse({"--conf", "1.5"}), std::runtime_error);
    EXPECT_THROW(parse({"--conf", "0.5x"}), std::runtime_error);
    EXPECT_THROW(parse({"--camera"}), std::runtime_error);  // missing value
    const Options all = parse({"--conf", "0.3"});
    EXPECT_FLOAT_EQ(all.confYolo, 0.3f);
    EXPECT_FLOAT_EQ(all.confFace, 0.3f);
}

// The message names the flag and the accepted range, so the user can fix the command.
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

// Bare arguments are photos for --enroll; they are an error anywhere else.
TEST(Options, EnrollTakesANameAndPhotos) {
    const Options opt = parse({"--enroll", "alex", "a.jpg", "b.jpg"});
    EXPECT_EQ(opt.enrollName, "alex");
    EXPECT_EQ(opt.enrollPhotos, (std::vector<std::string>{"a.jpg", "b.jpg"}));
    EXPECT_THROW(parse({"--enroll", "alex"}), std::runtime_error);  // no photos
    EXPECT_THROW(parse({"a.jpg"}), std::runtime_error);             // photos without --enroll
}

// Every label is a face, found in each photo: an earlier version's options for
// object labels (what the item is, close-ups of it) are gone.
TEST(Options, ObjectLabelOptionsAreGone) {
    EXPECT_THROW(parse({"--enroll", "alex", "a.jpg", "--class", "face"}), std::runtime_error);
    EXPECT_THROW(parse({"--enroll", "alex", "a.jpg", "--whole-photo"}), std::runtime_error);
}

TEST(Options, LabelCommands) {
    Options opt = parse({"--add-photos", "alex", "c.jpg"});
    EXPECT_EQ(opt.enrollName, "alex");
    EXPECT_TRUE(opt.addPhotos);
    EXPECT_EQ(opt.enrollPhotos, std::vector<std::string>{"c.jpg"});
    EXPECT_THROW(parse({"--add-photos", "alex"}), std::runtime_error);  // no photos
    EXPECT_TRUE(parse({"--sync-labels"}).syncLabels);
    opt = parse({"--rename-label", "alex", "alex_b"});
    EXPECT_EQ(opt.renameLabel, "alex");
    EXPECT_EQ(opt.renameTo, "alex_b");
    EXPECT_THROW(parse({"--rename-label", "alex"}), std::runtime_error);  // no new name
}

// --no-auto-photos only stops the automatic photos; recognition itself stays on.
TEST(Options, AutoPhotosAreOnUnlessTurnedOff) {
    EXPECT_TRUE(parse({}).autoPhotos);
    const Options opt = parse({"--no-auto-photos"});
    EXPECT_FALSE(opt.autoPhotos);
    EXPECT_TRUE(opt.labels) << "recognition stays on";
}

// A model named with a flag must load (explicitlySet); a default one that fails is just skipped.
TEST(Options, ModelFlagsMarkTheModelAsRequired) {
    const Options opt = parse({"--model", "other.onnx"});
    EXPECT_EQ(opt.yoloModel.path, "other.onnx");
    EXPECT_TRUE(opt.yoloModel.explicitlySet);
}

// Unknown flags, a camera run with no window and no way to stop, and bad flag values.
TEST(Options, MisuseIsRejected) {
    EXPECT_THROW(parse({"--frobnicate"}), std::runtime_error);
    EXPECT_THROW(parse({"--no-window"}), std::runtime_error);  // camera with no way to stop
    EXPECT_NO_THROW(parse({"--no-window", "--video", "clip.mov"}));
    EXPECT_THROW(parse({"--backend", "gpu"}), std::runtime_error);
}

// Hands aren't detected: their options are gone.
TEST(Options, HandOptionsAreGone) {
    EXPECT_THROW(parse({"--no-hands"}), std::runtime_error);
    EXPECT_THROW(parse({"--conf-hand", "0.5"}), std::runtime_error);
    EXPECT_THROW(parse({"--hand-model", "palm.onnx"}), std::runtime_error);
}
