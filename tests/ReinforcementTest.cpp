// Reinforcement: when a recognized face is photographed for its label, what
// happens to the photo, the faces of other people, and how the face network
// changes recognition. Synthetic embeddings and frames, so it runs in moments.
// Sections: the capture gates (capturePolicy) one by one, automatic photo names and replacement,
// NegativeStore, the network's relax and veto rules (combineMatch), and end-to-end tests that
// run a Recognizer and a Reinforcer over fake tracks and check the files they write.

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <sstream>

#include "Recognizer.h"
#include "Reinforcement.h"

namespace fs = std::filesystem;

namespace {

constexpr int kDim = 128;

Embedding normalized(Embedding e) {
    float n = 0;
    for (float x : e) n += x * x;
    for (float& x : e) x /= std::sqrt(n);
    return e;
}

// Someone's face: a random direction (different people are nearly orthogonal).
Embedding person(cv::RNG& rng) {
    Embedding e(kDim);
    for (float& x : e) x = static_cast<float>(rng.gaussian(1));
    return normalized(e);
}

// a with some of b: 1 / sqrt(1 + w^2) like a.
Embedding mix(const Embedding& a, const Embedding& b, float w) {
    Embedding e = a;
    for (size_t i = 0; i < e.size(); ++i) e[i] += w * b[i];
    return normalized(e);
}

std::vector<Embedding> views(const Embedding& center, int n, cv::RNG& rng, float spread) {
    std::vector<Embedding> out;
    for (int i = 0; i < n; ++i) {
        Embedding e = center;
        for (float& x : e) x += static_cast<float>(rng.gaussian(spread / std::sqrt(kDim)));
        out.push_back(normalized(e));
    }
    return out;
}

LabelSample sample(const std::string& image, const Embedding& e, bool flipped = false) {
    return {image, {0, 0, 10, 10}, flipped, e};
}

// A frame with one face, whose identity is painted into it: the face's blue
// channel is 10 x its code. The face is a red checkerboard, so it's sharp,
// unless it's `flat`.
const cv::Rect kBox(200, 100, 160, 200);

cv::Mat frameWith(int code, cv::Rect box = kBox, bool flat = false) {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(90, 90, 90));
    cv::Mat face = frame(box);
    face.setTo(cv::Scalar(code * 10, 128, 128));
    if (!flat)
        for (int y = 0; y < face.rows; ++y)
            for (int x = 0; x < face.cols; ++x)
                face.at<cv::Vec3b>(y, x)[2] = ((x / 4 + y / 4) % 2) ? 220 : 40;
    return frame;
}

// Embeds a face as the embedding of the code painted into it.
class CodeEmbedder : public IEmbedder {
public:
    std::string modelId() const override { return "test-code"; }
    float defaultThreshold() const override { return 0.363f; }  // SFace's
    Embedding embed(const cv::Mat& frame, const cv::Rect2f& box,
                    const std::vector<cv::Point2f>&) override {
        ++calls;
        const cv::Rect r = cv::Rect(box) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.area() <= 0) return {};
        const int code = static_cast<int>(std::lround(cv::mean(frame(r))[0] / 10));
        const auto it = faces.find(code);
        return it == faces.end() ? Embedding{} : it->second;
    }
    std::map<int, Embedding> faces;  // filled before tracking starts
    std::atomic<int> calls{0};
};

constexpr int kStranger = 21;    // a person who isn't a label
constexpr int kBorderline = 22;  // alex, scoring just below the threshold
constexpr int kLookAlike = 23;   // matches alex, but the network learned it's someone else

std::vector<std::string> autoPhotosIn(const fs::path& folder) {
    std::vector<std::string> names;
    for (const PhotoFile& p : listPhotos(folder.string()))
        if (isAutoPhotoName(p.name)) names.push_back(p.name);
    return names;
}

// Image files anywhere under dir.
int imagesUnder(const fs::path& dir) {
    int n = 0;
    for (const auto& entry : fs::recursive_directory_iterator(dir))
        n += entry.is_regular_file() && isPhotoName(entry.path().filename().string());
    return n;
}

// Fixture: a temp labels directory, a fake embedder that reads the "code" painted into each
// frame, and helpers that run a Recognizer and a Reinforcer over synthetic tracks, one frame per
// second of fake time.
class ReinforcementTest : public testing::Test {
protected:
    void SetUp() override {
        root = fs::path(testing::TempDir()) / "reinforcement";
        fs::remove_all(root);
        dir = (root / "labels").string();
        fs::create_directories(dir);
        library = LabelLibrary(dir);
        alex = person(rng);
        // Codes 1-20: new views of alex, each about 0.78 like the label's photos
        // and 0.64 like each other.
        for (int code = 1; code <= 20; ++code) embedder.faces[code] = mix(alex, person(rng), 0.75f);
        embedder.faces[kStranger] = person(rng);
    }
    void TearDown() override {
        stop();
        fs::remove_all(root);
    }

    // A label folder with a photo per embedding (named photo_N.jpg unless
    // `names` says), synced: its .embeddings.yml has each one, used.
    void makeLabel(const std::string& name, const std::vector<Embedding>& embeddings,
                   std::vector<std::string> names = {}) {
        const fs::path folder = fs::path(dir) / name;
        fs::create_directories(folder);
        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (names.size() <= i) names.push_back("photo_" + std::to_string(i) + ".jpg");
            cv::Mat image(40, 40 + static_cast<int>(i), CV_8UC3, cv::Scalar(i * 20, 90, 200));
            cv::imwrite((folder / names[i]).string(), image);
        }
        LabelCache cache{embedder.modelId(), kFaceClass, true, false, {}};
        for (const PhotoFile& p : listPhotos(folder.string())) {
            const size_t i = std::find(names.begin(), names.end(), p.name) - names.begin();
            PhotoRecord r;
            r.file = p;
            r.used = true;
            r.note = "face found, confidence 0.90";
            r.roi = {5, 5, 30, 30};
            r.conf = 0.9f;
            r.embedding = r.mirrored = embeddings[i];
            cache.photos.push_back(r);
        }
        writeCache((folder / kCacheFile).string(), cache);
    }

    // Other people's faces, as if collected while tracking: each list from one track.
    void saveOthers(const std::vector<std::vector<Embedding>>& tracks) {
        NegativeStore store;
        for (size_t i = 0; i < tracks.size(); ++i)
            for (const Embedding& e : tracks[i]) store.add(e, "track " + std::to_string(i));
        store.save((fs::path(dir) / kNegativesFile).string(), embedder.modelId());
    }
    std::vector<std::vector<Embedding>> strangers(int people, int each) {
        std::vector<std::vector<Embedding>> out;
        for (int i = 0; i < people; ++i) out.push_back(views(person(rng), each, rng, 0.6f));
        return out;
    }

    void start(RecognizerParams params = {}) {
        library.load();
        reinforcer = std::make_unique<Reinforcer>(dir, faceNetData(library, embedder.modelId()));
        recognizer =
            std::make_unique<Recognizer>(library, classes, embedder, params, reinforcer.get());
        recognizer->setDebugStream(&dump);
    }
    // The recognizer first (it hands over the other faces it kept), then the
    // reinforcer, which finishes writing.
    void stop() {
        recognizer.reset();
        reinforcer.reset();
    }

    Track face(int id, double t, cv::Rect box = kBox) const {
        Track tr;
        tr.id = id;
        tr.number = 1;
        tr.classId = classes.face();
        tr.conf = 0.9f;
        tr.box = tr.detected = box;
        tr.confirmed = tr.visible = tr.matched = true;
        tr.lastSeen = tr.lastStrong = t;
        return tr;
    }
    // One frame, then waits for the crops it sent to be embedded (their
    // results are used in the next frame).
    void step(double t, const cv::Mat& frame, const std::vector<Track>& tracks,
              const std::set<int>& lit = {}) {
        recognizer->update(frame, tracks, t, lit);
        recognizer->waitIdle();
    }
    // Alex in view from `from` to `to` s, one frame a second, a different view each time.
    void watchAlex(int from, int to, cv::Rect box = kBox, bool flat = false) {
        for (int t = from; t <= to; ++t)
            step(t, frameWith(t % 20 + 1, box, flat), {face(1, t, box)});
    }
    // Times of the photos taken (from the debug stream).
    std::vector<double> photoTimes() const {
        std::vector<double> times;
        std::stringstream lines(dump.str());
        for (std::string line; std::getline(lines, line);)
            if (line.size() > 6 && line.compare(line.size() - 6, 6, ",taken") == 0)
                times.push_back(std::stod(line.substr(0, line.find(','))));
        return times;
    }

    fs::path root;
    std::string dir;
    cv::RNG rng{2026};
    Embedding alex;
    ClassRegistry classes;
    CodeEmbedder embedder;
    LabelLibrary library{""};
    std::unique_ptr<Reinforcer> reinforcer;
    std::unique_ptr<Recognizer> recognizer;
    std::stringstream dump;
};

// The input to capturePolicy for a face that's worth a photo.
CaptureInput goodCapture() {
    CaptureInput in;
    in.enabled = true;
    in.streak = 5;
    in.score = 0.7f;
    in.threshold = 0.363f;
    in.runnerUp = 0.2f;
    in.detected = true;
    in.width = 160;
    in.sharpness = 1000;
    in.lit = false;
    in.similarity = 0.7f;
    in.now = 100;
    in.lastPhoto = -1;
    return in;
}

}  // namespace

// --- When a photo is taken ---------------------------------------------------

TEST(CapturePolicy, TakesAPhotoWhenEveryGatePasses) {
    EXPECT_EQ(capturePolicy(goodCapture()), CaptureGate::Take);
    EXPECT_STREQ(describe(CaptureGate::Take), "taken");
}

TEST(CapturePolicy, NotWhenTurnedOff) {
    CaptureInput in = goodCapture();
    in.enabled = false;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Off);
}

TEST(CapturePolicy, OnlyOnceTheLabelIsSteady) {
    CaptureInput in = goodCapture();
    in.streak = 4;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Unstable);
    in.streak = 0;  // not this track's label (yet)
    EXPECT_EQ(capturePolicy(in), CaptureGate::Unstable);
}

TEST(CapturePolicy, OnlyAClearMatch) {
    CaptureInput in = goodCapture();
    in.score = in.threshold + 0.049f;
    EXPECT_EQ(capturePolicy(in), CaptureGate::LowScore);
    in.score = in.threshold + 0.051f;
    in.runnerUp = in.score - 0.05f;  // enough for recognition, not for a photo
    EXPECT_EQ(capturePolicy(in), CaptureGate::CloseRunnerUp);
    in.runnerUp = in.score - 0.11f;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Take);
}

TEST(CapturePolicy, OnlyAFaceDetectedInThatFrame) {
    CaptureInput in = goodCapture();
    in.detected = false;  // coasting: the box is a prediction
    EXPECT_EQ(capturePolicy(in), CaptureGate::NotDetected);
}

TEST(CapturePolicy, OnlyALargeSharpFace) {
    CaptureInput in = goodCapture();
    in.width = 79;
    EXPECT_EQ(capturePolicy(in), CaptureGate::TooSmall);
    in.width = 80;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Take);
    in.sharpness = CaptureParams{}.minSharpness - 1;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Blurry);
}

TEST(CapturePolicy, NotTheTargetOrALitFace) {
    CaptureInput in = goodCapture();
    in.lit = true;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Lit);
}

TEST(CapturePolicy, OnlyAViewTheLabelDoesntHaveYet) {
    CaptureInput in = goodCapture();
    in.similarity = 0.86f;
    EXPECT_EQ(capturePolicy(in), CaptureGate::NotNew);
    in.similarity = 0.85f;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Take);
}

TEST(CapturePolicy, AtMostOnePhotoPerLabelEvery30Seconds) {
    CaptureInput in = goodCapture();
    in.lastPhoto = in.now - 29.9;
    EXPECT_EQ(capturePolicy(in), CaptureGate::TooSoon);
    in.lastPhoto = in.now - 30;
    EXPECT_EQ(capturePolicy(in), CaptureGate::Take);
    in.lastPhoto = 0;  // a photo at the very start of the run
    in.now = 10;
    EXPECT_EQ(capturePolicy(in), CaptureGate::TooSoon);
}

TEST(CapturePolicy, ABlurryFaceMeasuresLessSharp) {
    const cv::Mat sharp = frameWith(1);
    cv::Mat blurred;
    cv::GaussianBlur(sharp, blurred, {0, 0}, 4);
    const float s = sharpness(sharp, kBox), b = sharpness(blurred, kBox);
    EXPECT_GT(s, CaptureParams{}.minSharpness);
    EXPECT_LT(b, CaptureParams{}.minSharpness);
    EXPECT_EQ(sharpness(frameWith(1, kBox, true), kBox), 0) << "a flat face";
    EXPECT_EQ(sharpness(sharp, {700, 0, 10, 10}), 0) << "outside the image";
}

TEST(AutoPhotos, NamedByTheTimeTheyWereTaken) {
    std::tm when{};
    when.tm_year = 2026 - 1900, when.tm_mon = 10, when.tm_mday = 4;
    when.tm_hour = 9, when.tm_min = 5, when.tm_sec = 7;
    const std::time_t t = std::mktime(&when);
    auto none = [](const std::string&) { return false; };
    EXPECT_EQ(autoPhotoName(t, none), "auto_20261104_090507.jpg");
    std::set<std::string> taken = {"auto_20261104_090507.jpg", "auto_20261104_090507_2.jpg"};
    EXPECT_EQ(autoPhotoName(t, [&](const std::string& n) { return taken.count(n) > 0; }),
              "auto_20261104_090507_3.jpg");

    EXPECT_TRUE(isAutoPhotoName("auto_20261104_090507.jpg"));
    EXPECT_TRUE(isAutoPhotoName("auto_20261104_090507_3.jpg"));
    EXPECT_FALSE(isAutoPhotoName("auto_2026_090507.jpg"));
    EXPECT_FALSE(isAutoPhotoName("auto_20261104_090507.png"));
    EXPECT_FALSE(isAutoPhotoName("my_auto_20261104_090507.jpg"));
    EXPECT_FALSE(isAutoPhotoName("auto_20261104_090507_.jpg"));
    EXPECT_FALSE(isAutoPhotoName("photo.jpg"));
}

TEST(AutoPhotos, AFullLabelDropsItsMostRedundantAutoPhotoNeverAUsersOwn) {
    cv::RNG rng(5);
    const Embedding alex = person(rng);
    CustomLabel label{"alex", "test-model", 0, {}};
    // The user's two photos are almost the same: the most redundant of all.
    const Embedding mine = mix(alex, person(rng), 0.1f);
    label.samples = {sample("a.jpg", mine), sample("a.jpg", mine, true),
                     sample("b.jpg", mix(mine, person(rng), 0.01f))};
    // Three automatic ones, the second much like the user's.
    const std::vector<Embedding> autos = {
        mix(alex, person(rng), 0.8f), mix(mine, person(rng), 0.2f), mix(alex, person(rng), 0.8f)};
    const std::vector<std::string> names = {"auto_20261104_090000.jpg", "auto_20261104_090100.jpg",
                                            "auto_20261104_090200.jpg"};
    for (int i = 0; i < 3; ++i) {
        label.samples.push_back(sample(names[i], autos[i]));
        label.samples.push_back(sample(names[i], autos[i], true));
    }
    EXPECT_EQ(autoPhotoCount(label), 3);
    EXPECT_EQ(photoToReplace(label, 3), "") << "there's room";
    EXPECT_EQ(photoToReplace(label, 2), names[1]);
    EXPECT_NEAR(maxSimilarity(label, mine), 1, 1e-5);
}

// --- Other people's faces --------------------------------------------------

TEST(NegativeStore, KeepsAtMostItsCapacityDroppingTheMostRedundant) {
    cv::RNG rng(9);
    NegativeStore store(5);
    std::vector<Embedding> faces;
    for (int i = 0; i < 5; ++i) faces.push_back(person(rng));
    for (int i = 0; i < 5; ++i) store.add(faces[i], "track " + std::to_string(i));
    EXPECT_EQ(store.faces().size(), 5u);
    EXPECT_EQ(store.people(), 5);
    const Embedding twin = mix(faces[2], person(rng), 0.05f);  // the same person again
    store.add(twin, "track 5");
    ASSERT_EQ(store.faces().size(), 5u);
    auto has = [&](const Embedding& e) {
        return std::count(store.faces().begin(), store.faces().end(), e) > 0;
    };
    EXPECT_TRUE(has(faces[0]) && has(faces[1]) && has(faces[3]) && has(faces[4]));
    EXPECT_NE(has(faces[2]), has(twin)) << "one of the two is kept";
    EXPECT_EQ(NegativeStore().capacity(), 500);
}

TEST(NegativeStore, SavedAndLoadedForOneEmbeddingModel) {
    cv::RNG rng(10);
    const std::string file = testing::TempDir() + "/negatives_test.yml";
    NegativeStore store;
    for (int i = 0; i < 30; ++i) store.add(person(rng), "track " + std::to_string(i / 10));
    store.save(file, "test-model");

    NegativeStore loaded;
    ASSERT_TRUE(loaded.load(file, "test-model"));
    ASSERT_EQ(loaded.faces().size(), 30u);
    EXPECT_EQ(loaded.people(), 3);
    for (size_t i = 0; i < 30; ++i)
        EXPECT_NEAR(similarity(loaded.faces()[i], store.faces()[i]), 1, 1e-5);
    EXPECT_FALSE(loaded.load(file, "another-model")) << "faces of another model are no use";
    EXPECT_TRUE(loaded.faces().empty());
    fs::remove(file);
    EXPECT_FALSE(loaded.load(file, "test-model"));
}

// --- How the network changes recognition ------------------------------------

// combineMatch(): `probs` is the network's answer, a probability per label and then "other".
TEST(NetworkRule, AcceptsAFaceJustBelowTheThresholdWhenItsSure) {
    RecognizerParams p;
    LabelMatch m;  // the gallery: alex is closest, but below the threshold
    m.label = -1, m.best = 0, m.score = 0.30f, m.runnerUp = 0.1f;
    LabelVerdict v = combineMatch(m, 0.363f, {0.95f, 0.05f}, p);
    EXPECT_EQ(v.label, 0);
    EXPECT_TRUE(v.relaxed);
    EXPECT_EQ(combineMatch(m, 0.363f, {0.85f, 0.15f}, p).label, -1) << "not sure enough";
    m.score = 0.363f - 0.081f;
    EXPECT_EQ(combineMatch(m, 0.363f, {0.99f, 0.01f}, p).label, -1) << "too far below";
    m.score = 0.30f, m.runnerUp = 0.27f;
    EXPECT_EQ(combineMatch(m, 0.363f, {0.99f, 0.0f, 0.01f}, p).label, -1)
        << "a runner-up as close still means no match";
}

TEST(NetworkRule, RejectsAMatchItsSureIsSomeoneElse) {
    RecognizerParams p;
    LabelMatch m;
    m.label = m.best = 0, m.score = 0.5f, m.runnerUp = 0;
    LabelVerdict v = combineMatch(m, 0.363f, {0.05f, 0.95f}, p);
    EXPECT_EQ(v.label, -1);
    EXPECT_TRUE(v.vetoed);
    EXPECT_EQ(combineMatch(m, 0.363f, {0.2f, 0.8f}, p).label, 0) << "not sure enough";
}

TEST(NetworkRule, WithoutTheNetworkTheGalleryDecides) {
    RecognizerParams p;
    LabelMatch m;
    m.label = -1, m.best = 0, m.score = 0.35f;
    EXPECT_EQ(combineMatch(m, 0.363f, {}, p).label, -1);
    m.label = 0, m.score = 0.5f;
    const LabelVerdict v = combineMatch(m, 0.363f, {}, p);
    EXPECT_EQ(v.label, 0);
    EXPECT_FALSE(v.relaxed || v.vetoed);
}

// --- End to end: the recognizer, the reinforcer and the files --------------

TEST_F(ReinforcementTest, PhotographsARecognizedFaceAndRemembersIt) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    watchAlex(0, 10);
    ASSERT_NE(recognizer->labelName(1), nullptr);
    EXPECT_EQ(*recognizer->labelName(1), "alex");
    EXPECT_EQ(library.labels()[0].samples.size(), 8u)
        << "the photo's embedding and its mirrored copy are used right away";
    stop();

    const fs::path folder = fs::path(dir) / "alex";
    const std::vector<std::string> photos = autoPhotosIn(folder);
    ASSERT_EQ(photos.size(), 1u) << "one in 30 s";
    // The face with the recognizer's margin around it, as a JPEG.
    const cv::Mat image = cv::imread((folder / photos[0]).string());
    EXPECT_EQ(image.size(), cv::Size(256, 320));

    // The next start finds it in .embeddings.yml like any synced photo, so it
    // isn't looked at again.
    LabelLibrary again(dir);
    again.load();
    EXPECT_FALSE(again.needsSync());
    ASSERT_EQ(again.labels().size(), 1u);
    EXPECT_EQ(again.labels()[0].samples.size(), 8u);
    const auto& records = again.folders()[0].cache.photos;
    const auto r = std::find_if(records.begin(), records.end(),
                                [&](const PhotoRecord& r) { return r.file.name == photos[0]; });
    ASSERT_NE(r, records.end());
    EXPECT_TRUE(r->used);
    EXPECT_EQ(r->roi, cv::Rect(48, 60, 160, 200)) << "the face, in the photo";
    EXPECT_FALSE(r->mirrored.empty());
    float best = 0;
    for (int code = 1; code <= 20; ++code)
        best = std::max(best, similarity(r->embedding, embedder.faces[code]));
    EXPECT_NEAR(best, 1, 1e-5) << "the embedding of the view photographed";
}

TEST_F(ReinforcementTest, AtMostOnePhotoPerLabelEvery30Seconds) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    watchAlex(0, 59);
    stop();
    const std::vector<double> times = photoTimes();
    ASSERT_EQ(times.size(), 2u);
    EXPECT_GE(times[1] - times[0], 30);
    EXPECT_EQ(autoPhotosIn(fs::path(dir) / "alex").size(), 2u);
}

TEST_F(ReinforcementTest, NoPhotosOfAFaceItDoesntRecognize) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    for (int t = 0; t <= 60; ++t) step(t, frameWith(kStranger), {face(1, t)});
    EXPECT_EQ(recognizer->labelOf(1), -1);
    stop();
    EXPECT_EQ(imagesUnder(dir), 3) << "only alex's own photos";
}

TEST_F(ReinforcementTest, NoPhotosOfTheTargetOrALitFace) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    for (int t = 0; t <= 20; ++t) step(t, frameWith(t % 20 + 1), {face(1, t)}, {1});
    EXPECT_NE(recognizer->labelOf(1), -1) << "a lit face is still recognized";
    EXPECT_TRUE(photoTimes().empty());
    watchAlex(21, 30);
    stop();
    const std::vector<double> times = photoTimes();
    ASSERT_EQ(times.size(), 1u);
    EXPECT_GE(times[0], 22) << "not right after the light, while the camera adjusts";
}

TEST_F(ReinforcementTest, NoPhotosOfASmallOrBlurryFace) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    watchAlex(0, 15, cv::Rect(200, 100, 70, 90));
    EXPECT_NE(recognizer->labelOf(1), -1);
    stop();
    start();
    watchAlex(0, 15, kBox, true);
    EXPECT_NE(recognizer->labelOf(1), -1);
    stop();
    EXPECT_TRUE(photoTimes().empty());
    EXPECT_TRUE(autoPhotosIn(fs::path(dir) / "alex").empty());
}

TEST_F(ReinforcementTest, AutoPhotosCanBeTurnedOff) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    RecognizerParams off;
    off.autoPhotos = false;  // --no-auto-photos
    start(off);
    watchAlex(0, 15);
    EXPECT_NE(recognizer->labelOf(1), -1);
    stop();
    EXPECT_TRUE(autoPhotosIn(fs::path(dir) / "alex").empty());

    // Or for one label, in its label.yml.
    std::ofstream(fs::path(dir) / "alex" / kSettingsFile) << "auto_photos: false\n";
    start();
    ASSERT_FALSE(library.labels()[0].autoPhotos);
    watchAlex(0, 15);
    stop();
    EXPECT_TRUE(autoPhotosIn(fs::path(dir) / "alex").empty());
}

TEST_F(ReinforcementTest, AFullLabelReplacesItsMostRedundantAutoPhoto) {
    const Embedding mine = mix(alex, person(rng), 0.1f);
    const Embedding old = mix(alex, person(rng), 0.75f);
    const std::vector<std::string> names = {"me1.jpg", "me2.jpg", "auto_20260101_000000.jpg",
                                            "auto_20260101_000001.jpg"};
    makeLabel("alex", {mine, mix(mine, person(rng), 0.01f), old, mix(old, person(rng), 0.05f)},
              names);
    RecognizerParams params;
    params.capture.maxPhotos = 2;
    start(params);
    watchAlex(0, 10);
    EXPECT_EQ(library.labels()[0].samples.size(), 8u) << "one photo in, one out";
    stop();

    const fs::path folder = fs::path(dir) / "alex";
    EXPECT_TRUE(fs::exists(folder / "me1.jpg") && fs::exists(folder / "me2.jpg"))
        << "the user's own photos stay, however alike";
    const std::vector<std::string> autos = autoPhotosIn(folder);
    ASSERT_EQ(autos.size(), 2u);
    EXPECT_NE(fs::exists(folder / names[2]), fs::exists(folder / names[3]))
        << "one of the two alike automatic photos went";
    LabelLibrary again(dir);
    again.load();
    EXPECT_FALSE(again.needsSync()) << "the cache lists exactly the photos there";
    EXPECT_EQ(again.folders()[0].cache.photos.size(), 4u);
}

TEST_F(ReinforcementTest, OtherFacesAreKeptAsEmbeddingsOnly) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    for (int t = 0; t <= 30; ++t) step(t, frameWith(kStranger), {face(1, t)});
    step(31, frameWith(kStranger), {});  // the face is gone: its faces are handed over
    reinforcer->wait();
    EXPECT_EQ(reinforcer->others(), RecognizerParams{}.others.perTrack);
    stop();

    NegativeStore store;
    ASSERT_TRUE(store.load((fs::path(dir) / kNegativesFile).string(), embedder.modelId()));
    EXPECT_EQ(store.faces().size(), 10u);
    for (const Embedding& e : store.faces())
        EXPECT_NEAR(similarity(e, embedder.faces[kStranger]), 1, 1e-5);
    EXPECT_EQ(imagesUnder(dir), 3) << "no photos of them";
}

TEST_F(ReinforcementTest, NoOtherFacesFromAFaceThatMatchedALabelOrWasBrief) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    watchAlex(0, 5);
    // Then the same face stops looking like alex (turned away, say).
    for (int t = 6; t <= 30; ++t) step(t, frameWith(kStranger), {face(1, t)});
    // And a stranger seen only briefly.
    for (int t = 31; t <= 33; ++t) step(t, frameWith(kStranger), {face(2, t)});
    step(34, frameWith(kStranger), {});
    stop();
    NegativeStore store;
    store.load((fs::path(dir) / kNegativesFile).string(), embedder.modelId());
    EXPECT_TRUE(store.faces().empty());
}

TEST_F(ReinforcementTest, TheNetworkStaysOffUntilItHasSeenOtherPeople) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    start();
    EXPECT_EQ(reinforcer->net(), nullptr);
    EXPECT_EQ(reinforcer->status(), "off: needs other faces (0 of 20 seen)");
    EXPECT_EQ(recognizer->netStatus(), "off: needs other faces (0 of 20 seen)");
    watchAlex(0, 3);
    const TrackIdentity* id = recognizer->identity(1);
    ASSERT_NE(id, nullptr);
    EXPECT_LT(id->netProb, 0) << "no network, no probability";

    for (const std::vector<Embedding>& track : strangers(6, 4)) reinforcer->addOthers(track);
    reinforcer->wait();
    ASSERT_NE(reinforcer->net(), nullptr);
    EXPECT_TRUE(reinforcer->net()->on());
    EXPECT_EQ(reinforcer->status().rfind("on", 0), 0u) << reinforcer->status();
    watchAlex(4, 5);  // the recognizer picks the new network up
    EXPECT_EQ(recognizer->netStatus().rfind("on", 0), 0u);
    EXPECT_GE(recognizer->identity(1)->netProb, 0.9f);
    EXPECT_EQ(recognizer->identity(1)->netLabel, "alex");
    stop();
    EXPECT_TRUE(fs::exists(fs::path(dir) / kFaceNetFile));
    EXPECT_TRUE(fs::exists(fs::path(dir) / kNegativesFile));
}

TEST_F(ReinforcementTest, TheNetworkLetsAFaceJustBelowTheThresholdCount) {
    // Photos of alex from many angles: some only 0.3 like alex's typical look.
    std::vector<Embedding> photos;
    for (int i = 0; i < 20; ++i) photos.push_back(mix(alex, person(rng), 0.33f + 0.15f * i));
    makeLabel("alex", photos);
    library.load();
    // A view of alex that the photos alone score just below the threshold.
    const float threshold = embedder.defaultThreshold();
    Embedding borderline;
    for (float w = 1; w < 5 && borderline.empty(); w += 0.05f) {
        const Embedding e = mix(alex, person(rng), w);
        const float score = labelScore(library.labels()[0], e);
        if (score >= threshold - 0.07f && score < threshold - 0.01f) borderline = e;
    }
    ASSERT_FALSE(borderline.empty());
    embedder.faces[kBorderline] = borderline;

    // Without the network (no other faces yet), it's not recognized.
    start();
    for (int t = 0; t <= 8; ++t) step(t, frameWith(kBorderline), {face(1, t)});
    EXPECT_EQ(recognizer->labelOf(1), -1);
    stop();

    // With it, it is.
    saveOthers(strangers(10, 4));
    start();
    ASSERT_NE(reinforcer->net(), nullptr);
    ASSERT_GE(reinforcer->net()->predict(borderline)[0], 0.9f) << "the network is sure";
    for (int t = 0; t <= 8; ++t) step(t, frameWith(kBorderline), {face(2, t)});
    EXPECT_EQ(recognizer->labelOf(2), 0);
    stop();
}

TEST_F(ReinforcementTest, TheNetworkVetoesAMatchItLearnedIsSomeoneElse) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    library.load();
    const Embedding lookAlike = mix(alex, person(rng), 1.6f);
    ASSERT_GE(labelScore(library.labels()[0], lookAlike), embedder.defaultThreshold())
        << "the photos alone call it alex";
    embedder.faces[kLookAlike] = lookAlike;
    // (With photos on, the first run would photograph the look-alike for alex:
    // the photos alone call it alex clearly and steadily.)
    RecognizerParams noPhotos;
    noPhotos.autoPhotos = false;

    start(noPhotos);
    for (int t = 0; t <= 8; ++t) step(t, frameWith(kLookAlike), {face(1, t)});
    EXPECT_EQ(recognizer->labelOf(1), 0) << "no network yet";
    stop();

    std::vector<std::vector<Embedding>> seen;  // the network has seen this person, often
    for (int i = 0; i < 6; ++i) seen.push_back(views(lookAlike, 5, rng, 0.3f));
    saveOthers(seen);
    start(noPhotos);
    ASSERT_NE(reinforcer->net(), nullptr) << reinforcer->status();
    for (int t = 0; t <= 8; ++t) step(t, frameWith(kLookAlike), {face(2, t)});
    EXPECT_EQ(recognizer->labelOf(2), -1);
    const TrackIdentity* id = recognizer->identity(2);
    ASSERT_NE(id, nullptr);
    EXPECT_EQ(id->netLabel, "other");
    EXPECT_GE(id->netProb, 0.9f);
    stop();
}

TEST_F(ReinforcementTest, TheNetworkIsRetrainedWhenTheLabelsChange) {
    makeLabel("alex", views(alex, 6, rng, 0.3f));
    saveOthers(strangers(6, 4));
    const fs::path file = fs::path(dir) / kFaceNetFile;
    start();
    ASSERT_NE(reinforcer->net(), nullptr);
    EXPECT_EQ(reinforcer->net()->labels(), std::vector<std::string>{"alex"});
    stop();
    ASSERT_TRUE(fs::exists(file));
    const auto trained = fs::last_write_time(file);

    // Nothing changed: the saved network is used as it is.
    start();
    EXPECT_EQ(fs::last_write_time(file), trained);
    ASSERT_NE(reinforcer->net(), nullptr);
    stop();

    // A new label: retrained for both.
    makeLabel("sam", views(person(rng), 6, rng, 0.3f));
    start();
    ASSERT_NE(reinforcer->net(), nullptr);
    EXPECT_EQ(reinforcer->net()->labels(), (std::vector<std::string>{"alex", "sam"}));
    stop();
    FaceNet saved;
    ASSERT_TRUE(saved.load(file.string()));
    EXPECT_EQ(saved.labels(), (std::vector<std::string>{"alex", "sam"}));

    // A photo deleted: retrained without it.
    const fs::path photo = fs::path(dir) / "sam" / "photo_0.jpg";
    fs::remove(photo);
    LabelCache cache;
    readCache((fs::path(dir) / "sam" / kCacheFile).string(), cache);
    cache.photos.erase(cache.photos.begin());
    writeCache((fs::path(dir) / "sam" / kCacheFile).string(), cache);
    start();
    ASSERT_NE(reinforcer->net(), nullptr);
    EXPECT_EQ(reinforcer->net()->samples(), 22) << "6 + 5 photos, mirrored";
    stop();
}

TEST_F(ReinforcementTest, ANewPhotoRetrainsTheNetwork) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    saveOthers(strangers(6, 4));
    start();
    ASSERT_NE(reinforcer->net(), nullptr);
    EXPECT_EQ(reinforcer->net()->samples(), 6);
    watchAlex(0, 10);
    reinforcer->wait();
    ASSERT_NE(reinforcer->net(), nullptr);
    EXPECT_EQ(reinforcer->net()->samples(), 8) << "trained with the new photo";
    stop();
    FaceNet saved;
    ASSERT_TRUE(saved.load((fs::path(dir) / kFaceNetFile).string()));
    EXPECT_EQ(saved.samples(), 8);
}

// The --sync-labels and --list-labels path: trainFaceNet() and faceNetStatus() work from the saved
// files, with no Recognizer running.
TEST_F(ReinforcementTest, SyncLabelsTrainsTheNetworkAndSaysWhyItsOff) {
    makeLabel("alex", views(alex, 3, rng, 0.2f));
    library.load();
    EXPECT_EQ(trainFaceNet(library, embedder.modelId()), "off: needs other faces (0 of 20 seen)");
    EXPECT_FALSE(fs::exists(fs::path(dir) / kFaceNetFile));
    EXPECT_EQ(faceNetStatus(library, embedder.modelId()), "off: needs other faces (0 of 20 seen)");

    saveOthers(strangers(6, 4));
    EXPECT_EQ(faceNetStatus(library, embedder.modelId()),
              "not trained yet (it will be at the next start or --sync-labels)");
    const std::string status = trainFaceNet(library, embedder.modelId());
    EXPECT_EQ(status.rfind("on", 0), 0u) << status;
    EXPECT_TRUE(fs::exists(fs::path(dir) / kFaceNetFile));
    EXPECT_EQ(faceNetStatus(library, embedder.modelId()), status);
}
