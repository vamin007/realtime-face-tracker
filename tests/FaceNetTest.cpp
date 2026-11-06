// The face network on synthetic embeddings: clusters of unit vectors stand in
// for the SFace embeddings of different people (nearly orthogonal in 128-d).
// Covers: staying off with too little data, telling the labels from each other and from
// strangers, repeatable training, saving and loading, and noticing a stale network.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/core.hpp>

#include "FaceNet.h"

namespace fs = std::filesystem;

namespace {

constexpr int kDim = 128;

Embedding normalized(Embedding e) {
    float n = 0;
    for (float x : e) n += x * x;
    for (float& x : e) x /= std::sqrt(n);
    return e;
}

// Someone's face: a random direction.
Embedding person(cv::RNG& rng) {
    Embedding e(kDim);
    for (float& x : e) x = static_cast<float>(rng.gaussian(1));
    return normalized(e);
}

// n views of a person: their direction plus noise. spread 0.6 puts views about
// 0.86 from the person and 0.74 from each other, like photos of one face.
std::vector<Embedding> views(const Embedding& center, int n, cv::RNG& rng, float spread = 0.6f) {
    std::vector<Embedding> out;
    for (int i = 0; i < n; ++i) {
        Embedding e = center;
        for (float& x : e) x += static_cast<float>(rng.gaussian(spread / std::sqrt(kDim)));
        out.push_back(normalized(e));
    }
    return out;
}

// alex and sam, 20 samples each, and 40 faces of 10 other people (each seen once).
struct Scene {
    cv::RNG rng{12345};
    Embedding alex = person(rng), sam = person(rng);
    std::vector<Embedding> strangers;
    FaceNetData data;

    Scene() {
        data.model = "test-model";
        data.labels = {"alex", "sam"};
        data.samples = {views(alex, 20, rng), views(sam, 20, rng)};
        for (int i = 0; i < 10; ++i) {
            strangers.push_back(person(rng));
            for (const Embedding& e : views(strangers.back(), 4, rng)) data.others.push_back(e);
        }
        data.otherPeople = 10;
    }
};

}  // namespace

TEST(FaceNet, StaysOffWithoutFacesOfOtherPeople) {
    Scene s;
    s.data.others.clear();  // only the labeled people were ever seen
    EXPECT_EQ(FaceNet::shortfall(s.data), "needs other faces (0 of 20 seen)");
    FaceNet net;
    EXPECT_FALSE(net.train(s.data));
    EXPECT_FALSE(net.on());
    EXPECT_TRUE(net.predict(s.data.samples[0][0]).empty());

    s.data.others.resize(19);
    EXPECT_EQ(FaceNet::shortfall(s.data), "needs other faces (19 of 20 seen)");
    s.data.others.resize(20);
    EXPECT_EQ(FaceNet::shortfall(s.data), "");

    // Many faces of one or two people teach it too little about everyone else.
    s.data.otherPeople = 2;
    EXPECT_EQ(FaceNet::shortfall(s.data), "needs other faces (20 seen, but from 2 of 5 people)");
    EXPECT_FALSE(net.train(s.data));
}

TEST(FaceNet, StaysOffWhileALabelHasTooFewSamples) {
    Scene s;
    s.data.samples[1].resize(4);  // sam: 2 photos, each mirrored
    EXPECT_EQ(FaceNet::shortfall(s.data), "needs 3 usable photos of sam");
    FaceNet net;
    EXPECT_FALSE(net.train(s.data));
    EXPECT_FALSE(net.on());

    FaceNetData none = s.data;
    none.labels.clear();
    none.samples.clear();
    EXPECT_EQ(FaceNet::shortfall(none), "needs a label");
}

TEST(FaceNet, TellsTheLabelsApartAndFromOtherPeople) {
    Scene s;
    FaceNet net;
    ASSERT_TRUE(net.train(s.data));
    ASSERT_TRUE(net.on());
    EXPECT_EQ(net.labels(), s.data.labels);
    EXPECT_EQ(net.model(), "test-model");
    EXPECT_EQ(net.samples(), 40);
    EXPECT_EQ(net.others(), 40);

    // New views, not trained on: alex, sam, and people seen among the others.
    for (const Embedding& e : views(s.alex, 10, s.rng)) {
        const std::vector<float> p = net.predict(e);
        ASSERT_EQ(p.size(), 3u);  // alex, sam, other
        EXPECT_NEAR(std::accumulate(p.begin(), p.end(), 0.0f), 1, 1e-4);
        EXPECT_GE(p[0], 0.9f);
    }
    for (const Embedding& e : views(s.sam, 10, s.rng)) EXPECT_GE(net.predict(e)[1], 0.9f);
    // Averaging three networks keeps it from being sure too easily, either way.
    int sure = 0;
    for (const Embedding& stranger : s.strangers) {
        const float other = net.predict(views(stranger, 1, s.rng)[0])[2];
        EXPECT_GT(other, 0.5f);
        sure += other >= 0.9f;
    }
    EXPECT_GE(sure, 9);
}

TEST(FaceNet, ANewPersonIsntTakenForALabel) {
    Scene s;
    FaceNet net;
    ASSERT_TRUE(net.train(s.data));
    int sure = 0;
    for (int i = 0; i < 20; ++i) {
        const std::vector<float> p = net.predict(person(s.rng));
        sure += p[0] >= 0.9f || p[1] >= 0.9f;
    }
    EXPECT_EQ(sure, 0) << "never seen, so never confidently alex or sam";
}

// Training is seeded, so the same data gives the same networks and the same answers.
TEST(FaceNet, TrainingIsRepeatable) {
    Scene s;
    FaceNet a, b;
    ASSERT_TRUE(a.train(s.data));
    ASSERT_TRUE(b.train(s.data));
    const Embedding probe = person(s.rng);
    EXPECT_EQ(a.predict(probe), b.predict(probe));
}

TEST(FaceNet, SavedAndLoadedItPredictsTheSame) {
    Scene s;
    FaceNet net;
    ASSERT_TRUE(net.train(s.data));
    const std::string file = testing::TempDir() + "/face_net_test.yml";
    net.save(file);

    FaceNet loaded;
    ASSERT_TRUE(loaded.load(file));
    EXPECT_TRUE(loaded.on());
    EXPECT_EQ(loaded.labels(), net.labels());
    EXPECT_EQ(loaded.model(), "test-model");
    EXPECT_EQ(loaded.samples(), 40);
    EXPECT_EQ(loaded.others(), 40);
    EXPECT_EQ(loaded.otherPeople(), 10);
    EXPECT_TRUE(loaded.trainedOn(s.data));
    for (const Embedding& e : {s.alex, s.sam, s.strangers[0]}) {
        const std::vector<float> p = net.predict(e), q = loaded.predict(e);
        ASSERT_EQ(p.size(), q.size());
        for (size_t i = 0; i < p.size(); ++i) EXPECT_NEAR(p[i], q[i], 1e-5);
    }

    // A damaged file loads as "off" rather than throwing.
    std::ofstream(file) << "not: [a network";
    EXPECT_FALSE(loaded.load(file));
    EXPECT_FALSE(loaded.on());
    fs::remove(file);
    EXPECT_FALSE(loaded.load(file)) << "no file";
}

// trainedOn(): exactly this data (by fingerprint). fits(): the same labels and model, so its
// outputs still mean something even if a photo was added since.
TEST(FaceNet, KnowsWhetherItWasTrainedOnTheCurrentLabels) {
    Scene s;
    FaceNet net;
    ASSERT_TRUE(net.train(s.data));
    EXPECT_TRUE(net.trainedOn(s.data));
    EXPECT_TRUE(net.fits({"alex", "sam"}, "test-model"));

    FaceNetData renamed = s.data;
    renamed.labels[1] = "samuel";
    EXPECT_FALSE(net.trainedOn(renamed));
    EXPECT_FALSE(net.fits(renamed.labels, "test-model"));

    FaceNetData model = s.data;
    model.model = "other-model";
    EXPECT_FALSE(net.trainedOn(model));
    EXPECT_FALSE(net.fits(s.data.labels, "other-model"));

    // A photo added (or one deleted): the same labels, but not the same samples.
    FaceNetData photo = s.data;
    photo.samples[0].push_back(views(s.alex, 1, s.rng)[0]);
    EXPECT_FALSE(net.trainedOn(photo));
    EXPECT_TRUE(net.fits(photo.labels, "test-model")) << "still usable until it's retrained";
    FaceNetData others = s.data;
    others.others.pop_back();
    EXPECT_FALSE(net.trainedOn(others));
}
