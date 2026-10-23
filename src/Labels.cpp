#include "Labels.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

constexpr int kTopSamples = 3;
constexpr int kCacheFormat = 2;  // format 1 kept the embeddings in label.yml

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

long long mtimeOf(const fs::path& path) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               fs::last_write_time(path).time_since_epoch())
        .count();
}

Embedding readEmbedding(const cv::FileNode& node) {
    if (node.empty()) return {};
    cv::Mat m;
    node >> m;
    if (m.empty() || m.type() != CV_32F) return {};
    return Embedding(m.begin<float>(), m.end<float>());
}

// A yes/no setting: true/false, yes/no, on/off or 1/0.
bool readFlag(const cv::FileNode& node, const std::string& file) {
    if (node.isInt()) return static_cast<int>(node) != 0;
    if (node.isString()) {
        const std::string v = lower(trim(static_cast<std::string>(node)));
        if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
        if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    }
    throw std::runtime_error(file + ": " + node.name() + " should be true or false");
}

}  // namespace

std::string labelNameProblem(const std::string& name) {
    auto allowed = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
    };
    if (name.empty() || name.size() > 40 || !std::all_of(name.begin(), name.end(), allowed))
        return "label names may only use letters, digits, _ and -";
    // Labeled tracks show the label's name, and a TARGET tag would pass for the target.
    if (lower(name) == "target") return "TARGET is reserved for the red TARGET box";
    return "";
}

bool validLabelName(const std::string& name) { return labelNameProblem(name).empty(); }

float labelScore(const CustomLabel& label, const Embedding& e) {
    std::vector<float> sims;
    for (const LabelSample& s : label.samples) sims.push_back(similarity(s.embedding, e));
    if (sims.empty()) return 0;
    const size_t k = std::min<size_t>(kTopSamples, sims.size());
    std::partial_sort(sims.begin(), sims.begin() + k, sims.end(), std::greater<float>());
    float sum = 0;
    for (size_t i = 0; i < k; ++i) sum += sims[i];
    return sum / k;
}

bool isPhotoName(const std::string& fileName) {
    const std::string ext = lower(fs::path(fileName).extension().string());
    // HEIC and AVIF count as photos so that a sync can say they can't be read.
    for (const char* known : {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif",
                              ".heic", ".heif", ".avif"})
        if (ext == known) return true;
    return false;
}

std::vector<PhotoFile> listPhotos(const std::string& folder) {
    std::vector<PhotoFile> photos;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(folder, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.empty() || name[0] == '.' || !entry.is_regular_file() || !isPhotoName(name))
            continue;
        photos.push_back({name, static_cast<long long>(entry.file_size()), mtimeOf(entry.path())});
    }
    std::sort(photos.begin(), photos.end(),
              [](const PhotoFile& a, const PhotoFile& b) { return a.name < b.name; });
    return photos;
}

LabelSettings readSettings(const std::string& file) {
    LabelSettings s;
    std::ifstream in(file);
    if (!in) return s;
    std::stringstream text;
    text << in.rdbuf();
    // A file of nothing but comments sets nothing (OpenCV won't parse it).
    std::stringstream lines(text.str());
    bool hasSettings = false;
    for (std::string line; std::getline(lines, line);) {
        line = trim(line);
        if (!line.empty() && line[0] != '#' && line[0] != '%' && line != "---") hasSettings = true;
    }
    if (!hasSettings) return s;

    cv::FileStorage yaml;
    try {
        yaml.open(text.str(),
                  cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_YAML);
    } catch (const cv::Exception&) {
    }
    if (!yaml.isOpened() || !yaml.root().isMap())
        throw std::runtime_error(file +
                                 " isn't readable; write one setting per line, e.g. class: cup");
    for (const cv::FileNode& node : yaml.root()) {
        const std::string key = node.name();
        if (key == "class") {
            if (!node.isString())
                throw std::runtime_error(file + ": class should be a class name, e.g. cup");
            s.baseClass = trim(static_cast<std::string>(node));
        } else if (key == "whole_photo") {
            s.wholePhoto = readFlag(node, file);
        } else if (key == "threshold") {
            const float t = node.isReal() || node.isInt() ? static_cast<float>(node) : -1.0f;
            if (t < 0 || t > 1)
                throw std::runtime_error(file + ": threshold should be a number from 0 to 1");
            s.threshold = t;
        } else {
            throw std::runtime_error(file + ": unknown setting '" + key +
                                     "' (it can set class, whole_photo and threshold)");
        }
    }
    return s;
}

void writeSettings(const std::string& file, const LabelSettings& settings) {
    cv::FileStorage out(file, cv::FileStorage::WRITE);
    if (!out.isOpened()) throw std::runtime_error("Can't write " + file);
    out.writeComment("Settings for this label. Its photos are the image files in this folder.");
    out.writeComment("class: what the item is (tracker --list-classes); without it, the class");
    out.writeComment("       that stands out in the photos is chosen");
    out.writeComment("whole_photo: true if each photo shows just the item (close-ups)");
    out.writeComment("threshold: similarity needed to recognize it, 0-1; 0 = the model's default");
    if (!settings.baseClass.empty()) out << "class" << settings.baseClass;
    out << "whole_photo" << (settings.wholePhoto ? "true" : "false");
    out << "threshold" << settings.threshold;
}

bool readCache(const std::string& file, LabelCache& cache) {
    cache = {};
    if (!fs::exists(file)) return false;
    try {
        cv::FileStorage in(file, cv::FileStorage::READ);
        if (!in.isOpened() || static_cast<int>(in["format"]) != kCacheFormat) return false;
        in["model"] >> cache.model;
        in["class"] >> cache.baseClass;
        cache.autoClass = static_cast<int>(in["class_auto"]) != 0;
        cache.wholePhoto = static_cast<int>(in["whole_photo"]) != 0;
        for (const cv::FileNode& node : in["photos"]) {
            PhotoRecord r;
            int64_t size = 0, mtime = 0;
            node["file"] >> r.file.name;
            node["size"] >> size;
            node["mtime"] >> mtime;
            r.file.size = size;
            r.file.mtime = mtime;
            r.used = static_cast<int>(node["used"]) != 0;
            node["note"] >> r.note;
            if (r.used) {
                node["roi"] >> r.roi;
                r.conf = static_cast<float>(node["conf"]);
                r.embedding = readEmbedding(node["embedding"]);
                r.mirrored = readEmbedding(node["mirrored"]);
                r.used = !r.embedding.empty();
            }
            cache.photos.push_back(std::move(r));
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Warning: can't read " << file << "; its photos will be looked at again\n";
        cache = {};
        return false;
    }
    return true;
}

void writeCache(const std::string& file, const LabelCache& cache) {
    // Written next to it, then renamed over it: an interrupted sync never
    // leaves half a file. (The .yml extension tells OpenCV to write YAML.)
    const std::string temp = file + ".tmp.yml";
    {
        cv::FileStorage out(temp, cv::FileStorage::WRITE);
        if (!out.isOpened()) throw std::runtime_error("Can't write " + temp);
        out.writeComment("Written by the tracker: what it found in each photo. Delete it to");
        out.writeComment("have every photo looked at again.");
        out << "format" << kCacheFormat << "model" << cache.model << "class" << cache.baseClass
            << "class_auto" << (cache.autoClass ? 1 : 0) << "whole_photo"
            << (cache.wholePhoto ? 1 : 0);
        out << "photos" << "[";
        for (const PhotoRecord& r : cache.photos) {
            out << "{" << "file" << r.file.name << "size" << static_cast<int64_t>(r.file.size)
                << "mtime" << static_cast<int64_t>(r.file.mtime) << "used" << (r.used ? 1 : 0)
                << "note" << r.note;
            if (r.used) {
                out << "roi" << r.roi << "conf" << r.conf << "embedding"
                    << cv::Mat(r.embedding, false);
                if (!r.mirrored.empty()) out << "mirrored" << cv::Mat(r.mirrored, false);
            }
            out << "}";
        }
        out << "]";
    }
    fs::rename(temp, file);
}

SyncPlan planSync(const LabelSettings& settings, const LabelCache& cache,
                  const std::vector<PhotoFile>& photos, const std::string& model) {
    SyncPlan plan;
    if (!settings.baseClass.empty() && settings.baseClass != cache.baseClass) {
        plan.redoAll = true;
        if (!cache.baseClass.empty())
            plan.reason = "its class changed from " + cache.baseClass + " to " + settings.baseClass;
    } else if (settings.baseClass.empty() && !cache.baseClass.empty() && !cache.autoClass) {
        plan.redoAll = true;
        plan.reason = "label.yml no longer sets its class";
    } else if (settings.wholePhoto != cache.wholePhoto) {
        plan.redoAll = true;
        plan.reason =
            settings.wholePhoto ? "whole_photo was turned on" : "whole_photo was turned off";
    } else if (!model.empty() && !cache.model.empty() && cache.model != model) {
        plan.redoAll = true;
        plan.reason = "the embedding model changed";
    }

    for (const PhotoRecord& r : cache.photos) {
        const bool present = std::any_of(photos.begin(), photos.end(),
                                         [&](const PhotoFile& p) { return p.name == r.file.name; });
        if (!present) plan.removed.push_back(r.file.name);
    }
    for (const PhotoFile& p : photos) {
        const auto r = std::find_if(cache.photos.begin(), cache.photos.end(),
                                    [&](const PhotoRecord& r) { return r.file.name == p.name; });
        if (r == cache.photos.end()) {
            plan.look.push_back(p);
        } else if (r->file.size != p.size || r->file.mtime != p.mtime) {
            plan.look.push_back(p);
            plan.changed.push_back(p.name);
        } else {
            plan.keep.push_back(*r);
        }
    }
    // Without a class there is nothing to keep: when the photos change, the
    // class is chosen again from all of them.
    if (cache.baseClass.empty() && (!plan.look.empty() || !plan.removed.empty()))
        plan.redoAll = true;
    if (plan.redoAll) {
        plan.look = photos;
        plan.keep.clear();
    }
    return plan;
}

int LabelFolder::usedPhotos() const {
    return static_cast<int>(std::count_if(cache.photos.begin(), cache.photos.end(),
                                          [](const PhotoRecord& r) { return r.used; }));
}

bool migrateOldLabel(const std::string& folder) {
    const fs::path dir(folder), file = dir / kSettingsFile;
    if (!fs::exists(file)) return false;
    CustomLabel label;
    try {
        cv::FileStorage in(file.string(), cv::FileStorage::READ);
        if (!in.isOpened() || in["samples"].empty()) return false;
        in["base_class"] >> label.baseClass;
        in["model"] >> label.model;
        in["threshold"] >> label.threshold;
        for (const cv::FileNode& node : in["samples"]) {
            LabelSample s;
            node["image"] >> s.image;
            node["roi"] >> s.roi;
            s.flipped = static_cast<int>(node["flipped"]) != 0;
            s.embedding = readEmbedding(node["embedding"]);
            label.samples.push_back(std::move(s));
        }
    } catch (const cv::Exception&) {
        return false;  // not the old format; reading the settings reports what's wrong
    }
    label.name = dir.filename().string();
    // --whole-photo stored the whole photo as the item's box.
    label.wholePhoto = !label.samples.empty();
    for (const LabelSample& s : label.samples) {
        if (s.flipped || !label.wholePhoto) continue;
        cv::Mat image;
        if (s.roi.x == 0 && s.roi.y == 0) image = cv::imread((dir / s.image).string());
        label.wholePhoto = !image.empty() && s.roi == cv::Rect(0, 0, image.cols, image.rows);
    }
    fs::copy_file(file, dir / kOldLabelBackup, fs::copy_options::overwrite_existing);
    LabelLibrary(dir.parent_path().string()).save(label);
    std::cout << "Converted label '" << label.name << "' to the folder format (the old "
              << kSettingsFile << " is kept as " << (dir / kOldLabelBackup).string() << ")\n";
    return true;
}

void LabelLibrary::load() {
    labels_.clear();
    folders_.clear();
    loosePhotos_.clear();
    if (!fs::exists(dir_)) return;
    std::vector<fs::path> dirs;
    for (const auto& entry : fs::directory_iterator(dir_)) {
        const std::string name = entry.path().filename().string();
        if (name.empty() || name[0] == '.') continue;  // .removed/ and other hidden ones
        if (entry.is_directory()) dirs.push_back(entry.path());
        else if (isPhotoName(name)) loosePhotos_.push_back(name);
    }
    std::sort(dirs.begin(), dirs.end());
    std::sort(loosePhotos_.begin(), loosePhotos_.end());
    for (const fs::path& path : dirs) {
        LabelFolder f;
        f.name = path.filename().string();
        f.path = path.string();
        try {
            if (const std::string why = labelNameProblem(f.name); !why.empty())
                throw std::runtime_error(why + "; rename the folder");
            migrateOldLabel(f.path);
            f.settings = readSettings((path / kSettingsFile).string());
        } catch (const std::exception& e) {
            f.problem = e.what();
            folders_.push_back(std::move(f));
            continue;
        }
        readCache((path / kCacheFile).string(), f.cache);
        f.photos = listPhotos(f.path);
        f.plan = planSync(f.settings, f.cache, f.photos, "");

        // The label as the last sync left it.
        if (!f.cache.baseClass.empty()) {
            CustomLabel label{f.name, f.cache.baseClass, f.cache.model, f.settings.threshold,
                              {},     f.cache.wholePhoto};
            for (const PhotoRecord& r : f.cache.photos) {
                if (!r.used) continue;
                label.samples.push_back({r.file.name, r.roi, false, r.embedding});
                if (!r.mirrored.empty())
                    label.samples.push_back({r.file.name, r.roi, true, r.mirrored});
            }
            if (!label.samples.empty()) labels_.push_back(std::move(label));
        }
        folders_.push_back(std::move(f));
    }
}

void LabelLibrary::save(const CustomLabel& label) const {
    const fs::path path = folder(label.name);
    fs::create_directories(path);
    LabelCache cache{label.model, label.baseClass, false, label.wholePhoto, {}};
    for (const LabelSample& s : label.samples) {
        auto r = std::find_if(cache.photos.begin(), cache.photos.end(),
                              [&](const PhotoRecord& r) { return r.file.name == s.image; });
        if (r == cache.photos.end()) {
            PhotoRecord record;
            record.file.name = s.image;
            const fs::path photo = path / s.image;
            if (fs::exists(photo)) {
                record.file.size = static_cast<long long>(fs::file_size(photo));
                record.file.mtime = mtimeOf(photo);
            }
            record.used = true;
            record.note = "kept from before";
            record.roi = s.roi;
            cache.photos.push_back(std::move(record));
            r = cache.photos.end() - 1;
        }
        if (s.flipped) {
            r->mirrored = s.embedding;
        } else {
            r->embedding = s.embedding;
            r->roi = s.roi;
        }
    }
    writeCache((path / kCacheFile).string(), cache);
    writeSettings((path / kSettingsFile).string(),
                  {label.baseClass, label.wholePhoto, label.threshold});
}

int LabelLibrary::find(const std::string& name) const {
    for (size_t i = 0; i < labels_.size(); ++i)
        if (labels_[i].name == name) return static_cast<int>(i);
    return -1;
}

int LabelLibrary::findFolder(const std::string& name) const {
    for (size_t i = 0; i < folders_.size(); ++i)
        if (folders_[i].name == name) return static_cast<int>(i);
    return -1;
}

bool LabelLibrary::needsSync() const {
    return std::any_of(folders_.begin(), folders_.end(),
                       [](const LabelFolder& f) { return f.problem.empty() && !f.plan.empty(); });
}

bool LabelLibrary::hasLabelsFor(const std::string& baseClass) const {
    return std::any_of(labels_.begin(), labels_.end(),
                       [&](const CustomLabel& l) { return l.baseClass == baseClass; });
}

LabelMatch LabelLibrary::match(const std::string& baseClass, const Embedding& e,
                               float defaultThreshold, float minMargin) const {
    LabelMatch best;
    float bestScore = -1;
    for (size_t i = 0; i < labels_.size(); ++i) {
        if (labels_[i].baseClass != baseClass) continue;
        const float score = labelScore(labels_[i], e);
        if (score > bestScore) {
            best.runnerUp = std::max(bestScore, 0.0f);
            bestScore = score;
            best.label = static_cast<int>(i);
        } else {
            best.runnerUp = std::max(best.runnerUp, score);
        }
    }
    if (best.label < 0) return {};
    best.score = bestScore;
    const CustomLabel& label = labels_[best.label];
    const float threshold = label.threshold > 0 ? label.threshold : defaultThreshold;
    if (best.score < threshold || best.score - best.runnerUp < minMargin) best.label = -1;
    return best;
}

std::vector<PhotoAgreement> photoAgreement(const CustomLabel& label) {
    std::vector<PhotoAgreement> result;
    for (const LabelSample& s : label.samples) {
        if (s.flipped) continue;
        float sum = 0;
        int n = 0;
        for (const LabelSample& other : label.samples) {
            if (other.image == s.image) continue;
            sum += similarity(s.embedding, other.embedding);
            ++n;
        }
        result.push_back({s.image, n ? sum / n : 1});
    }
    return result;
}

LabelConfusion labelConfusion(const CustomLabel& a, const CustomLabel& b, float threshold,
                              float margin) {
    LabelConfusion c;
    for (const LabelSample& s : a.samples) {
        if (s.flipped) continue;
        // A photo stands in for a live view of the item, so it's compared with
        // the label's other photos only.
        CustomLabel others = a;
        others.samples.erase(
            std::remove_if(others.samples.begin(), others.samples.end(),
                           [&](const LabelSample& o) { return o.image == s.image; }),
            others.samples.end());
        const float own = labelScore(others, s.embedding);
        const float other = labelScore(b, s.embedding);
        ++c.photos;
        c.ownScore += own;
        c.otherScore += other;
        if (other >= threshold && own - other < margin) ++c.ambiguous;
    }
    if (c.photos > 0) {
        c.ownScore /= c.photos;
        c.otherScore /= c.photos;
    }
    return c;
}
