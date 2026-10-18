#include "ClassRegistry.h"

#include <fstream>
#include <stdexcept>

namespace {

const std::string kUnknown = "unknown";

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

}  // namespace

ClassRegistry::ClassRegistry(const std::vector<std::string>& cocoNames) {
    for (const std::string& raw : cocoNames) {
        const std::string name = trim(raw);
        // Older names files listed hand and face as well; they're built in now.
        if (name.empty() || name == "hand" || name == "face") continue;
        add(name, ClassKind::Object, -1);
    }
    cocoCount_ = size();
    person_ = find("person");
    hand_ = add("hand", ClassKind::Hand, -1);
    face_ = add("face", ClassKind::Face, -1);
}

ClassRegistry ClassRegistry::fromFile(const std::string& cocoNamesPath) {
    std::ifstream in(cocoNamesPath);
    if (!in) throw std::runtime_error("Failed to open class names: " + cocoNamesPath);
    std::vector<std::string> names;
    for (std::string line; std::getline(in, line);) names.push_back(line);
    return ClassRegistry(names);
}

ClassId ClassRegistry::addCustom(const std::string& name, ClassId baseClass) {
    if (!valid(baseClass) || kind(baseClass) == ClassKind::Custom)
        throw std::invalid_argument("Custom label '" + name + "' needs a built-in base class");
    return add(name, ClassKind::Custom, baseClass);
}

ClassId ClassRegistry::add(const std::string& name, ClassKind kind, ClassId base) {
    if (find(name) >= 0) throw std::invalid_argument("Duplicate class name: " + name);
    const ClassId id = size();
    entries_.push_back({name, kind, base < 0 ? id : base});
    return id;
}

ClassId ClassRegistry::find(const std::string& name) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].name == name) return static_cast<ClassId>(i);
    return -1;
}

const std::string& ClassRegistry::name(ClassId id) const {
    return valid(id) ? entries_[id].name : kUnknown;
}

ClassKind ClassRegistry::kind(ClassId id) const {
    return valid(id) ? entries_[id].kind : ClassKind::Object;
}

ClassId ClassRegistry::base(ClassId id) const { return valid(id) ? entries_[id].base : id; }
