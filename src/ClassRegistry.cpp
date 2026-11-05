#include "ClassRegistry.h"

#include <stdexcept>

namespace {

const std::string kUnknown = "unknown";

}  // namespace

// Of YOLO's 80 COCO classes only person is tracked; YoloDetector skips the rest.
ClassRegistry::ClassRegistry() {
    person_ = add("person", ClassKind::Object, -1);
    face_ = add("face", ClassKind::Face, -1);
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
