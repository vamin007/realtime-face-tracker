#pragma once
#include <string>
#include <vector>

#include "Detection.h"

enum class ClassKind { Object, Face, Custom };  // Object: a YOLO class (person)

// The one place class ids come from: person (id 0, YOLO's only class here)
// and face, then custom labels.
class ClassRegistry {
public:
    // The tracker's classes: person and face.
    ClassRegistry();

    // A custom label refines a built-in class, e.g. "me" refines "face".
    ClassId addCustom(const std::string& name, ClassId baseClass);

    ClassId find(const std::string& name) const;  // -1 if unknown
    const std::string& name(ClassId id) const;    // "unknown" for an invalid id
    ClassKind kind(ClassId id) const;
    ClassId base(ClassId id) const;  // a custom label's base class; any other class itself

    bool valid(ClassId id) const { return id >= 0 && id < size(); }
    int size() const { return static_cast<int>(entries_.size()); }

    ClassId person() const { return person_; }
    ClassId face() const { return face_; }

private:
    struct Entry {
        std::string name;
        ClassKind kind;
        ClassId base;
    };
    ClassId add(const std::string& name, ClassKind kind, ClassId base);

    std::vector<Entry> entries_;
    ClassId person_ = -1;
    ClassId face_ = -1;
};
