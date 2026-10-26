#pragma once
#include <string>
#include <vector>

#include "Detection.h"

enum class ClassKind { Object, Hand, Face, Custom };  // Object: a YOLO class (person)

// The one place class ids come from: YOLO's classes first (just person, id 0,
// unless other names are given), then the built-in hand and face classes, then
// custom labels.
class ClassRegistry {
public:
    // The tracker's classes: person, hand and face.
    ClassRegistry();
    // YOLO's classes named here, in its output order, instead of just person
    // (tests use stand-ins; fromFile reads a names file, one name per line).
    explicit ClassRegistry(const std::vector<std::string>& cocoNames);
    static ClassRegistry fromFile(const std::string& cocoNamesPath);

    // A custom label refines a built-in class, e.g. "me" refines "face".
    ClassId addCustom(const std::string& name, ClassId baseClass);

    ClassId find(const std::string& name) const;  // -1 if unknown
    const std::string& name(ClassId id) const;    // "unknown" for an invalid id
    ClassKind kind(ClassId id) const;
    ClassId base(ClassId id) const;  // a custom label's base class; any other class itself

    bool valid(ClassId id) const { return id >= 0 && id < size(); }
    int size() const { return static_cast<int>(entries_.size()); }
    int cocoCount() const { return cocoCount_; }  // YOLO's classes listed here

    ClassId person() const { return person_; }
    ClassId hand() const { return hand_; }
    ClassId face() const { return face_; }

private:
    struct Entry {
        std::string name;
        ClassKind kind;
        ClassId base;
    };
    ClassId add(const std::string& name, ClassKind kind, ClassId base);

    std::vector<Entry> entries_;
    int cocoCount_ = 0;
    ClassId person_ = -1;
    ClassId hand_ = -1;
    ClassId face_ = -1;
};
