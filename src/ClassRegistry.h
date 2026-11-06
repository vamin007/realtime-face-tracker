// The table of class names and ids shared by the whole pipeline: detectors tag detections with
// a ClassId, the tracker numbers tracks per class, the overlay and CSV log print the names, and
// --classes is parsed against it. There are two built-in classes (person from YOLO, face from
// YuNet); each person the user taught the tracker to recognize (a "custom label", e.g. "alex")
// is added at startup as a third kind of class that refines "face". Having one registry avoids
// hard-coded ids and name strings scattered across files.
#pragma once
#include <string>
#include <vector>

#include "Detection.h"

// Face is the face detector's class; Custom is a recognized person's label, always layered on
// a built-in class.
enum class ClassKind { Object, Face, Custom };  // Object: a YOLO class (person)

// The one place class ids come from: person (id 0, YOLO's only class here)
// and face, then custom labels.
// Ids are indices, assigned in order and never reused, so they stay valid for a whole run.
class ClassRegistry {
public:
    // The tracker's classes: person and face.
    ClassRegistry();

    // A custom label refines a built-in class, e.g. "me" refines "face".
    // Returns the new class's id. Throws std::invalid_argument if the name is taken or
    // baseClass is not a valid built-in class.
    ClassId addCustom(const std::string& name, ClassId baseClass);

    ClassId find(const std::string& name) const;  // -1 if unknown
    const std::string& name(ClassId id) const;    // "unknown" for an invalid id
    ClassKind kind(ClassId id) const;
    ClassId base(ClassId id) const;  // a custom label's base class; any other class itself

    bool valid(ClassId id) const { return id >= 0 && id < size(); }
    int size() const { return static_cast<int>(entries_.size()); }

    // The two built-ins. person() equals kCocoPerson (YoloDetector.h), so the detector's output
    // needs no translation.
    ClassId person() const { return person_; }
    ClassId face() const { return face_; }

private:
    // One class: its display name, what kind it is, and the class it refines (itself for
    // built-ins).
    struct Entry {
        std::string name;
        ClassKind kind;
        ClassId base;
    };
    // Shared by the constructor and addCustom(); rejects duplicate names.
    ClassId add(const std::string& name, ClassKind kind, ClassId base);

    std::vector<Entry> entries_;
    ClassId person_ = -1;
    ClassId face_ = -1;
};
