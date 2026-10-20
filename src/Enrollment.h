#pragma once
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Detector.h"
#include "Labels.h"
#include "Recognizer.h"

// Brings every label folder up to date with its photos: photos added or
// changed since the last sync are looked at (the item found with the detectors
// and embedded, plus a mirrored copy), removed ones are dropped, and new
// folders become labels. A label's class comes from its label.yml, or else is
// the one most prominent across its photos (a face rather than the whole
// person), and stays once chosen. Prints what changed, and warns about unusable
// photos, labels with few usable photos, photos that don't look like the rest
// and labels that look alike; `all` reports on unchanged labels as well.
// Reloads the library. Returns the number of labels that changed.
int syncLabels(LabelLibrary& library, Detector& detector, const ClassRegistry& classes,
               Embedders& embedders, bool all);

// Teaches the tracker a specific item from photos: copies them into
// <labels>/<name>/ and syncs. `classHint` (e.g. "cup") and `wholePhoto` (each
// photo shows just the item) are saved in the label's label.yml. Enrolling an
// existing name adds photos to it; with `mustExist` the label has to exist.
void enroll(const std::string& name, const std::string& classHint,
            const std::vector<std::string>& images, bool wholePhoto, Detector& detector,
            const ClassRegistry& classes, Embedders& embedders, LabelLibrary& library,
            bool mustExist = false);

// Prints each label: its class, photos, usable photos and changes not synced yet.
void listLabels(const LabelLibrary& library);
// Renames a label's folder.
void renameLabel(const std::string& from, const std::string& to, const LabelLibrary& library,
                 const ClassRegistry& classes);
// Moves a label's folder to <labels>/.removed/, photos and all. Returns where it went.
std::string removeLabel(const std::string& name, const LabelLibrary& library);
