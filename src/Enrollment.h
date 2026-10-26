#pragma once
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Detector.h"
#include "Embedder.h"
#include "Labels.h"

// Brings every label folder up to date with its photos: photos added or
// changed since the last sync are looked at (the face found with the face
// detector and embedded, plus a mirrored copy), removed ones are dropped, and
// new folders become labels. Prints what changed, and warns about unusable
// photos, labels with few usable photos, photos that don't look like the rest
// and labels that look alike; `all` reports on unchanged labels as well.
// Reloads the library. Returns the number of labels that changed.
int syncLabels(LabelLibrary& library, Detector& detector, const ClassRegistry& classes,
               IEmbedder& embedder, bool all);

// The detectors a sync needs that aren't among `loaded` (Detector::names()). A
// sync looks for faces in the photos, so it needs the face detector: without
// it, photos would be recorded as showing no face, and not looked at again
// once it's back.
std::vector<std::string> missingForSync(const std::vector<std::string>& loaded);

// Teaches the tracker a person's face from photos: copies them into
// <labels>/<name>/ and syncs. Enrolling an existing name adds photos to it;
// with `mustExist` the label has to exist.
void enroll(const std::string& name, const std::vector<std::string>& images, Detector& detector,
            const ClassRegistry& classes, IEmbedder& embedder, LabelLibrary& library,
            bool mustExist = false);

// Prints each label: its photos, usable photos and changes not synced yet.
void listLabels(const LabelLibrary& library);
// Renames a label's folder.
void renameLabel(const std::string& from, const std::string& to, const LabelLibrary& library,
                 const ClassRegistry& classes);
// Moves a label's folder to <labels>/.removed/, photos and all. Returns where it went.
std::string removeLabel(const std::string& name, const LabelLibrary& library);
