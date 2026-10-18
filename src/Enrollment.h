#pragma once
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Detector.h"
#include "Labels.h"
#include "Recognizer.h"

// Teaches the tracker a specific item from photos. Finds the item in each photo
// with the detectors (the largest one of `classHint`, or of the class most
// prominent across the photos; a face rather than the whole person), copies the
// photos to <labels>/<name>/, and stores an embedding of each (plus a mirrored
// copy) in label.yml. Photos where the item isn't found are skipped, unless
// `wholePhoto` says each photo shows just the item. Enrolling an existing name
// adds photos to it. Prints what it found and how consistent the photos are.
void enroll(const std::string& name, const std::string& classHint,
            const std::vector<std::string>& images, bool wholePhoto, Detector& detector,
            const ClassRegistry& classes, Embedders& embedders, LabelLibrary& library);
