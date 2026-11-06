// Optimal one-to-one assignment, used by the Tracker to decide which detection continues which
// track. Matching greedily (cheapest pair first) can steal a detection from the track that needs
// it most; the Hungarian algorithm instead minimizes the total cost over all pairs at once.
// Optimal one-to-one assignment, used by the Tracker to decide which detection continues which
// track. Matching greedily (cheapest pair first) can steal a detection from the track that needs
// it most; the Hungarian algorithm instead minimizes the total cost over all pairs at once.
#pragma once
#include <vector>

// Minimum-cost one-to-one assignment (the Hungarian algorithm) for a rows x cols
// cost matrix. Pairs with a negative cost are not allowed. Among the assignments
// with the most allowed pairs it picks the one with the lowest total cost.
// Returns, for each row, its column, or -1 if the row stays unassigned.
//
// The matrix must be rectangular (every row the same length) and need not be square. Costs are
// floats; the Tracker uses tracks as rows, detections as columns, and a cost of
// (1 - IoU) + a weighted center distance, with -1 for implausible pairs.
//
// The matrix must be rectangular (every row the same length) and need not be square. Costs are
// floats; the Tracker uses tracks as rows, detections as columns, and a cost of
// (1 - IoU) + a weighted center distance, with -1 for implausible pairs.
std::vector<int> assign(const std::vector<std::vector<float>>& cost);
