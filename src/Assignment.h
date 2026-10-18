#pragma once
#include <vector>

// Minimum-cost one-to-one assignment (the Hungarian algorithm) for a rows x cols
// cost matrix. Pairs with a negative cost are not allowed. Among the assignments
// with the most allowed pairs it picks the one with the lowest total cost.
// Returns, for each row, its column, or -1 if the row stays unassigned.
std::vector<int> assign(const std::vector<std::vector<float>>& cost);
