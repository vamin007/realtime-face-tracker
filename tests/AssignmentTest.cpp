// Tests for assign() (the Hungarian algorithm): a lowest total cost beats greedy matching,
// matrices with more rows or more columns, forbidden (negative-cost) pairs, and empty inputs.
#include <gtest/gtest.h>

#include "Assignment.h"

TEST(Assignment, PicksTheLowestTotalCost) {
    // Greedy (cheapest pair first) would take row 0 -> col 0 (cost 1) and then be
    // forced into row 1 -> col 1 (cost 10): total 11. Optimal is 2 + 3 = 5.
    const std::vector<std::vector<float>> cost = {{1, 2}, {3, 10}};
    EXPECT_EQ(assign(cost), (std::vector<int>{1, 0}));
}

// Three rows compete for one column: the cheapest row gets it, the others stay unassigned (-1).
TEST(Assignment, MoreRowsThanColumns) {
    const std::vector<std::vector<float>> cost = {{5}, {1}, {3}};
    EXPECT_EQ(assign(cost), (std::vector<int>{-1, 0, -1}));
}

// One row takes its cheapest column.
TEST(Assignment, MoreColumnsThanRows) {
    const std::vector<std::vector<float>> cost = {{4, 1, 3}};
    EXPECT_EQ(assign(cost), (std::vector<int>{1}));
}

// A negative cost means "not allowed": row 0 has no allowed pair, so it stays unassigned.
TEST(Assignment, ForbiddenPairsStayUnassigned) {
    const std::vector<std::vector<float>> cost = {{-1, -1}, {0.5f, -1}};
    EXPECT_EQ(assign(cost), (std::vector<int>{-1, 0}));
}

TEST(Assignment, PrefersMorePairsOverACheaperSingle) {
    // Row 0 alone could take col 0 for 0.1, but then row 1 gets nothing.
    const std::vector<std::vector<float>> cost = {{0.1f, 0.9f}, {0.2f, -1}};
    EXPECT_EQ(assign(cost), (std::vector<int>{1, 0}));
}

TEST(Assignment, EmptyInputs) {
    EXPECT_TRUE(assign({}).empty());
    const std::vector<std::vector<float>> noCols = {{}, {}};
    EXPECT_EQ(assign(noCols), (std::vector<int>{-1, -1}));
}
