// Hungarian algorithm (Kuhn-Munkres) behind assign(): finds the minimum-cost one-to-one matching
// between rows and columns of a cost matrix, plus the handling of forbidden pairs and of matrices
// with unequal numbers of rows and columns.
#include "Assignment.h"

#include <limits>

namespace {

// Hungarian algorithm with potentials, O(n² m), for n <= m. a is n x m, 1-based
// internally. Returns the column for each row.
//
// The potentials u (per row) and v (per column) are the dual variables of the assignment problem.
// They are kept so that every pair's "reduced cost" a[i][j] - u[i] - v[j] is >= 0 and is exactly 0
// on the pairs chosen so far. Rows are added one at a time; each addition finds the cheapest way
// to fit the new row in, possibly moving earlier rows to other columns (an "augmenting path").
std::vector<int> hungarian(const std::vector<std::vector<double>>& a) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(a[0].size());
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0), v(m + 1, 0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);  // p[j]: row matched to column j
    for (int i = 1; i <= n; ++i) {
        // Column 0 is a dummy: the new row i starts the search from it.
        p[0] = i;
        int j0 = 0;
        // minv[j]: the lowest reduced cost found so far for reaching column j from the rows in the
        // search tree; way[j]: the column before j on that cheapest path.
        std::vector<double> minv(m + 1, inf);
        std::vector<char> used(m + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                // Reduced cost of reaching column j from row i0; minv keeps the cheapest.
                const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            // Shift the potentials by delta so the cheapest column becomes reachable at zero
            // reduced cost, then add it to the tree.
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        // Augment: walk the path back to the start, handing each column to the row before it.
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int> rowToCol(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j] != 0) rowToCol[p[j] - 1] = j - 1;
    return rowToCol;
}

}  // namespace

std::vector<int> assign(const std::vector<std::vector<float>>& cost) {
    const size_t rows = cost.size();
    if (rows == 0) return {};
    const size_t cols = cost[0].size();
    if (cols == 0) return std::vector<int>(rows, -1);

    // Forbidden pairs get a cost so high (far above any real cost) that the algorithm only uses
    // them when it has to fill the assignment, so it first maximizes the number of allowed pairs
    // and then minimizes their total cost. The forbidden ones are dropped afterwards.
    const double forbidden = 1e9;
    // A tall matrix is transposed (rows and columns swap roles) and the result mapped back below.
    const bool transpose = rows > cols;  // the algorithm needs rows <= columns
    std::vector<std::vector<double>> a(transpose ? cols : rows,
                                       std::vector<double>(transpose ? rows : cols));
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            const double value = cost[r][c] < 0 ? forbidden : cost[r][c];
            (transpose ? a[c][r] : a[r][c]) = value;
        }

    const std::vector<int> matched = hungarian(a);
    std::vector<int> result(rows, -1);
    for (size_t i = 0; i < matched.size(); ++i) {
        if (matched[i] < 0) continue;
        const size_t r = transpose ? static_cast<size_t>(matched[i]) : i;
        const size_t c = transpose ? i : static_cast<size_t>(matched[i]);
        // A forbidden pair was only filler: drop it, leaving this row unassigned.
        if (cost[r][c] >= 0) result[r] = static_cast<int>(c);
    }
    return result;
}
