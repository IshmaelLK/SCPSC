#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// Shared geometry for BIT*, JPS-3D interfaces, and Lazy Theta*. A one voxel
// certifies its entire CLOSED box plus the physical envelope. A zero voxel is
// uncertified. Valid points/segments belong to the union of certified boxes.
namespace closed_voxel {
using Point = std::array<double, 3>;
using Cell = std::array<int, 3>;

inline Point coordinates(const Point &p, const Point &origin, double h) {
    Point u{};
    for (int d = 0; d < 3; ++d) u[d] = (p[d] - origin[d]) / h + 0.5;
    return u;
}

template<class FreeCell>
bool covered_grid_point(const Point &u, const FreeCell &free) {
    std::array<std::array<int, 2>, 3> choice{};
    Cell count{1, 1, 1};
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(u[d]) || std::abs(u[d]) >= std::numeric_limits<int>::max() - 1.)
            return false;
        const double k = std::round(u[d]);
        if (std::abs(u[d] - k) <= 1e-10) {
            choice[d] = {static_cast<int>(k) - 1, static_cast<int>(k)};
            count[d] = 2;
        } else choice[d][0] = static_cast<int>(std::floor(u[d]));
    }
    for (int x = 0; x < count[0]; ++x)
        for (int y = 0; y < count[1]; ++y)
            for (int z = 0; z < count[2]; ++z)
                if (free(Cell{choice[0][x], choice[1][y], choice[2][z]})) return true;
    return false;
}

template<class FreeCell>
bool state(const Point &p, const Point &origin, double h, const FreeCell &free) {
    return covered_grid_point(coordinates(p, origin, h), free);
}

// Event-form DDA: union membership is constant in each open interval between
// grid-plane crossings. Check all intervals and events; do not forbid an
// uncertified neighbor that is only touched at a covered corner or edge.
// Numerical ties use 1e-10 grid units, shared by all geometric baselines.
template<class FreeCell>
bool segment(const Point &p, const Point &q, const Point &origin, double h, const FreeCell &free) {
    const Point a = coordinates(p, origin, h), b = coordinates(q, origin, h);
    if (!covered_grid_point(a, free) || !covered_grid_point(b, free)) return false;
    std::vector<double> events{0., 1.};
    for (int d = 0; d < 3; ++d) {
        if (a[d] == b[d]) continue;
        const int first = static_cast<int>(std::floor(std::min(a[d], b[d]))) + 1;
        const int last = static_cast<int>(std::ceil(std::max(a[d], b[d]))) - 1;
        for (int k = first; k <= last; ++k) {
            const double t = (k - a[d]) / (b[d] - a[d]);
            if (t > 0 && t < 1) events.push_back(t);
        }
    }
    std::sort(events.begin(), events.end());
    auto at = [&](double t) {
        Point u{};
        for (int d = 0; d < 3; ++d) u[d] = a[d] + t * (b[d] - a[d]);
        return covered_grid_point(u, free);
    };
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (!at(events[i])) return false;
        if (i && events[i] > events[i - 1] && !at(0.5 * (events[i - 1] + events[i]))) return false;
    }
    return true;
}
} // namespace closed_voxel
