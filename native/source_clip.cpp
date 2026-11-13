// The source-triangle clipping in seed_plane.py, with identical halfspace order.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

#if defined(_WIN32)
#define SCPSC_EXPORT extern "C" __declspec(dllexport)
#else
#define SCPSC_EXPORT extern "C"
#endif

namespace {
using Point = std::array<double, 3>;
using Polygon = std::array<Point, 7>;

int clip_polygon(const Polygon& input, int count, Polygon& output,
                 double nx, double ny, double bound) {
    int size = 0;
    for (int i = 0; i < count; ++i) {
        const Point& a = input[i];
        const Point& b = input[(i + 1) % count];
        const double ga = a[0] * nx + a[1] * ny - bound;
        const double gb = b[0] * nx + b[1] * ny - bound;
        if (ga <= 0.0) {
            if (size == 7) return -1;
            output[size++] = a;
        }
        if ((ga < 0.0 && gb > 0.0) || (gb < 0.0 && ga > 0.0)) {
            if (size == 7) return -1;
            const double t = ga / (ga - gb);
            for (int j = 0; j < 3; ++j)
                output[size][j] = a[j] + t * (b[j] - a[j]);
            ++size;
        }
    }
    return size;
}
}  // namespace

// tris: [triangle_count, 3, 3]; lo/hi: XY; output_capacity counts XYZ points.
// Returns the number of lexicographically sorted, exactly unique points.
// -1: invalid argument, -2: output too small, -3: allocation failed,
// -4: clipping exceeded the seven-vertex triangle/rectangle bound.
// The caller checks that the source triangles are finite. eta is added in Python.
SCPSC_EXPORT std::int64_t scpsc_clip_triangles(
    const double* tris, std::int64_t triangle_count, const double* lo,
    const double* hi, double* output, std::int64_t output_capacity) noexcept {
    if (triangle_count < 0 || output_capacity < 0 || !lo || !hi ||
        !(lo[0] < hi[0]) || !(lo[1] < hi[1]) ||
        (triangle_count > 0 && (!tris || !output))) return -1;
    if (triangle_count == 0) return 0;
    try {
        std::vector<Point> vertices;
        if (static_cast<std::uint64_t>(triangle_count) > vertices.max_size() / 7) return -1;
        vertices.reserve(static_cast<std::size_t>(triangle_count) * 7);
        // Match the reference's input ordering before unique: interior triangles
        // first, then clipped boundary triangles, retaining their source order.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::int64_t i = 0; i < triangle_count; ++i) {
                const double* tri = tris + 9 * i;
                bool inside = true;
                for (int j = 0; j < 3; ++j)
                    inside = inside && tri[3*j] >= lo[0] && tri[3*j] <= hi[0]
                                    && tri[3*j+1] >= lo[1] && tri[3*j+1] <= hi[1];
                if ((pass == 0) != inside) continue;
                Polygon polygon{}, clipped{};
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k) polygon[j][k] = tri[3*j+k];
                int count = 3;
                if (!inside) {
                    const std::array<std::array<double, 3>, 4> planes{{
                        {1.0, 0.0, hi[0]}, {-1.0, 0.0, -lo[0]},
                        {0.0, 1.0, hi[1]}, {0.0, -1.0, -lo[1]}}};
                    for (const auto& plane : planes) {
                        if (count == 0) break;
                        count = clip_polygon(polygon, count, clipped,
                                             plane[0], plane[1], plane[2]);
                        if (count < 0) return -4;
                        polygon = clipped;
                    }
                }
                vertices.insert(vertices.end(), polygon.begin(), polygon.begin() + count);
            }
        }
        // Numeric comparison, as in np.unique(axis=0): +0 and -0 are equal.
        // Their representative sign bit is not guaranteed across sorting libraries.
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
        if (vertices.size() > static_cast<std::uint64_t>(output_capacity)) return -2;
        for (std::size_t i = 0; i < vertices.size(); ++i)
            for (int j = 0; j < 3; ++j) output[3*i+j] = vertices[i][j];
        return static_cast<std::int64_t>(vertices.size());
    } catch (const std::bad_alloc&) {
        return -3;
    }
}

// Same clipped vertex set and ordering as scpsc_clip_triangles, without
// materialising the raster's triangle array or repeating interior grid nodes.
// -1: invalid argument, -2: NoData in selected cells, -3: allocation failed,
// -4: clipping exceeded seven vertices, -5: output too small.
SCPSC_EXPORT std::int64_t scpsc_clip_raster_vertices(
    const double* x, const double* y, const double* height,
    std::int64_t columns, std::int64_t rows, std::int64_t i0,
    std::int64_t i1, std::int64_t j0, std::int64_t j1,
    const double* lo, const double* hi, double* output,
    std::int64_t output_capacity) noexcept {
    if (!x || !y || !height || !lo || !hi || output_capacity < 0 ||
        i0 < 0 || j0 < 0 || i1 < i0 || j1 < j0 ||
        i1 >= columns || j1 >= rows || !(lo[0] < hi[0]) || !(lo[1] < hi[1])) return -1;
    if (i0 == i1 || j0 == j1) return 0;
    if (!output) return -1;
    try {
        std::vector<Point> vertices;
        vertices.reserve(static_cast<std::size_t>((i1-i0+1)*(j1-j0+1)));
        // Complete-cell NoData rejection remains independent of clipping.
        // Iterate x first so the interior nodes already have output order.
        for (auto i = i0; i <= i1; ++i) {
            for (auto j = j0; j <= j1; ++j) {
                const double z = height[j*columns+i];
                if (!std::isfinite(x[i]) || !std::isfinite(y[j]) || !std::isfinite(z)) return -2;
                if (x[i] >= lo[0] && x[i] <= hi[0] && y[j] >= lo[1] && y[j] <= hi[1])
                    vertices.push_back({x[i], y[j], z});
            }
        }
        const std::array<std::array<double, 3>, 4> planes{{
            {1.0, 0.0, hi[0]}, {-1.0, 0.0, -lo[0]},
            {0.0, 1.0, hi[1]}, {0.0, -1.0, -lo[1]}}};
        for (int family = 0; family < 2; ++family) {
            for (auto j = j0; j < j1; ++j) {
                for (auto i = i0; i < i1; ++i) {
                    // Every vertex of an interior cell was collected above.
                    if (x[i] >= lo[0] && x[i+1] <= hi[0] &&
                        y[j] >= lo[1] && y[j+1] <= hi[1]) continue;
                    const auto k = j*columns+i;
                    Polygon polygon{}, clipped{};
                    polygon[0] = {x[i], y[j], height[k]};
                    if (family == 0) {
                        polygon[1] = {x[i+1], y[j], height[k+1]};
                        polygon[2] = {x[i+1], y[j+1], height[k+columns+1]};
                    } else {
                        polygon[1] = {x[i+1], y[j+1], height[k+columns+1]};
                        polygon[2] = {x[i], y[j+1], height[k+columns]};
                    }
                    bool inside = true;
                    for (int v = 0; v < 3; ++v)
                        inside = inside && polygon[v][0] >= lo[0] && polygon[v][0] <= hi[0]
                                        && polygon[v][1] >= lo[1] && polygon[v][1] <= hi[1];
                    // Its three original nodes have also been collected above.
                    if (inside) continue;
                    int count = 3;
                    for (const auto& plane : planes) {
                        if (count == 0) break;
                        count = clip_polygon(polygon, count, clipped, plane[0], plane[1], plane[2]);
                        if (count < 0) return -4;
                        polygon = clipped;
                    }
                    vertices.insert(vertices.end(), polygon.begin(), polygon.begin()+count);
                }
            }
        }
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()), vertices.end());
        if (vertices.size() > static_cast<std::uint64_t>(output_capacity)) return -5;
        for (std::size_t i = 0; i < vertices.size(); ++i)
            for (int j = 0; j < 3; ++j) output[3*i+j] = vertices[i][j];
        return static_cast<std::int64_t>(vertices.size());
    } catch (const std::bad_alloc&) {
        return -3;
    }
}
