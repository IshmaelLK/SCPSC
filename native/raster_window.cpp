// Materialize the two original triangle families of a regular grid window.
// This is the input to Algorithm 1, ClipWindow; the source surface is Eq. (1).
#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define SCPSC_RASTER_API extern "C" __declspec(dllexport)
#else
#define SCPSC_RASTER_API extern "C"
#endif

namespace {
void store(double* p, double x, double y, double z) {
    p[0] = x;
    p[1] = y;
    p[2] = z;
}
}  // namespace

SCPSC_RASTER_API int scpsc_raster_window(
    const double* x, const double* y, const double* height, std::int64_t columns,
    std::int64_t rows, std::int64_t i0, std::int64_t i1, std::int64_t j0,
    std::int64_t j1, double* output, std::int64_t triangle_count) {
    if (!x || !y || !height || !output || i0 < 0 || j0 < 0 || i1 < i0 || j1 < j0 ||
        i1 >= columns || j1 >= rows) return -1;
    const auto width = i1 - i0;
    const auto cells = width * (j1 - j0);
    if (triangle_count != 2 * cells) return -1;
    for (auto j = j0; j < j1; ++j) {
        for (auto i = i0; i < i1; ++i) {
            const auto k = j * columns + i;
            const double sw = height[k], se = height[k + 1];
            const double nw = height[k + columns], ne = height[k + columns + 1];
            if (!(std::isfinite(sw) && std::isfinite(se) && std::isfinite(nw) &&
                  std::isfinite(ne) && std::isfinite(x[i]) && std::isfinite(x[i+1]) &&
                  std::isfinite(y[j]) && std::isfinite(y[j+1]))) return -2;
            // Keep row-major order *within each family* so fragment IDs are stable.
            double* a = output + 9 * ((j - j0) * width + i - i0);
            double* b = a + 9 * cells;
            store(a, x[i], y[j], sw);
            store(a + 3, x[i + 1], y[j], se);
            store(a + 6, x[i + 1], y[j + 1], ne);
            store(b, x[i], y[j], sw);
            store(b + 3, x[i + 1], y[j + 1], ne);
            store(b + 6, x[i], y[j + 1], nw);
        }
    }
    return 0;
}
