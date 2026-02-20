#include "manifold_native.h"

#include <cmath>
#include <limits>

static inline float torus_axis_delta(float a, float b) {
    float d = std::fabs(a - b);
    return d < (1.0f - d) ? d : (1.0f - d);
}

extern "C" size_t perspective_nearest_expert_3d(
    const float* positions_xyz,
    size_t n_experts,
    float qx,
    float qy,
    float qz
) {
    if (positions_xyz == nullptr || n_experts == 0) {
        return 0;
    }

    size_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < n_experts; ++i) {
        const size_t base = i * 3;
        const float dx = torus_axis_delta(positions_xyz[base], qx);
        const float dy = torus_axis_delta(positions_xyz[base + 1], qy);
        const float dz = torus_axis_delta(positions_xyz[base + 2], qz);
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_dist) {
            best_dist = d2;
            best_idx = i;
        }
    }

    return best_idx;
}
