#ifndef PERSPECTIVE_MANIFOLD_NATIVE_H
#define PERSPECTIVE_MANIFOLD_NATIVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

float perspective_fold_axis(float current, float target, float fold_rate);

size_t perspective_nearest_expert_3d(
    const float* positions_xyz,
    size_t n_experts,
    float qx,
    float qy,
    float qz
);

#ifdef __cplusplus
}
#endif

#endif
