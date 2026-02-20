#include "manifold_native.h"

static float wrap01(float x) {
    while (x >= 1.0f) {
        x -= 1.0f;
    }
    while (x < 0.0f) {
        x += 1.0f;
    }
    return x;
}

float perspective_fold_axis(float current, float target, float fold_rate) {
    if (fold_rate <= 0.0f) {
        return wrap01(current);
    }
    if (fold_rate >= 1.0f) {
        return wrap01(target);
    }

    float delta = target - current;
    if (delta > 0.5f) {
        delta -= 1.0f;
    } else if (delta < -0.5f) {
        delta += 1.0f;
    }

    return wrap01(current + fold_rate * delta);
}
