//! Native manifold kernels (C/C++) for routing hot paths.
//!
//! Enabled with feature `native-manifold`.

#[cfg(feature = "native-manifold")]
#[link(name = "perspective_manifold_fold", kind = "static")]
unsafe extern "C" {}

#[cfg(feature = "native-manifold")]
#[link(name = "perspective_manifold_nearest", kind = "static")]
unsafe extern "C" {}

#[cfg(feature = "native-manifold")]
unsafe extern "C" {
    fn perspective_fold_axis(current: f32, target: f32, fold_rate: f32) -> f32;
    fn perspective_nearest_expert_3d(
        positions_xyz: *const f32,
        n_experts: usize,
        qx: f32,
        qy: f32,
        qz: f32,
    ) -> usize;
}

/// Fold one torus axis toward a target observation.
#[cfg(feature = "native-manifold")]
pub fn fold_axis(current: f32, target: f32, fold_rate: f32) -> f32 {
    unsafe { perspective_fold_axis(current, target, fold_rate) }
}

/// Native nearest-expert search over 3D torus coordinates.
#[cfg(feature = "native-manifold")]
pub fn nearest_expert_3d(positions_xyz: &[f32], qx: f32, qy: f32, qz: f32) -> Option<usize> {
    if positions_xyz.is_empty() || positions_xyz.len() % 3 != 0 {
        return None;
    }
    let n_experts = positions_xyz.len() / 3;
    let idx =
        unsafe { perspective_nearest_expert_3d(positions_xyz.as_ptr(), n_experts, qx, qy, qz) };
    if idx < n_experts {
        Some(idx)
    } else {
        None
    }
}
