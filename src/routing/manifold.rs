//! 3D flat torus manifold — the geometric substrate for expert placement.
//!
//! The 128 experts live on an 8×4×4 torus T³ = [0,1)³ with wrap-around topology.
//! This enables:
//! - Geodesic distance: meaningful notion of "nearby" experts
//! - Voronoi cells: natural load-balancing regions
//! - Delta streaming: neighbours share structure -> small diffs
//! - Fold updates: new routing evidence updates existing expert positions

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::config::{GRID_X, GRID_Y, GRID_Z, N_EXPERTS};

/// A position on the 3D flat torus [0, 1)^3.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TorusPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl TorusPoint {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: x.rem_euclid(1.0),
            y: y.rem_euclid(1.0),
            z: z.rem_euclid(1.0),
        }
    }

    /// Geodesic distance on the flat torus.
    ///
    /// d_T(a, b) = sqrt(sum_i min(|d_i|, 1-|d_i|)^2), i in {x, y, z}
    pub fn distance(&self, other: &TorusPoint) -> f32 {
        let dx = torus_axis_abs_delta(self.x, other.x);
        let dy = torus_axis_abs_delta(self.y, other.y);
        let dz = torus_axis_abs_delta(self.z, other.z);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Move by a delta, wrapping around the torus.
    pub fn translate(&self, dx: f32, dy: f32, dz: f32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }
}

/// The expert manifold: positions of all 128 experts on the torus.
#[derive(Clone, Serialize, Deserialize)]
pub struct ExpertManifold {
    /// Expert positions on the torus.
    pub positions: Vec<TorusPoint>,

    /// Pre-computed pairwise distances [i][j].
    pub distances: Vec<Vec<f32>>,

    /// Neighbour lists: for each expert, indices of its k-nearest neighbours.
    pub neighbours: Vec<Vec<usize>>,

    /// Number of neighbours per expert.
    pub k_neighbours: usize,

    /// Packed xyz layout for native nearest-neighbour lookup.
    #[serde(skip)]
    packed_positions_xyz: Vec<f32>,
}

impl ExpertManifold {
    /// Create a default lattice layout: 128 experts on an 8×4×4 grid.
    pub fn default_grid() -> Self {
        Self::default_grid_with_jitter(0.0, 0)
    }

    /// Create a default 8×4×4 lattice with optional Gaussian jitter.
    pub fn default_grid_with_jitter(jitter_std: f32, seed: u64) -> Self {
        let mut positions = Vec::with_capacity(N_EXPERTS);
        let jitter_std = if jitter_std.is_finite() && jitter_std > 0.0 {
            jitter_std
        } else {
            0.0
        };
        let jitter_dist = if jitter_std > 0.0 {
            Normal::new(0.0, jitter_std as f64).ok()
        } else {
            None
        };
        let mut rng = StdRng::seed_from_u64(seed);

        for z in 0..GRID_Z {
            for y in 0..GRID_Y {
                for x in 0..GRID_X {
                    let mut px = (x as f32 + 0.5) / GRID_X as f32;
                    let mut py = (y as f32 + 0.5) / GRID_Y as f32;
                    let mut pz = (z as f32 + 0.5) / GRID_Z as f32;
                    if let Some(ref dist) = jitter_dist {
                        px += dist.sample(&mut rng) as f32;
                        py += dist.sample(&mut rng) as f32;
                        pz += dist.sample(&mut rng) as f32;
                    }
                    positions.push(TorusPoint::new(px, py, pz));
                }
            }
        }

        let k_neighbours = 6; // 3D lattice face-neighbour equivalent
        let mut manifold = Self {
            positions,
            distances: Vec::new(),
            neighbours: Vec::new(),
            k_neighbours,
            packed_positions_xyz: Vec::new(),
        };
        manifold.repack_positions();
        manifold.recompute_distances();
        manifold.recompute_neighbours();
        manifold
    }

    fn repack_positions(&mut self) {
        self.packed_positions_xyz.clear();
        self.packed_positions_xyz.reserve(self.positions.len() * 3);
        for p in &self.positions {
            self.packed_positions_xyz
                .extend_from_slice(&[p.x, p.y, p.z]);
        }
    }

    /// Recompute pairwise distance matrix.
    pub fn recompute_distances(&mut self) {
        let n = self.positions.len();
        self.distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.positions[i].distance(&self.positions[j]);
                self.distances[i][j] = d;
                self.distances[j][i] = d;
            }
        }
    }

    fn recompute_distances_for(&mut self, expert_id: usize) {
        let n = self.positions.len();
        if expert_id >= n || self.distances.len() != n {
            self.recompute_distances();
            return;
        }
        self.distances[expert_id][expert_id] = 0.0;
        for j in 0..n {
            if j == expert_id {
                continue;
            }
            let d = self.positions[expert_id].distance(&self.positions[j]);
            self.distances[expert_id][j] = d;
            self.distances[j][expert_id] = d;
        }
    }

    /// Recompute neighbour lists (k-nearest on torus).
    pub fn recompute_neighbours(&mut self) {
        let n = self.positions.len();
        self.neighbours = Vec::with_capacity(n);
        for i in 0..n {
            let mut indexed: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, self.distances[i][j]))
                .collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let k = self.k_neighbours.min(indexed.len());
            self.neighbours
                .push(indexed[..k].iter().map(|&(idx, _)| idx).collect());
        }
    }

    /// Find the nearest expert to a given point on the torus.
    pub fn nearest_expert(&self, point: &TorusPoint) -> usize {
        #[cfg(feature = "native-manifold")]
        {
            if let Some(idx) = crate::kernels::manifold_native::nearest_expert_3d(
                &self.packed_positions_xyz,
                point.x,
                point.y,
                point.z,
            ) {
                return idx;
            }
        }

        self.positions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.distance(point)
                    .partial_cmp(&b.distance(point))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Fold one routing observation into an existing expert position.
    ///
    /// This mutates the existing coordinate instead of appending a new point.
    pub fn fold_observation(
        &mut self,
        expert_id: usize,
        observation: &TorusPoint,
        fold_rate: f32,
    ) -> bool {
        if expert_id >= self.positions.len() {
            return false;
        }

        let rate = fold_rate.clamp(0.0, 1.0);
        let current = self.positions[expert_id];

        #[cfg(feature = "native-manifold")]
        let updated = TorusPoint::new(
            crate::kernels::manifold_native::fold_axis(current.x, observation.x, rate),
            crate::kernels::manifold_native::fold_axis(current.y, observation.y, rate),
            crate::kernels::manifold_native::fold_axis(current.z, observation.z, rate),
        );

        #[cfg(not(feature = "native-manifold"))]
        let updated = TorusPoint::new(
            fold_axis(current.x, observation.x, rate),
            fold_axis(current.y, observation.y, rate),
            fold_axis(current.z, observation.z, rate),
        );

        self.positions[expert_id] = updated;
        if self.packed_positions_xyz.len() == self.positions.len() * 3 {
            let base = expert_id * 3;
            self.packed_positions_xyz[base] = updated.x;
            self.packed_positions_xyz[base + 1] = updated.y;
            self.packed_positions_xyz[base + 2] = updated.z;
        } else {
            self.repack_positions();
        }

        self.recompute_distances_for(expert_id);
        self.recompute_neighbours();
        true
    }

    /// Fold a batch of routing observations, then recompute manifold topology.
    pub fn fold_observations(&mut self, observations: &[(usize, TorusPoint)], fold_rate: f32) {
        let rate = fold_rate.clamp(0.0, 1.0);
        if rate <= 0.0 || observations.is_empty() {
            return;
        }

        for (expert_id, point) in observations {
            if *expert_id >= self.positions.len() {
                continue;
            }
            let current = self.positions[*expert_id];
            #[cfg(feature = "native-manifold")]
            let updated = TorusPoint::new(
                crate::kernels::manifold_native::fold_axis(current.x, point.x, rate),
                crate::kernels::manifold_native::fold_axis(current.y, point.y, rate),
                crate::kernels::manifold_native::fold_axis(current.z, point.z, rate),
            );

            #[cfg(not(feature = "native-manifold"))]
            let updated = TorusPoint::new(
                fold_axis(current.x, point.x, rate),
                fold_axis(current.y, point.y, rate),
                fold_axis(current.z, point.z, rate),
            );

            self.positions[*expert_id] = updated;
        }

        self.repack_positions();
        self.recompute_distances();
        self.recompute_neighbours();
    }

    /// Get the distance between two experts by index.
    pub fn expert_distance(&self, a: usize, b: usize) -> f32 {
        self.distances[a][b]
    }

    /// Check if two experts are neighbours.
    pub fn are_neighbours(&self, a: usize, b: usize) -> bool {
        self.neighbours[a].contains(&b)
    }

    /// Get all neighbours of an expert.
    pub fn get_neighbours(&self, expert: usize) -> &[usize] {
        &self.neighbours[expert]
    }

    /// Grid coordinates for an expert (z, y, x).
    pub fn grid_coords(expert_id: usize) -> (usize, usize, usize) {
        let plane = GRID_X * GRID_Y;
        let z = expert_id / plane;
        let rem = expert_id % plane;
        let y = rem / GRID_X;
        let x = rem % GRID_X;
        (z, y, x)
    }

    /// Expert ID from grid coordinates (z, y, x).
    pub fn from_grid_coords(z: usize, y: usize, x: usize) -> usize {
        z * GRID_X * GRID_Y + y * GRID_X + x
    }
}

#[inline]
fn torus_axis_abs_delta(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

#[inline]
#[cfg(not(feature = "native-manifold"))]
fn torus_axis_signed_delta(from: f32, to: f32) -> f32 {
    let mut delta = to - from;
    if delta > 0.5 {
        delta -= 1.0;
    } else if delta < -0.5 {
        delta += 1.0;
    }
    delta
}

#[inline]
#[cfg(not(feature = "native-manifold"))]
fn fold_axis(current: f32, target: f32, fold_rate: f32) -> f32 {
    let step = torus_axis_signed_delta(current, target);
    (current + fold_rate * step).rem_euclid(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_distance_same_point() {
        let a = TorusPoint::new(0.5, 0.5, 0.5);
        assert!((a.distance(&a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_torus_distance_wrap() {
        let a = TorusPoint::new(0.0, 0.0, 0.0);
        let b = TorusPoint::new(0.9, 0.0, 0.0);
        assert!((a.distance(&b) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_torus_distance_diagonal_wrap_3d() {
        let a = TorusPoint::new(0.0, 0.0, 0.0);
        let b = TorusPoint::new(0.9, 0.9, 0.9);
        let expected = (0.1_f32 * 0.1 + 0.1 * 0.1 + 0.1 * 0.1).sqrt();
        assert!((a.distance(&b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_torus_wrap() {
        let p = TorusPoint::new(1.5, -0.3, 2.25);
        assert!((p.x - 0.5).abs() < 1e-6);
        assert!((p.y - 0.7).abs() < 1e-6);
        assert!((p.z - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_default_grid() {
        let manifold = ExpertManifold::default_grid();
        assert_eq!(manifold.positions.len(), N_EXPERTS);
        assert_eq!(manifold.neighbours.len(), N_EXPERTS);
        for nbrs in &manifold.neighbours {
            assert_eq!(nbrs.len(), manifold.k_neighbours);
        }
    }

    #[test]
    fn test_default_grid_with_jitter_is_seeded() {
        let a = ExpertManifold::default_grid_with_jitter(0.01, 42);
        let b = ExpertManifold::default_grid_with_jitter(0.01, 42);
        assert_eq!(a.positions.len(), b.positions.len());
        for (pa, pb) in a.positions.iter().zip(b.positions.iter()) {
            assert!((pa.x - pb.x).abs() < 1e-8);
            assert!((pa.y - pb.y).abs() < 1e-8);
            assert!((pa.z - pb.z).abs() < 1e-8);
        }
    }

    #[test]
    fn test_default_grid_with_jitter_differs_from_zero_jitter() {
        let base = ExpertManifold::default_grid();
        let jittered = ExpertManifold::default_grid_with_jitter(0.01, 7);
        let differs = base
            .positions
            .iter()
            .zip(jittered.positions.iter())
            .any(|(a, b)| {
                (a.x - b.x).abs() > 1e-8 || (a.y - b.y).abs() > 1e-8 || (a.z - b.z).abs() > 1e-8
            });
        assert!(differs);
    }

    #[test]
    fn test_nearest_expert() {
        let manifold = ExpertManifold::default_grid();
        let nearest = manifold.nearest_expert(&TorusPoint::new(0.06, 0.13, 0.12));
        assert_eq!(nearest, 0);
    }

    #[test]
    fn test_fold_observation_updates_in_place() {
        let mut manifold = ExpertManifold::default_grid();
        let initial_len = manifold.positions.len();
        let before = manifold.positions[0];
        let obs = TorusPoint::new(0.95, 0.95, 0.95);
        assert!(manifold.fold_observation(0, &obs, 0.5));
        let after = manifold.positions[0];
        assert_eq!(manifold.positions.len(), initial_len);
        assert!(before.distance(&after) > 0.0);
    }

    #[test]
    fn test_fold_observation_wraps_shortest_path() {
        let mut manifold = ExpertManifold::default_grid();
        manifold.positions[0] = TorusPoint::new(0.98, 0.02, 0.98);
        manifold.repack_positions();
        manifold.recompute_distances();
        manifold.recompute_neighbours();

        let obs = TorusPoint::new(0.02, 0.98, 0.01);
        assert!(manifold.fold_observation(0, &obs, 0.5));
        let p = manifold.positions[0];

        assert!(p.x < 0.05 || p.x > 0.95);
        assert!(p.y < 0.05 || p.y > 0.95);
        assert!(p.z < 0.05 || p.z > 0.95);
    }

    #[test]
    fn test_grid_coords_roundtrip() {
        for id in 0..N_EXPERTS {
            let (z, y, x) = ExpertManifold::grid_coords(id);
            assert_eq!(ExpertManifold::from_grid_coords(z, y, x), id);
        }
    }

    #[test]
    fn test_symmetry() {
        let manifold = ExpertManifold::default_grid();
        for i in 0..N_EXPERTS {
            for j in 0..N_EXPERTS {
                assert!(
                    (manifold.distances[i][j] - manifold.distances[j][i]).abs() < 1e-8,
                    "Distance not symmetric: d[{}][{}] != d[{}][{}]",
                    i,
                    j,
                    j,
                    i
                );
            }
        }
    }
}
