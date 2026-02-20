//! 2D flat torus manifold — the geometric substrate for expert placement.
//!
//! The 128 experts live on a 16×8 torus T² = [0,1)² with wrap-around topology.
//! This enables:
//! - Geodesic distance: meaningful notion of "nearby" experts
//! - Voronoi cells: natural load-balancing regions
//! - Delta streaming: neighbours share structure → small diffs

use serde::{Deserialize, Serialize};

use crate::config::{GRID_COLS, GRID_ROWS, N_EXPERTS};

/// A position on the 2D flat torus [0, 1)².
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TorusPoint {
    pub x: f32,
    pub y: f32,
}

impl TorusPoint {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: x.rem_euclid(1.0),
            y: y.rem_euclid(1.0),
        }
    }

    /// Geodesic distance on the flat torus.
    ///
    /// d_T(a, b) = sqrt(min(|dx|, 1-|dx|)² + min(|dy|, 1-|dy|)²)
    pub fn distance(&self, other: &TorusPoint) -> f32 {
        let dx = (self.x - other.x).abs();
        let dy = (self.y - other.y).abs();
        let dx = dx.min(1.0 - dx);
        let dy = dy.min(1.0 - dy);
        (dx * dx + dy * dy).sqrt()
    }

    /// Move by a delta, wrapping around the torus.
    pub fn translate(&self, dx: f32, dy: f32) -> Self {
        Self::new(self.x + dx, self.y + dy)
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
}

impl ExpertManifold {
    /// Create a default grid layout: 128 experts on a 16×8 grid.
    pub fn default_grid() -> Self {
        let mut positions = Vec::with_capacity(N_EXPERTS);
        for row in 0..GRID_ROWS {
            for col in 0..GRID_COLS {
                let x = (col as f32 + 0.5) / GRID_COLS as f32;
                let y = (row as f32 + 0.5) / GRID_ROWS as f32;
                positions.push(TorusPoint::new(x, y));
            }
        }

        let k_neighbours = 6; // 4-connected + 2 diagonal
        let mut manifold = Self {
            positions,
            distances: Vec::new(),
            neighbours: Vec::new(),
            k_neighbours,
        };
        manifold.recompute_distances();
        manifold.recompute_neighbours();
        manifold
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

    /// Recompute neighbour lists (k-nearest on torus).
    pub fn recompute_neighbours(&mut self) {
        let n = self.positions.len();
        self.neighbours = Vec::with_capacity(n);
        for i in 0..n {
            let mut indexed: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, self.distances[i][j]))
                .collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let k = self.k_neighbours.min(indexed.len());
            self.neighbours.push(indexed[..k].iter().map(|&(idx, _)| idx).collect());
        }
    }

    /// Find the nearest expert to a given point on the torus.
    pub fn nearest_expert(&self, point: &TorusPoint) -> usize {
        self.positions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.distance(point)
                    .partial_cmp(&b.distance(point))
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
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

    /// Grid coordinates for an expert (row, col).
    pub fn grid_coords(expert_id: usize) -> (usize, usize) {
        (expert_id / GRID_COLS, expert_id % GRID_COLS)
    }

    /// Expert ID from grid coordinates.
    pub fn from_grid_coords(row: usize, col: usize) -> usize {
        row * GRID_COLS + col
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_distance_same_point() {
        let a = TorusPoint::new(0.5, 0.5);
        assert!((a.distance(&a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_torus_distance_wrap() {
        let a = TorusPoint::new(0.0, 0.0);
        let b = TorusPoint::new(0.9, 0.0);
        // Direct distance = 0.9, wrapped distance = 0.1
        assert!((a.distance(&b) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_torus_distance_diagonal_wrap() {
        let a = TorusPoint::new(0.0, 0.0);
        let b = TorusPoint::new(0.9, 0.9);
        // dx = min(0.9, 0.1) = 0.1, dy = min(0.9, 0.1) = 0.1
        let expected = (0.1_f32 * 0.1 + 0.1 * 0.1).sqrt();
        assert!((a.distance(&b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_torus_wrap() {
        let p = TorusPoint::new(1.5, -0.3);
        assert!((p.x - 0.5).abs() < 1e-6);
        assert!((p.y - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_default_grid() {
        let manifold = ExpertManifold::default_grid();
        assert_eq!(manifold.positions.len(), N_EXPERTS);
        assert_eq!(manifold.neighbours.len(), N_EXPERTS);

        // All experts should have the configured number of neighbours
        for nbrs in &manifold.neighbours {
            assert_eq!(nbrs.len(), manifold.k_neighbours);
        }
    }

    #[test]
    fn test_nearest_expert() {
        let manifold = ExpertManifold::default_grid();
        // Expert 0 is at (0.0625, 0.03125) — nearest to (0.06, 0.03) should be 0
        let nearest = manifold.nearest_expert(&TorusPoint::new(0.06, 0.03));
        assert_eq!(nearest, 0);
    }

    #[test]
    fn test_grid_coords_roundtrip() {
        for id in 0..N_EXPERTS {
            let (row, col) = ExpertManifold::grid_coords(id);
            assert_eq!(ExpertManifold::from_grid_coords(row, col), id);
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
                    i, j, j, i
                );
            }
        }
    }
}
