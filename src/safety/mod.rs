//! Safety Polytope Projection (SPP) module.
//!
//! The Safety Polytope is a convex hull defined by ~1000 anchor points and
//! ~500 halfspace constraints. Every model output is projected onto this
//! polytope using Dykstra's alternating projection algorithm. This
//! guarantees a hard geometric constraint — OUTPUT MEMBERSHIP in the
//! polytope — for every projected point. It does not guarantee semantic
//! safety, and it is not immune to gradient-based or gradient-free
//! adversarial attacks; see `safety::projection` for details.

pub mod anchors;
pub mod halfspace;
pub mod polytope;
pub mod projection;
