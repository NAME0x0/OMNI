//! Safety Polytope Projection (SPP) module.
//!
//! The Safety Polytope is a convex hull defined by ~1000 anchor points and
//! ~500 halfspace constraints. Every model output is projected onto this
//! polytope *non-differentiably* using Dykstra's alternating projection
//! algorithm. This ensures hard safety guarantees that cannot be
//! gradient-attacked.

pub mod polytope;
pub mod anchors;
pub mod halfspace;
pub mod projection;
