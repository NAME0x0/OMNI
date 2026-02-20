//! Forward-Mode Evolutionary Adaptation (FMEA).
//!
//! A novel continual learning system that avoids backpropagation entirely:
//! - **JVP** (Jacobian-Vector Products) for gradient-free directional derivatives
//! - **LoRA rank-4** adapters for parameter-efficient updates
//! - **NES** (Natural Evolution Strategies) for routing optimisation

pub mod evolutionary;
pub mod fmea;
pub mod jvp;
pub mod lora;
