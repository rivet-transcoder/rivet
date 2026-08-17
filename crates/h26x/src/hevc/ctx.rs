//! CABAC context variables for HEVC: the layout (offsets per syntax
//! element, matching the generated `CABAC_INIT` table) and initialisation
//! (9.3.2.2).

use crate::cabac::{Ctx, init_ctx_hevc};

pub use super::tables_gen::*;

/// The full context state of a slice / substream.
#[derive(Clone)]
pub struct Contexts {
    /// One byte per context.
    pub c: [Ctx; NUM_CONTEXTS],
    /// `StatCoeff` (persistent Rice adaptation; unused without the range
    /// extension but kept with the state so WPP storage is uniform).
    pub stat_coeff: [u8; 4],
}

impl Contexts {
    /// Initialise for `init_type` (0: I, 1: P, 2: B after cabac_init_flag
    /// swap) at `SliceQpY`.
    pub fn new(init_type: usize, qp: i32) -> Self {
        let mut c = [0u8; NUM_CONTEXTS];
        for (i, v) in c.iter_mut().enumerate() {
            *v = init_ctx_hevc(CABAC_INIT[init_type][i], qp);
        }
        Contexts { c, stat_coeff: [0; 4] }
    }
}
