//! Clock → search budget mapping, shared by UCI clock mode and the lichess
//! client. Sims/sec is tracked as an EWMA over completed searches so the
//! budget adapts to the position mix and the hardware.

use std::time::{Duration, Instant};

/// Upper bound on per-move simulations. Past a model-dependent point the search
/// starts exploiting value-head artifacts (over-rated quiet flank moves like
/// 1.h4) and converges onto junk — a model limitation, not a search bug. The
/// EMA net degrades by ~2000 sims; the base net stays sound to ~5000 (gauntlet:
/// both ~2108 Elo @ 800). The deployed bot uses the base int8 model, so cap at
/// 4000 — uses much more of the clock while keeping margin below the ~5k edge.
const MAX_SIMS: u32 = 4000;

pub struct TimeManager {
    ewma_sims_per_sec: f64,
}

impl TimeManager {
    /// `initial_sims_per_sec`: a rough seed until real measurements arrive
    /// (updated after every search).
    pub fn new(initial_sims_per_sec: f64) -> Self {
        TimeManager { ewma_sims_per_sec: initial_sims_per_sec }
    }

    pub fn update(&mut self, sims: u32, elapsed: Duration) {
        let secs = elapsed.as_secs_f64();
        if secs > 1e-3 && sims > 0 {
            let rate = sims as f64 / secs;
            self.ewma_sims_per_sec = 0.7 * self.ewma_sims_per_sec + 0.3 * rate;
        }
    }

    pub fn sims_per_sec(&self) -> f64 {
        self.ewma_sims_per_sec
    }

    /// Plan a move budget from the remaining clock and increment:
    /// (sim budget, hard deadline). The deadline overshoots the soft
    /// allocation by 1.5x but never eats more than a third of the clock; the
    /// per-wave deadline check in the search bounds overshoot to ~one wave.
    pub fn plan(&self, remaining: Duration, inc: Duration) -> (u32, Instant) {
        let reserve = Duration::from_millis(150); // network / posting latency
        let min_alloc = Duration::from_millis(50);
        let usable = remaining.saturating_sub(reserve);
        let alloc = (usable / 30 + inc.mul_f64(0.7)).clamp(min_alloc, (usable / 4).max(min_alloc));
        let sims = (alloc.as_secs_f64() * self.ewma_sims_per_sec).round() as u32;
        let sims = sims.clamp(64, MAX_SIMS);
        let hard = alloc.mul_f64(1.5).min((usable / 3).max(min_alloc));
        (sims, Instant::now() + hard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_is_sane_across_clocks() {
        let tm = TimeManager::new(500.0);
        // 3+2 blitz with a full clock: a few seconds of search.
        let (sims, _) = tm.plan(Duration::from_secs(180), Duration::from_secs(2));
        assert!((1000..=5000).contains(&sims), "blitz sims: {sims}");
        // Nearly flagged: still spends >= the minimum budget.
        let (sims, deadline) = tm.plan(Duration::from_millis(400), Duration::ZERO);
        assert_eq!(sims, 64);
        assert!(deadline <= Instant::now() + Duration::from_millis(200));
    }
}
