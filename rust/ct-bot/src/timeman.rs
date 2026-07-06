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
pub const MAX_SIMS: u32 = 4000;

/// Extra TOTAL-root-visit headroom above the per-move sims cap when pondering
/// is enabled. The visit-cap A/B (2026-07-03, logs/ab_visitcap_3600.log:
/// cap4000 scored 41.7% vs uncapped over 24 games, 0W 20D 4L) showed that
/// capping totals near the fresh-sims cap LOSES strength — accumulated warm
/// visits are an asset, and the "~5k artifact edge" is about fresh sims, not
/// totals. So the headroom is deliberately loose: it exists only to bound the
/// pathological chained-ponder tails (live logs hit 20k+ total visits), not
/// to shape normal operation (totals ≤ ~8k at the default 4000 cap).
pub const PONDER_HEADROOM: u32 = 4000;

/// Fresh-sim budget for a move given `plan` sims from the clock, `warm`
/// visits already on the root, and the total-visits ceiling `total_cap`
/// (per-speed max sims + PONDER_HEADROOM). Min-cap form, NOT `plan - warm`:
/// warm visits cost zero clock, so only clip when totals approach the
/// artifact edge; a deep ponder hit then plays near-instantly and banks the
/// clock.
pub fn effective_sims(plan: u32, warm: u32, total_cap: u32) -> u32 {
    plan.min(total_cap.saturating_sub(warm)).max(64)
}

/// Ponder-search budget: like `effective_sims` but with no floor — a ponder
/// root that already carries `total_cap` visits gets 0 sims. Without this
/// clip, chained ponder+reuse compounds tree totals without bound (live logs
/// hit 20k+ total visits, far past the ~5k validated envelope).
pub fn ponder_budget(max_sims: u32, warm: u32, total_cap: u32) -> u32 {
    max_sims.min(total_cap.saturating_sub(warm))
}

pub struct TimeManager {
    ewma_sims_per_sec: f64,
    max_sims: u32,
}

impl TimeManager {
    /// `initial_sims_per_sec`: a rough seed until real measurements arrive
    /// (updated after every search).
    pub fn new(initial_sims_per_sec: f64) -> Self {
        TimeManager { ewma_sims_per_sec: initial_sims_per_sec, max_sims: MAX_SIMS }
    }

    /// Per-game override of the sims cap (live logs: 81% of blitz and 100% of
    /// rapid moves are cap-bound, so this — not the alloc formula — is the
    /// binding time-management knob).
    pub fn set_max_sims(&mut self, max_sims: u32) {
        self.max_sims = max_sims.max(64);
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
        let sims = sims.clamp(64, self.max_sims);
        let hard = alloc.mul_f64(1.5).min((usable / 3).max(min_alloc));
        (sims, Instant::now() + hard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_sims_min_cap_form() {
        let cap = 5000;
        // No warm visits: the clock plan passes through untouched.
        assert_eq!(effective_sims(4000, 0, cap), 4000);
        // Warm visits below the cap headroom don't reduce the budget
        // (they cost zero clock).
        assert_eq!(effective_sims(3000, 1500, cap), 3000);
        // Near the cap: only the remaining headroom is spent.
        assert_eq!(effective_sims(4000, 3900, cap), 1100);
        // Deep ponder hit: clamps to the 64-sim floor, plays near-instantly.
        assert_eq!(effective_sims(4000, 4990, cap), 64);
        assert_eq!(effective_sims(4000, 6000, cap), 64);
    }

    #[test]
    fn ponder_budget_hard_bounds_totals() {
        let cap = 5000;
        // Fresh ponder root: full budget.
        assert_eq!(ponder_budget(4000, 0, cap), 4000);
        // Inherited tree near the cap: only the headroom.
        assert_eq!(ponder_budget(4000, 4200, cap), 800);
        // At/over the cap: zero — no floor, chaining stops compounding.
        assert_eq!(ponder_budget(4000, 5000, cap), 0);
        assert_eq!(ponder_budget(4000, 9000, cap), 0);
    }

    #[test]
    fn per_speed_max_sims_override() {
        let mut tm = TimeManager::new(1300.0);
        tm.set_max_sims(6000);
        // Rapid-style clock: plan should now exceed the old 4000 cap.
        let (sims, _) = tm.plan(Duration::from_secs(600), Duration::ZERO);
        assert_eq!(sims, 6000);
    }

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
