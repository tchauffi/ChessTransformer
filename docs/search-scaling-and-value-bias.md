# Search scaling, value-head bias, and two engine fixes (2026-10-06)

**TL;DR.** (1) The 4000-sim cap in `timeman.rs` exists because search quality
*drops* at high budgets; the cause is a **selection bias in the value head's
human-outcome labels**, not a search bug, and it is measurable in the training
data. A screen confirms the cap is justified — raising it is not a free win.
(2) Two engine changes that need no retraining: Lc0's **two-fold repetition
rule** inside the search, and **Syzygy tablebase** probing (in-tree WDL + root
DTZ). Tablebases take KQvKR conversion from 2/8 to 8/8 and KBNvK from 0/8 to
8/8. Neither change has an Elo number yet — that needs a GPU-box gate.
(3) `scripts/head_to_head.py` scored claimable draws for the *winning* side;
fixed, and earlier results from it carry that bias.

## 1. Why the search drifts at high sim counts

`ct-bot probe` (new hidden subcommand: root visits/Q/prior + PV per budget),
v2.1 int8, start position:

| sims | best | Q(best) | Q(e2e4) |
|---|---|---|---|
| 800 | e2e4 | +0.044 | +0.044 |
| 3,200 | b1c3 | +0.092 | +0.057 |
| 12,800 | **a2a4** | **+0.169** | +0.051 |
| 25,600 | a2a4 | +0.162 | (all 12,800 extra sims went to a2a4) |

PV at 12,800: `1.a4 e5 2.h4 Bc5 3.e3 d5 4.a5 Nf6 5.d3 O-O 6.Be2 Nc6` —
White's Q *rises* (+0.17 → +0.24) while White shuffles rook pawns and Black
develops normally. The search is faithfully maximizing what the value head
says; the value head is wrong.

### It is selection bias in the outcome labels

1.5M games sampled from `data/elite_db.h5`:

| first move | games | White score | Elo-expected | excess | mean W−B Elo | avg Elo |
|---|---:|---:|---:|---:|---:|---:|
| all | 1,500,000 | 0.526 | 0.501 | +0.024 | +1.1 | 2510 |
| 1.e4 | 682,703 | 0.522 | 0.498 | +0.024 | −1.8 | 2509 |
| 1.d4 | 510,714 | 0.528 | 0.502 | +0.026 | +1.5 | 2508 |
| 1.Nc3 | 7,717 | 0.530 | 0.507 | +0.023 | +5.8 | 2507 |
| **1.a4** | **812** | **0.565** | 0.547 | +0.018 | **+39.4** | **2628** |

1.a4 has the highest raw White score of any first move (~2.7σ above 1.e4),
and **all of it is explained by who plays it**: its excess over the Elo
expectation (+0.018) is no larger than 1.e4's. Stronger players use rare rim
openings against weaker opponents. In value units the raw outcomes are
e4 = +0.044 — exactly the head's Q(e4) at 800 sims — and a4 = +0.13. The net
learned "the stronger player plays 1.a4", and deeper search amplifies it.

The same pattern holds one level down: 1.Nc3, the 3,200-sim choice, also
carries a positive rating gap (+5.8) and a raw score (0.530) above 1.e4's.

## 2. Sims-scaling screen

Does move quality keep improving past the 4,000 cap? 120 positions sampled
from `data/elite_db.h5` (Stockfish 19 depth-18 best move), each searched from
scratch at 800 / 3,200 / 12,800 sims; chosen moves scored at depth 16. Opening
= ply < 20, endgame = ≤ 12 pieces.

| phase | n | cp loss @800 | @3,200 | @12,800 | SF agree @800 / 3,200 / 12,800 |
|---|---:|---:|---:|---:|---|
| opening | 32 | 8.0 | **5.8** | 9.2 | 78% / 78% / 69% |
| middlegame | 77 | 16.9 | **13.9** | 14.8 | 70% / 73% / 73% |
| endgame | 11 | 5.3 | 5.3 | 2.6 | 73% / 73% / 82% |

Paired per-position (bootstrap 95% CI on Δcp):

| | 800 → 3,200 | 3,200 → 12,800 |
|---|---|---|
| opening | −2.2 [−8.2, +2.5]; 3 better / 1 worse | **+3.4 [+0.2, +7.8]; 0 better / 4 worse** |
| middlegame | −3.0 [−13.8, +4.8]; 6 / 4 | +0.9 [−6.1, +8.9]; 4 / 4 |

**The 4,000 cap is justified.** Past ~3–4k sims the opening gets worse and
the middlegame stops improving, so lifting the cap — even outside the opening —
is not supported. The engine is network-limited at that budget, which is why
the value-head bias above, not search, is the lever for further strength.
(Proxy on 120 positions; most positions pick the same move at every budget.)


## 3. Engine changes

### Two-fold repetition inside the search

`chess_core::encoding::search_terminal_value`: a position that repeats one
already on the search path (root included) is a draw at its second occurrence;
repetitions reaching back into pre-root history still need threefold (Lc0's
rule). Self-play generation (`selfplay-core`) is unchanged.

Test `finds_perpetual_via_twofold`: `6k1/5ppp/R7/Q7/3q4/6PP/7K/8 b` — Black is
a queen and rook down and every move loses (Stockfish d26: −6.5 or worse)
except the perpetual Qf2+ Kh1 Qf1+ Kh2. At 3,000 sims the new rule plays Qf2+
(value ≈ 0); with the old threefold-only rule the same search plays Qb2+ and
loses (value −0.85).

### Syzygy tablebases

`--syzygy-path` (`uci`, `lichess`, `serve`, `probe`; UCI option `SyzygyPath`).

- In-tree: leaves reached by a capture or pawn move (halfmove clock 0) that
  the tables cover are scored exactly from WDL; cursed wins / blessed losses
  are draws. Needs WDL tables only.
- Root: a won position plays the DTZ-optimal move without searching
  (guaranteed conversion under the 50-move rule); a drawn or lost one
  restricts the search to result-preserving moves. Needs DTZ tables too.

**Conversion test.** 8 random tablebase-won positions per material (DTZ ≥ 10),
bot as White at 800 nodes vs Stockfish 19 (depth 12) defending *with* the same
tables; draw claims only by the defender or on an actual third occurrence /
halfmove 100. Same positions in every column:

| material | base (no TB) | two-fold only (no TB) | with TB |
|---|---:|---:|---:|
| KQvK, KRvK, KPvK | 24/24 | 24/24 | 24/24 |
| KQvKR | 2/8 | 3/8 | **8/8** |
| KBNvK | 0/8 | 0/8 | **8/8** |

Failures without tables are almost all the 50-move rule (aimless shuffling with
the value head reading "won" everywhere) plus two stalemate blunders. Two-fold
does not change conversion — its job is seeing perpetuals and repetition draws
in the middlegame, which this test does not exercise. (With-TB numbers come from
a separate draw of the same materials; root play is DTZ-optimal by construction.)

Tables: `https://tablebase.lichess.ovh/tables/standard/3-4-5-{wdl,dtz}/`
(5-piece: 396 MB WDL + 588 MB DTZ).

## 4. Recommendations

1. **Deploy the tablebases** on the Lichess box (5-piece set, ~1 GB) and pass
   `--syzygy-path`. Gate first: `scripts/h2h_gpu.sh --bot-a <new build>
   --bot-b <old build> --syzygy-a <dir> --openings 64` — new vs old binary,
   same model.
2. **Keep the 4,000-sim cap.** §2 says more sims do not help this network.
3. **Fix the value head's selection bias in training**: give the loss the
   per-game rating gap as a fixed Elo offset in logit space, so the network does
   not have to explain the stronger player's results through position features;
   infer with gap = 0. See `doc/value_elo_offset.md` once implemented.
4. **Re-read earlier `head_to_head.py` results** with the draw-claim bug in mind:
   they over-count draws and compress real differences toward 50%.

## Harness bug: auto-claimed draws

`head_to_head.py` ended games on `is_game_over(claim_draw=True)`, which in
python-chess is true whenever the *side to move* could claim — including when a
legal move would make a third occurrence. A winning side never claims
(`elo_gauntlet.py` documents exactly this and avoids it), so won games were
scored as draws. Now it uses `engine_match.py`'s rule: automatic endings, an
actual third occurrence, or the 50-move counter at 100.

## Reproducing

```bash
# search behaviour at several budgets
rust/target/release/ct-bot probe --sims 800 3200 12800 [--fen ...] [--syzygy-path DIR]
# tablebase tests (3-4-piece tables suffice)
SYZYGY_PATH=DIR cargo test --release --manifest-path rust/Cargo.toml -p ct-bot -- --include-ignored
```
