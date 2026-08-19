# Model introspection — seeing what Pos2MoveV2 thinks

`scripts/introspect_model.py` runs the network over one or more positions, records
what happens inside it, and writes a self-contained HTML viewer. Nothing is
sampled or approximated away: attention is recomputed from each layer's captured
input using the model's own projections, and the script checks that
reconstruction against the real forward pass before it writes anything.

```bash
# the start position
python scripts/introspect_model.py --out out/think.html

# a game, every third ply from move 6 onwards
python scripts/introspect_model.py --pgn best_iter10.pgn --from-ply 12 --every 3 \
    --max-positions 8 --out out/game.html

# one position, raw weights vs. the EMA copy
python scripts/introspect_model.py \
    --fen "r2qk1nr/p1p2p2/1Rb1p3/3pP1pp/3P2Q1/P1P5/2PN1PPP/2B1KB1R w Kkq - 0 13" \
    --compare data/models/pos2move_v2.1 --compare-ema --out out/ema_diff.html
```

Open the HTML directly, or publish it as an artifact. `--json out.json` writes the
same data unrendered, for scripted analysis.

## What it records

| Panel | What it answers |
| --- | --- |
| Policy arrows + move table | What does the model want to play, and how sure is it? |
| Child value column | What does the value head think of the position *after* each legal move? |
| Value head's pick tile | Which move would the value head choose if the policy did not? |
| Where the answer forms | At which layer does the final move and the final value appear? |
| Attention explorer | Which squares does each head read, layer by layer? |
| Relation bias | Which board geometry is each head *wired* to prefer, before seeing a position? |
| Value / move saliency | Delete each piece: how far does the verdict move? |

## The three that earn their keep for debugging

**Policy vs. value disagreement.** The move table sorts by policy probability and
shows each move's child value alongside. When the value head's favourite sits at
policy rank #9 with a 1.6 % prior, MCTS has to find it through PUCT — and at a
0.5 % prior it never will (see `scripts/eval_search_coverage.py`). That is a
concrete, per-position picture of the visibility floor.

**The depth trace.** Value and policy are decoded from every layer's residual
through the final norm and the real heads. If the top move is locked in by layer
6 and the remaining ten layers only sharpen it, the depth is buying value-head
accuracy, not policy accuracy — which is exactly the kind of thing that decides
whether a bigger model is worth training.

**Occlusion saliency.** Each non-king piece is deleted in turn and the position
re-scored. A value that swings on a piece nobody is contesting is a good sign the
head has latched onto a positional shortcut rather than the tactics.

## Cost and size

Everything is a handful of batched forward passes, so a position takes well under
a second on CPU. The attention tensor dominates the output: 16 layers × 8 heads ×
67 × 67, stored as sqrt-companded `uint8` and deflated, is about 0.55 MB per
position (~0.9 MB after base64). Twelve positions land near 8 MB, which is inside
the 16 MB artifact limit; beyond that, pass `--no-attention` for a dump that is a
few kB per position.

`--no-saliency` skips the occlusion pass if you only want policy and attention.

## Accuracy notes

- Attention is recomputed, not hooked out of SDPA (which does not return
  weights). `verify_attention()` rebuilds each layer's output as `probs @ V` and
  compares it against what the module actually returned — printed on every run as
  `max |Δ|`, and typically around `1e-7`. Above `1e-3` the run warns, and the maps
  should not be trusted.
- The model is loaded in **fp32 eager**, while the bots run bf16 with
  `torch.compile`. Expect small numeric differences against a live game.
- The logit lens applies the *final* RMSNorm to intermediate layers. That norm was
  only ever trained on the last layer's residual, so early-layer readouts are
  indicative, not exact.
- Child values come from the value head only — no search. They are the network's
  static opinion of the resulting position, negated into the mover's frame.
