#!/usr/bin/env python3
"""Convert a Pos2MoveV2 checkpoint to opt-in architecture features without changing its play.

  --full        relation buckets -> one learned bias per (head, token pair), filled from
                the buckets, so every pair starts at the value it had.
  --smolgen     add Leela-style position-dependent attention bias (Smolgen); its shared
                output projection starts at zero, so it adds nothing until trained.
  --rule-state  add the fifty-move counter and repetition count inputs; their embeddings
                start at zero, so they add nothing until trained. --rule50-free N feeds the
                clock only past N plies (Pos2MoveV2 rule50_free).
  --drop-rule-state  remove those inputs again: the network plays as when fed clock 0 and no
                repetition (gpu_match --blind-rule-state), their embeddings folded into the
                player embedding. For data that carries no rule state, such as self-play
                recorded before the engine kept it.

Features the source already has are kept, so a converted checkpoint can be converted again.

The converted network computes the same function as the source. The script checks that
on random positions and on the start position before saving.

    uv run python scripts/convert_attn_bias.py --src data/rl/models/gen01 \
        --dst data/rl/models/gen01_sg --full --smolgen
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chesstransformer.models.transformer.pos2move_v2 import Pos2MoveV2  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--full", action="store_true", help="attn_bias 'full'")
    p.add_argument("--smolgen", action="store_true")
    p.add_argument("--rule-state", action="store_true", help="rule50 + repetition inputs")
    p.add_argument("--rule50-free", type=int, default=None,
                   help="with --rule-state: clock plies the network does not see (default: all seen)")
    p.add_argument("--drop-rule-state", action="store_true",
                   help="fold clock 0 / no repetition into the player embedding")
    p.add_argument("--compress", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not (args.full or args.smolgen or args.rule_state or args.drop_rule_state):
        p.error("nothing to convert: pass --full, --smolgen, --rule-state or --drop-rule-state")
    if args.rule_state and args.drop_rule_state:
        p.error("--rule-state and --drop-rule-state are opposites")

    src, dst = Path(args.src), Path(args.dst)
    cfg = json.loads((src / "model_config.json").read_text())
    sd = {k.replace("_orig_mod.", ""): v for k, v in load_file(str(src / "model.safetensors")).items()}
    old = Pos2MoveV2(**cfg)
    old.load_state_dict(sd)

    new_cfg = dict(cfg)
    if args.full:
        new_cfg["attn_bias"] = "full"
    if args.smolgen:
        new_cfg.update(smolgen=True, smolgen_compress=args.compress, smolgen_hidden=args.hidden,
                       smolgen_gen=args.gen)
    if args.rule_state:
        new_cfg["rule_state"] = True
        if args.rule50_free is not None:
            new_cfg["rule50_free"] = args.rule50_free
    elif args.rule50_free is not None:
        p.error("--rule50-free needs --rule-state")
    if args.drop_rule_state:
        if not cfg.get("rule_state"):
            p.error(f"{src} has no rule-state inputs to drop")
        new_cfg.pop("rule_state"), new_cfg.pop("rule50_free", None)
        # A free clock adds nothing at 0; a per-value table adds its row 0.
        clock0 = (sd["rule50_embedding.weight"][0] if cfg.get("rule50_free") is None
                  else torch.zeros_like(sd["player_embedding.weight"][0]))
        sd["player_embedding.weight"] = (sd["player_embedding.weight"] + clock0
                                         + sd["repetition_embedding.weight"][0])
        del sd["rule50_embedding.weight"], sd["repetition_embedding.weight"]
    torch.manual_seed(args.seed)
    new = Pos2MoveV2(**new_cfg)
    missing, unexpected = new.load_state_dict(dict(sd), strict=False)
    added = ("smolgen", "rule50_embedding", "repetition_embedding")
    stray = [k for k in missing if not any(a in k for a in added)]
    if stray or unexpected:
        raise SystemExit(f"unexpected key mismatch: missing {stray}, unexpected {unexpected}")

    # Same function: random token soups plus the start position.
    g = torch.Generator().manual_seed(1)
    B = 256
    x = [torch.randint(0, 13, (B, 64), generator=g), torch.randint(0, 2, (B,), generator=g),
         torch.randint(0, 16, (B,), generator=g), torch.randint(0, 9, (B,), generator=g)]
    # a1..h8; white 1-6 / black 7-12 = P N B R Q K; white to move, all castling, no e.p.
    x[0][0] = torch.tensor([4, 2, 3, 5, 6, 3, 2, 4] + [1] * 8 + [0] * 32
                           + [7] * 8 + [10, 8, 9, 11, 12, 9, 8, 10])
    x[1][0], x[2][0], x[3][0] = 1, 15, 8
    rules = dict(rule50=torch.randint(0, 101, (B,), generator=g), repetition=torch.randint(0, 3, (B,), generator=g))
    old_rules = new_rules = rules
    if args.drop_rule_state:  # the source as gpu_match --blind-rule-state plays it
        old_rules, new_rules = {k: torch.zeros_like(v) for k, v in rules.items()}, {}
    # In float64, so the check sees a changed function and not float32 summation order
    # (folding the rule state re-associates one addition).
    weights = {k: v.clone() for k, v in new.state_dict().items()}
    old, new = old.double().eval(), new.double().eval()
    with torch.no_grad():
        (lo, vo), (ln, vn) = old(*x, **old_rules), new(*x, **new_rules)
    dl, dv = (lo - ln).abs().max().item(), (vo - vn).abs().max().item()
    print(f"max |d logits| {dl:.2e} | max |d value| {dv:.2e}")
    if dl > 1e-4 or dv > 1e-4:
        raise SystemExit("converted network does not reproduce the source")

    dst.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items()}, str(dst / "model.safetensors"))
    (dst / "model_config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    n_old = sum(t.numel() for t in old.parameters())
    n_new = sum(t.numel() for t in new.parameters())
    print(f"saved {dst}: {n_old:,} -> {n_new:,} parameters ({n_new - n_old:+,}, "
          f"{(n_new - n_old) / n_old:+.1%})")


if __name__ == "__main__":
    main()
