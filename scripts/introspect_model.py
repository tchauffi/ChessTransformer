#!/usr/bin/env python
"""Dump what the network is thinking about a position, and render a viewer.

Examples
--------
Single position (defaults to the start position)::

    python scripts/introspect_model.py --out out/think.html

A whole game, every position from move 10 on::

    python scripts/introspect_model.py --pgn game.pgn --from-ply 20 \
        --out out/think.html

A position where the bot blundered, compared against another checkpoint::

    python scripts/introspect_model.py \
        --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" \
        --compare data/models/pos2move_v2.1-sfvalue --out out/think.html

The HTML is self-contained: open it directly, or publish it as an Artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chesstransformer.introspection.analyze import PositionAnalyzer, load_model  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "src" / "chesstransformer" / "introspection" / "viewer.html"
DEFAULT_MODEL = REPO / "data" / "models" / "pos2move_v2.1"


def positions_from_pgn(path: Path, from_ply: int, max_positions: int, every: int):
    with open(path) as f:
        game = chess.pgn.read_game(f)
    if game is None:
        raise SystemExit(f"No game found in {path}")

    board = game.board()
    out = []
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= from_ply and (ply - from_ply) % every == 0:
            out.append(
                {
                    "board": board.copy(stack=False),
                    "ply": ply,
                    "played_san": board.san(move),
                    "played_uci": move.uci(),
                    "last_move": board.move_stack[-1].uci() if board.move_stack else None,
                }
            )
        board.push(move)
        if len(out) >= max_positions:
            break
    if len(out) < max_positions:
        out.append(
            {
                "board": board.copy(stack=False),
                "ply": len(list(game.mainline_moves())),
                "played_san": None,
                "played_uci": None,
                "last_move": board.move_stack[-1].uci() if board.move_stack else None,
            }
        )
    return out, game.headers.get("White", "?") + " vs " + game.headers.get("Black", "?")


def verify_attention(analyzer: PositionAnalyzer, board: chess.Board) -> float:
    """Check the recomputed attention against the real SDPA output.

    Rebuilds each layer's attention output as ``probs @ V`` and compares it
    with what the attention module actually returned. A large residual means
    the recomputation drifted from the module and the maps are not
    trustworthy — better to know before staring at pretty pictures.
    """
    worst = 0.0
    for layer_idx, layer in enumerate(analyzer.model.transformer_layers):
        attn = layer.attn
        captured = {}

        def pre_hook(_m, args):
            captured["x"] = args[0].detach()

        def post_hook(_m, _a, out):
            captured["out"] = out.detach()

        h1 = attn.register_forward_pre_hook(pre_hook)
        h2 = attn.register_forward_hook(post_hook)
        try:
            analyzer._forward_batch([board])
        finally:
            h1.remove()
            h2.remove()

        with torch.no_grad():
            x = captured["x"]
            probs = analyzer._attention_probs(layer_idx, x).unsqueeze(0)
            b, t, _ = x.shape
            v = attn.v_proj(x).view(b, t, attn.num_kv_groups, attn.head_dim).transpose(1, 2)
            v = v.repeat_interleave(attn.group_size, dim=1)
            ctx = (probs @ v).transpose(1, 2).contiguous().view(b, t, attn.d_out)
            rebuilt = attn.proj(ctx)
            err = (rebuilt - captured["out"]).abs().max().item()
        worst = max(worst, err)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("position source")
    src.add_argument("--fen", action="append", help="FEN to analyse (repeatable)")
    src.add_argument("--pgn", type=Path, help="PGN file; analyses positions along the mainline")
    src.add_argument("--from-ply", type=int, default=0, help="First ply to analyse from a PGN")
    src.add_argument("--every", type=int, default=1, help="Analyse every Nth ply of the PGN")
    src.add_argument("--max-positions", type=int, default=12, help="Cap on positions analysed")

    mdl = ap.add_argument_group("model")
    mdl.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    mdl.add_argument("--ema", action="store_true", help="Load ema_state.pt instead of the raw weights")
    mdl.add_argument("--compare", type=Path, help="Second checkpoint; adds a policy/value delta column")
    mdl.add_argument("--compare-ema", action="store_true", help="Use EMA weights for --compare")
    mdl.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    out = ap.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("out/model_think.html"),
                     help=".html (viewer) or .json (raw dump)")
    out.add_argument("--json", type=Path, help="Also write the raw dump here")
    out.add_argument("--no-attention", action="store_true", help="Skip attention maps (much smaller output)")
    out.add_argument("--no-saliency", action="store_true", help="Skip occlusion saliency")
    out.add_argument("--title", default=None, help="Title shown in the viewer")
    args = ap.parse_args()

    if args.pgn:
        items, label = positions_from_pgn(args.pgn, args.from_ply, args.max_positions, args.every)
        title = args.title or f"{args.pgn.stem} — {label}"
    else:
        fens = args.fen or [chess.STARTING_FEN]
        items = [
            {"board": chess.Board(f), "ply": None, "played_san": None, "played_uci": None, "last_move": None}
            for f in fens[: args.max_positions]
        ]
        title = args.title or "Position analysis"

    print(f"Loading {args.model_dir}{' (EMA)' if args.ema else ''} on {args.device} …")
    model, config = load_model(args.model_dir, device=args.device, use_ema=args.ema)
    analyzer = PositionAnalyzer(model, device=args.device, saliency=not args.no_saliency)

    err = verify_attention(analyzer, items[0]["board"])
    print(f"Attention recomputation check: max |Δ| vs the real forward pass = {err:.2e}")
    if err > 1e-3:
        print("  WARNING: attention maps may not match the model's actual attention.")

    compare_model = None
    if args.compare:
        print(f"Loading comparison model {args.compare} …")
        compare_model, _ = load_model(args.compare, device=args.device, use_ema=args.compare_ema)
        compare_analyzer = PositionAnalyzer(compare_model, device=args.device, saliency=False)

    positions = []
    for i, item in enumerate(items):
        board = item["board"]
        print(f"[{i + 1}/{len(items)}] {board.fen()}")
        rec = analyzer.analyze(board, include_attention=not args.no_attention)
        analyzer.child_values(board, rec["policy"]["moves"])
        rec["ply"] = item["ply"]
        rec["played_san"] = item["played_san"]
        rec["played_uci"] = item["played_uci"]
        rec["last_move"] = item["last_move"]

        if compare_model is not None:
            other = compare_analyzer.analyze(board, include_attention=False)
            by_uci = {m["uci"]: m["prob"] for m in other["policy"]["moves"]}
            for m in rec["policy"]["moves"]:
                m["compare_prob"] = by_uci.get(m["uci"])
            rec["compare"] = {
                "value": other["value"],
                "entropy": other["policy"]["entropy"],
                "top1": other["policy"]["moves"][0]["san"] if other["policy"]["moves"] else None,
            }
        positions.append(rec)

    # Per-head bias over the 8 chess-relation buckets: a static, position-free
    # summary of which geometry each head is wired to prefer.
    relation_bias = [
        layer.attn.bias_table.detach().float().cpu().tolist()
        for layer in model.transformer_layers
    ]

    dump = {
        "meta": {
            "title": title,
            "relation_bias": relation_bias,
            "relation_names": [
                "same",
                "file",
                "rank",
                "diagonal",
                "knight",
                "king-adj",
                "nearby",
                "far/global",
            ],
            "model_dir": str(args.model_dir),
            "ema": args.ema,
            "compare_dir": str(args.compare) if args.compare else None,
            "config": config,
            "n_layers": analyzer.n_layers,
            "n_heads": analyzer.n_heads,
            "device": args.device,
            "attention_check": err,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
        "positions": positions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dump))
        print(f"Raw dump  → {args.json} ({args.json.stat().st_size / 1e6:.1f} MB)")

    if args.out.suffix == ".json":
        args.out.write_text(json.dumps(dump))
    else:
        html = TEMPLATE.read_text()
        # `</` only occurs inside JSON strings here, and `\/` is a valid JSON
        # escape — so this cannot end the <script> block early.
        payload = json.dumps(dump).replace("</", "<\\/")
        args.out.write_text(html.replace("__PAYLOAD__", payload))
    print(f"Viewer    → {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
