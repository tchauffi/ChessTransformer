"""Prove the shards are the same data as the HDF5 path, then measure what they buy.

Implements ``doc/dataset_shards.md`` §5. The shards are only worth having if they are
*provably* the same distribution, not merely faster — an off-by-one in the action-plane
encoding would silently degrade the model and nothing else in the pipeline would notice.

Four checks and one benchmark:

1. **Encoding equivalence** — replay the exact (game, ply) pairs a shard holds through the
   untouched ``HDF5ChessDataset`` and compare every field byte for byte, including the
   ``(64, 73)`` legal mask after the GPU scatter.
2. **Distribution equivalence** — the ``move_number`` histogram of shard samples against
   the online sampler's, which is what proves the triangular middlegame weighting survived
   the move to build time.
3. **Split disjointness** — no game appears in two splits.
4. **Padding safety** — the scatter never marks a1/plane-0 legal on a padded slot.
5. **Loader throughput** — samples/s and bytes/batch against the Phase 1 targets.

Usage::

    uv run scripts/verify_shards.py --shards data/shards/elite_k16
    uv run scripts/verify_shards.py --shards data/shards/elite_k16 --skip-bench
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_shards import sample_plies  # noqa: E402

from chesstransformer.datasets.flat_shard_dataset import (  # noqa: E402
    FlatShardDataset,
    make_dataloader,
    scatter_legal_planes,
)
from chesstransformer.datasets.h5_lichess_dataset import HDF5ChessDataset  # noqa: E402
from chesstransformer.datasets.shard_format import LEGAL_CAP, RECORD_BYTES  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check_encoding_equivalence(meta: dict, shard_dir: Path, n_shards: int, max_games: int):
    """Field-by-field comparison against the reference implementation.

    The shard does not store a game id, so the pairing is recovered structurally: rows are
    written game by game in ascending game order, and within a game in ascending ply order,
    with the ply set reproducible from ``(seed, shard_id)``. Re-deriving the ply set and
    then encoding it through the *reference* path is what makes this a real test — it
    compares two independent encoders, not the builder against itself.
    """
    print("\n[1/5] Encoding equivalence vs HDF5ChessDataset")
    ref = HDF5ChessDataset(
        meta["source_h5"],
        cache_size=4096,
        sample_weighting=meta["sample_weighting"],
        skip_opening_plies=meta["skip_opening_plies"],
    )

    cfg = {
        "sample_weighting": meta["sample_weighting"],
        "skip_opening_plies": meta["skip_opening_plies"],
        "sample_mode": meta["sample_mode"],
    }
    k = meta["k"]

    rng_pick = np.random.default_rng(12345)
    shard_ids = rng_pick.choice(
        [m["shard_id"] for m in meta["shards"] if m["num_samples"] > 0],
        size=min(n_shards, len(meta["shards"])),
        replace=False,
    )

    compared = 0
    mismatches = []
    capped = 0

    for shard_id in sorted(int(s) for s in shard_ids):
        sm = next(m for m in meta["shards"] if m["shard_id"] == shard_id)
        rows = np.load(shard_dir / sm["path"], mmap_mode="r")
        rng = np.random.default_rng([meta["seed"], shard_id])

        cursor = 0
        games_done = 0
        for game_idx in range(sm["game_start"], sm["game_end"]):
            nm = int(ref.num_moves_per_game[game_idx])
            if nm < 2:
                continue
            plies = sample_plies(nm, k, rng, cfg)
            if len(plies) == 0:
                continue

            if games_done >= max_games:
                break
            games_done += 1

            for ply in plies:
                if cursor >= len(rows):
                    break
                rec = rows[cursor]
                cursor += 1

                # Force the reference to the ply the shard chose, then let its untouched
                # __getitem__ produce the sample.
                ref._sample_move_idx = lambda _m, _p=int(ply): _p
                got = ref[game_idx]

                for field, want in (
                    ("position", np.asarray(got["position"], dtype=np.int64)),
                    ("from_square", got["from_square"]),
                    ("action_plane", got["action_plane"]),
                    ("castling_rights", got["castling_rights"]),
                    ("en_passant_file", got["en_passant_file"]),
                    ("is_white", int(bool(got["is_white"]))),
                    ("result", got["result"]),
                    ("move_number", got["move_number"]),
                ):
                    have = rec[field]
                    ok = np.array_equal(np.asarray(have, dtype=np.int64), np.asarray(want, dtype=np.int64))
                    if not ok:
                        mismatches.append((game_idx, int(ply), field, np.asarray(have), np.asarray(want)))

                # Legal mask, after the GPU-side scatter the trainer will run.
                idx = torch.from_numpy(np.asarray(rec["legal_idx"], dtype=np.int64)).unsqueeze(0)
                planes = scatter_legal_planes(idx)[0].numpy()
                want_planes = got["legal_moves_planes"].numpy()
                if int(rec["legal_cnt"]) >= LEGAL_CAP:
                    # Capped: the stored set must be a faithful subset that kept the label.
                    capped += 1
                    if not np.all(want_planes[planes]):
                        mismatches.append((game_idx, int(ply), "legal_subset", None, None))
                    if not want_planes[int(rec["from_square"]), int(rec["action_plane"])]:
                        mismatches.append((game_idx, int(ply), "legal_target_lost", None, None))
                elif not np.array_equal(planes, want_planes):
                    mismatches.append((game_idx, int(ply), "legal_moves_planes", None, None))

                compared += 1

    print(f"      compared {compared:,} samples across {len(shard_ids)} shards "
          f"({capped} at the C={LEGAL_CAP} cap)")
    if mismatches:
        print(f"      {FAIL} {len(mismatches)} mismatches, first 5:")
        for game_idx, ply, field, have, want in mismatches[:5]:
            print(f"        game {game_idx} ply {ply} field {field}: {have} != {want}")
        return False
    print(f"      {PASS} every field byte-identical to the reference replay")
    return True


def check_distribution(meta: dict, shard_dir: Path, n: int):
    """The shard's ply marginal against the online sampler's.

    A build that quietly reverted to uniform sampling would still pass every equivalence
    check above — this is the one that would catch it.
    """
    print("\n[2/5] Ply-distribution equivalence")
    train = FlatShardDataset(shard_dir, "train")
    rng = np.random.default_rng(7)
    idx = rng.choice(len(train), size=min(n, len(train)), replace=False)
    shard_plies = train[np.sort(idx)]["move_number"].numpy().astype(np.int64)

    ref = HDF5ChessDataset(
        meta["source_h5"],
        cache_size=16,
        sample_weighting=meta["sample_weighting"],
        skip_opening_plies=meta["skip_opening_plies"],
    )
    # Draw the reference games from the *same* game ranges the split covers. Games sit in
    # the HDF5 in roughly chronological order and their length distribution drifts across
    # the file, so comparing against a whole-file sample would measure that drift instead
    # of the sampler.
    eligible = np.concatenate([np.arange(g0, g1) for g0, g1 in train.game_ranges])
    games = rng.choice(eligible, size=min(n, len(eligible)), replace=len(eligible) < n)
    online = np.empty(len(games), dtype=np.int64)
    for i, g in enumerate(games):
        nm = int(ref.num_moves_per_game[g])
        max_idx = nm - 1
        online[i] = 0 if max_idx <= 0 else ref._sample_move_idx(max_idx)

    edges = np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 10_000])
    h_shard = np.histogram(shard_plies, bins=edges)[0] / len(shard_plies)
    h_online = np.histogram(online, bins=edges)[0] / len(online)
    tvd = 0.5 * np.abs(h_shard - h_online).sum()

    print(f"      {'ply range':>12} {'shard':>8} {'online':>8} {'delta':>8}")
    for i in range(len(edges) - 1):
        hi = "inf" if edges[i + 1] > 1000 else str(edges[i + 1])
        print(f"      {f'{edges[i]}-{hi}':>12} {h_shard[i]:>8.4f} {h_online[i]:>8.4f} "
              f"{h_shard[i] - h_online[i]:>+8.4f}")
    print(f"      mean ply: shard {shard_plies.mean():.2f}  online {online.mean():.2f}")
    print(f"      total variation distance: {tvd:.4f}")

    if tvd < 0.02:
        print(f"      {PASS} distributions match (TVD < 0.02)")
        return True
    if tvd < 0.05:
        print(f"      {WARN} TVD {tvd:.4f} — small drift, expected with "
              f"--sample-mode={meta['sample_mode']}; rebuild with 'iid' if it matters")
        return True
    print(f"      {FAIL} TVD {tvd:.4f} — the build is not reproducing the online marginal")
    return False


def check_splits(shard_dir: Path):
    print("\n[3/5] Split disjointness")
    sets = {}
    for split in ("train", "val", "test"):
        ds = FlatShardDataset(shard_dir, split)
        sets[split] = ds.game_id_set()
        print(f"      {split:>5}: {len(ds):>10,} samples  {len(sets[split]):>9,} games")

    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            print(f"      {FAIL} {a} n {b} = {len(overlap):,} shared games")
            ok = False
    if ok:
        print(f"      {PASS} no game appears in two splits")
    return ok


def check_padding(shard_dir: Path, n: int = 4096):
    """A padded slot must not light up a1/plane 0 — the classic version of this bug."""
    print("\n[4/5] Padding safety in the GPU scatter")
    ds = FlatShardDataset(shard_dir, "train")
    rng = np.random.default_rng(3)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    batch = ds[idx]
    planes = scatter_legal_planes(batch["legal_idx"])
    got = planes.reshape(len(idx), -1).sum(dim=1).numpy()
    want = batch["legal_cnt"].numpy()

    if np.array_equal(got, want):
        print(f"      {PASS} set-bit count == legal_cnt for all {len(idx):,} samples "
              f"(mean {want.mean():.1f} legal moves)")
        return True
    bad = int((got != want).sum())
    print(f"      {FAIL} {bad} samples where scattered bits != legal_cnt")
    return False


def bench(shard_dir: Path, batch_size: int, workers: int, batches: int):
    print("\n[5/5] Loader throughput")
    ds = FlatShardDataset(shard_dir, "train")
    for nw in sorted({0, 1, 4, workers}):
        loader = make_dataloader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=nw, drop_last=True, pin_memory=True)
        it = iter(loader)
        for _ in range(min(10, batches)):  # warm the page cache and the workers
            next(it)
        t0 = time.perf_counter()
        n = 0
        for _ in range(batches):
            batch = next(it)
            n += batch["position"].size(0)
        elapsed = time.perf_counter() - t0
        nbytes = sum(v.numel() * v.element_size() for v in batch.values())
        print(f"      workers={nw:>2}: {n / elapsed:>10,.0f} samples/s   "
              f"{nbytes / 1e6:.2f} MB/batch")
        del loader, it
    print(f"      record on disk: {RECORD_BYTES} B/sample "
          f"(HDF5 path: 5.81 MB/batch of 512)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="data/shards/elite_k16")
    ap.add_argument("--equiv-shards", type=int, default=3, help="shards to sample for check 1")
    ap.add_argument("--equiv-games", type=int, default=120, help="games per shard for check 1")
    ap.add_argument("--dist-samples", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bench-batches", type=int, default=200)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    shard_dir = Path(args.shards)
    with (shard_dir / "meta.json").open() as fh:
        meta = json.load(fh)

    print("=" * 74)
    print("VERIFY SHARDS")
    print("=" * 74)
    print(f"shards : {shard_dir}")
    print(f"built  : K={meta['k']} seed={meta['seed']} mode={meta['sample_mode']} "
          f"weighting={meta['sample_weighting']} git={meta['git_sha'][:10]}")
    print(f"samples: {meta['num_samples']:,}  source: {meta['source_h5']}")

    results = {
        "encoding": check_encoding_equivalence(meta, shard_dir, args.equiv_shards, args.equiv_games),
        "distribution": check_distribution(meta, shard_dir, args.dist_samples),
        "splits": check_splits(shard_dir),
        "padding": check_padding(shard_dir),
    }
    if not args.skip_bench:
        bench(shard_dir, args.batch_size, args.workers, args.bench_batches)

    print("\n" + "=" * 74)
    for name, ok in results.items():
        print(f"  {name:<14} {PASS if ok else FAIL}")
    print("=" * 74)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
