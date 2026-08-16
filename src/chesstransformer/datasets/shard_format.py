"""On-disk record format for the flat, memory-mappable training shards.

Shared by ``scripts/build_shards.py`` (writer), ``FlatShardDataset`` (reader) and
``scripts/verify_shards.py`` (equivalence harness), so the layout is defined exactly
once. See ``doc/dataset_shards.md`` §3 for the design rationale and the measurements
behind the choices below.

One sample is one fixed-width row. Only the nine fields ``pos2move_v2_trainer.py``
actually reads are stored; everything else ``HDF5ChessDataset`` returns is dropped.

The legal-move mask is the whole design problem: ``(64, 73)`` bool is 4,672 bytes and
would dominate the record 20:1. It is stored instead as a fixed-width list of at most
``LEGAL_CAP`` flat ``from_square * 73 + plane`` indices (129 bytes), which is what makes
one memmap row per sample — and therefore trivial rank/worker slicing — possible. The
mask is rebuilt on GPU by :func:`~chesstransformer.datasets.flat_shard_dataset.scatter_legal_planes`.

Fixed-width was chosen over CSR (~72 B/sample, no cap) deliberately: CSR is ~28 % smaller
and is what Megatron's ``IndexedDataset`` does, but it needs a custom ragged collate, and
the 1.4 GB it saves is on a dataset that already fits in page cache. The cap is not a
correctness risk because the *target* move's index is always written first (see
``LEGAL_CAP`` below), so clipping can only ever drop alternative legal moves, never the
label. Observed legal-move counts over 400 real positions: mean 33.9, p99.9 = 53, max 53.
"""

import numpy as np

NUM_ACTION_PLANES = 73
NUM_SQUARES = 64

#: Number of legal-move indices stored per sample. Sits well above the observed max of
#: 53. The builder counts overflows and prints the rate; if it ever climbs above ~0.01 %
#: raise this to 96 and rebuild.
LEGAL_CAP = 64

#: Flat action index space is ``[0, 4672)``. Unused slots in ``legal_idx`` are filled with
#: ``PAD_ACTION`` (== 4672), one past the end, so a GPU ``scatter_`` can write padding into
#: a scratch column that is then sliced off. Writing padding as 0 instead would silently
#: mark square a1 / plane 0 legal on every single sample.
PAD_ACTION = NUM_SQUARES * NUM_ACTION_PLANES

#: Packed record layout. Offsets are explicit rather than auto-computed so that both
#: ``uint16`` fields land on even offsets — an unaligned memmap view of ``legal_idx``
#: would force numpy into a slow per-element copy on every batch gather.
#:
#: ==========  ======  =========================================================
#: offset      dtype   field
#: ==========  ======  =========================================================
#: 0-63        u1×64   position          board tokens, vocab 0-12
#: 64-191      u2×64   legal_idx         flat legal actions, target first, PAD-filled
#: 192         u1      legal_cnt         number of valid entries in legal_idx
#: 193         u1      from_square       target move source square, 0-63
#: 194         u1      action_plane      target move AlphaZero plane, 0-72
#: 195         u1      castling_rights   4-bit mask, 0-15
#: 196         u1      en_passant_file   0-7 = file a-h, 8 = none
#: 197         u1      is_white          side to move
#: 198         i1      result            0 = draw, 1 = white win, 2 = black win (as in the HDF5)
#: 199         u1      (pad)             keeps move_number 2-byte aligned
#: 200-201     u2      move_number       ply index, max observed 601
#: ==========  ======  =========================================================
RECORD_DTYPE = np.dtype(
    {
        "names": [
            "position",
            "legal_idx",
            "legal_cnt",
            "from_square",
            "action_plane",
            "castling_rights",
            "en_passant_file",
            "is_white",
            "result",
            "move_number",
        ],
        "formats": [
            (np.uint8, NUM_SQUARES),
            (np.uint16, LEGAL_CAP),
            np.uint8,
            np.uint8,
            np.uint8,
            np.uint8,
            np.uint8,
            np.uint8,
            np.int8,
            np.uint16,
        ],
        "offsets": [0, 64, 192, 193, 194, 195, 196, 197, 198, 200],
        "itemsize": 202,
    }
)

RECORD_BYTES = RECORD_DTYPE.itemsize

#: Bumped whenever the layout above changes in a way that makes old shards unreadable.
#: ``FlatShardDataset`` refuses to load a directory built by a different version.
FORMAT_VERSION = 1


def dtype_to_meta() -> dict:
    """Serialise :data:`RECORD_DTYPE` into ``meta.json``.

    Stored so a shard directory is self-describing: a reader can reconstruct the exact
    layout it was written with instead of trusting that this module never changed.
    """
    return {
        "names": list(RECORD_DTYPE.names),
        "formats": [
            (
                [RECORD_DTYPE[n].subdtype[0].str, RECORD_DTYPE[n].subdtype[1][0]]
                if RECORD_DTYPE[n].subdtype
                else RECORD_DTYPE[n].str
            )
            for n in RECORD_DTYPE.names
        ],
        "offsets": [RECORD_DTYPE.fields[n][1] for n in RECORD_DTYPE.names],
        "itemsize": RECORD_BYTES,
    }


def dtype_from_meta(meta: dict) -> np.dtype:
    """Rebuild a record dtype from what ``meta.json`` recorded."""
    formats = [tuple(f) if isinstance(f, list) else f for f in meta["formats"]]
    return np.dtype(
        {
            "names": list(meta["names"]),
            "formats": formats,
            "offsets": list(meta["offsets"]),
            "itemsize": int(meta["itemsize"]),
        }
    )
