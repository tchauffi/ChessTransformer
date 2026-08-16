"""Memmap-backed dataset over the flat shards written by ``scripts/build_shards.py``.

Implements ``doc/dataset_shards.md`` §4. Every expensive thing ``HDF5ChessDataset`` does
per sample — gzip decompression, python-chess replay, legal-move enumeration, tokenizing —
has already been paid at build time. What is left here is a numpy gather out of page cache.

Two decisions carry most of the speed:

**Batch-level fetch.** At 202 bytes per sample the per-item Python call overhead dominates
the actual work, so ``__getitem__`` accepts a *list* of indices and does one vectorised
gather per batch. Drive it with :func:`make_dataloader`, which wires up a ``BatchSampler``
and ``batch_size=None`` so the DataLoader hands the whole index list through untouched.

**The legal mask is not built on CPU.** Returning ``(64, 73)`` bools would put 2.4 MB per
batch of 512 back on the H2D wire — the exact cost the shard format exists to remove. The
loader ships the ``(B, 64)`` uint16 index list (64 KB) and the training loop calls
:func:`scatter_legal_planes` on device.

**Distributed use.** This is a map-style dataset, so rank splitting is
``DistributedSampler``'s job, not the dataset's — see :func:`make_dataloader`. Map-style +
``DistributedSampler`` is preferred over an ``IterableDataset`` that owns shard *k* when
``k % (world_size * num_workers) == rank * num_workers + worker`` because the latter can
only shuffle within a worker's own shards: sample correlation then survives into the batch,
and the assignment silently unbalances whenever ``world_size * num_workers`` does not
divide the shard count. The shards here are a build-parallelism and train/val/test-split
unit; they deliberately do not constrain who reads what.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from chesstransformer.datasets.shard_format import (
    FORMAT_VERSION,
    NUM_ACTION_PLANES,
    NUM_SQUARES,
    PAD_ACTION,
    dtype_from_meta,
)

#: Fields handed to the trainer, in the dtype the model wants them in. Kept narrow on the
#: wire (uint8/int16) and widened on device — an int64 ``position`` would be 8x the copy
#: for an embedding lookup that accepts int32 anyway.
_SCALAR_FIELDS = ("from_square", "action_plane", "castling_rights", "en_passant_file", "is_white")


class FlatShardDataset(torch.utils.data.Dataset):
    """One split of a shard directory, as one addressable array of samples."""

    def __init__(self, shard_dir: str | Path, split: str = "train"):
        self.shard_dir = Path(shard_dir)
        self.split = split

        with (self.shard_dir / "meta.json").open() as fh:
            self.meta = json.load(fh)

        if self.meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"{self.shard_dir} was built with format_version "
                f"{self.meta['format_version']}, this code reads {FORMAT_VERSION}. Rebuild."
            )

        if split not in self.meta["splits"]:
            raise KeyError(f"unknown split {split!r}; have {list(self.meta['splits'])}")

        self.dtype = dtype_from_meta(self.meta["record_dtype"])
        self.pad_action = int(self.meta["pad_action"])

        wanted = set(self.meta["splits"][split])
        self.shard_meta = [m for m in self.meta["shards"] if m["shard_id"] in wanted]
        if not self.shard_meta:
            raise ValueError(f"split {split!r} of {self.shard_dir} is empty")

        # Cumulative offsets: a global index resolves to (shard, row) with one searchsorted.
        counts = np.array([m["num_samples"] for m in self.shard_meta], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        self._len = int(self.offsets[-1])

        # Opened lazily, per process. np.memmap survives fork harmlessly, but building the
        # handles in __init__ would map every shard in the parent for no reason.
        self._shards = None
        self._shards_pid = None

        # The whole point of memmap is that the page cache holds one copy that all workers
        # share. np.load(...) without mmap_mode would give each of N workers its own.
        self.game_ranges = [(m["game_start"], m["game_end"]) for m in self.shard_meta]

    def __len__(self):
        return self._len

    @property
    def shards(self):
        pid = os.getpid()
        if self._shards is None or self._shards_pid != pid:
            self._shards = [
                np.load(self.shard_dir / m["path"], mmap_mode="r") for m in self.shard_meta
            ]
            self._shards_pid = pid
        return self._shards

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shards"] = None
        state["_shards_pid"] = None
        return state

    def _gather(self, idx: np.ndarray) -> np.ndarray:
        """Fetch records for a batch of global indices, preserving their order."""
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        rows = idx - self.offsets[shard_of]

        shards = self.shards
        out = np.empty(len(idx), dtype=self.dtype)
        for sid in np.unique(shard_of):
            where = np.nonzero(shard_of == sid)[0]
            r = rows[where]
            # Read each shard in ascending row order: sequential page touches, even though
            # the batch itself is shuffled.
            order = np.argsort(r, kind="stable")
            out[where[order]] = shards[sid][r[order]]
        return out

    def __getitem__(self, index):
        """Accepts a single index or — the fast path — a list/array of them."""
        if isinstance(index, (int, np.integer)):
            batch = self._gather(np.array([index], dtype=np.int64))
        else:
            batch = self._gather(np.asarray(index, dtype=np.int64))

        # Everything stays at its on-disk width. A batch of 512 is then 103 KB rather than
        # the 5.81 MB the HDF5 path collates, and the widening to int64/bool happens on
        # device in the trainer's unpack_batch, where it is free.
        #
        # int16 rather than uint16 for the two u2 fields because torch has no uint16; both
        # are safely in range (legal_idx <= 4672, move_number max observed 601).
        out = {
            "position": torch.from_numpy(np.ascontiguousarray(batch["position"])),
            "legal_idx": torch.from_numpy(np.ascontiguousarray(batch["legal_idx"]).astype(np.int16)),
            "legal_cnt": torch.from_numpy(np.ascontiguousarray(batch["legal_cnt"])),
            "result": torch.from_numpy(np.ascontiguousarray(batch["result"])),
            "move_number": torch.from_numpy(
                np.ascontiguousarray(batch["move_number"]).astype(np.int16)
            ),
        }
        for name in _SCALAR_FIELDS:
            out[name] = torch.from_numpy(np.ascontiguousarray(batch[name]))
        return out

    def game_id_set(self) -> set:
        """Game indices this split covers. Used by the split-disjointness assertion."""
        ids = set()
        for g0, g1 in self.game_ranges:
            ids.update(range(g0, g1))
        return ids


def scatter_legal_planes(legal_idx: torch.Tensor, pad_action: int = PAD_ACTION) -> torch.Tensor:
    """Rebuild the ``(B, 64, 73)`` bool legal mask from the packed index list, on device.

    ``legal_idx`` is ``(B, C)`` with unused slots set to ``pad_action`` (== 64*73), one past
    the end of the action space. Scattering into a tensor with one extra column parks that
    padding in a scratch slot which is then sliced off — if padding were 0 instead, every
    sample would come back with square a1 / plane 0 marked legal.
    """
    b = legal_idx.size(0)
    flat = torch.zeros(b, pad_action + 1, dtype=torch.bool, device=legal_idx.device)
    flat.scatter_(1, legal_idx.long(), True)
    return flat[:, :pad_action].view(b, NUM_SQUARES, NUM_ACTION_PLANES)


def worker_init_fn(worker_id: int):
    """Give every worker (and every rank) its own numpy RNG stream.

    PyTorch reseeds ``random`` and ``torch`` per worker but *not* ``numpy``, so all workers
    inherit the parent's numpy state and draw the identical sequence — and under DDP so
    would every rank. Nothing in this dataset's hot path uses ``np.random`` any more, but
    ``HDF5ChessDataset._sample_move_idx`` does, and the fix belongs wherever a worker starts.
    """
    rank = int(os.environ.get("RANK", 0))
    np.random.seed((torch.initial_seed() + 977 * rank) % (2**32))


def make_dataloader(
    dataset: FlatShardDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    seed: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """Build the batch-fetch DataLoader this dataset is designed for.

    ``batch_size=None`` disables the DataLoader's own batching so the ``BatchSampler``'s
    index *list* reaches ``__getitem__`` intact and the gather happens once per batch
    rather than once per sample. Without this the dataset still works and runs roughly an
    order of magnitude slower.

    ``distributed=True`` wraps a ``DistributedSampler`` so each rank sees a disjoint
    1/world_size of the split. Remember to call ``loader.sampler.sampler.set_epoch(epoch)``
    — the BatchSampler nests it one level down.
    """
    if distributed:
        from torch.utils.data import DistributedSampler

        inner = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
    elif shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        inner = RandomSampler(dataset, generator=generator)
    else:
        inner = SequentialSampler(dataset)

    sampler = BatchSampler(inner, batch_size=batch_size, drop_last=drop_last)

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )
