# Applying distributed training for ChessTransformer

## Introduction
In this blog post I am going to discuss about how I implemented distributed training for my ChessTransformer project. The goal of this project is to train a transformer model to predict the next move in a chess game, given the current game state.

During this project, I trained a model with compute constraints limits by using a single consumer grade GPU. However, I wanted to scale up the training process to explore bigger models and larger datasets. As transformer models require large batch sizes to train effectively, I decided to implement distributed training to leverage multiple GPUs and speed up the training process. Also inspired by the [nanochat](https://github.com/karpathy/nanochat/tree/master) project, I wanted to challenge myself to try speed-run the training process, from multiple days to a few hours, by using distributed training. In this blog post, I will explain the steps I took to implement distributed training for my ChessTransformer project, and the challenges I faced along the way.

## Distributed training introduction

Distributed training is a technique that allows us to train a model across multiple GPUs or even multiple machines. It is really usefull when our model is larger than the memory of a single GPU or when we want to speed up the training process by leveraging multiple GPUs. There are two main approaches to distributed training: data parallelism and model parallelism.

### Data parallelism
In data parallelism, we split the training data into smaller batches and distribute them across multiple GPUs. Each GPU computes the gradients for its own batch of data, and then the gradients are averaged across all GPUs to update the model parameters. This approach is suitable for models that can fit into the memory of a single GPU, but require larger batch sizes to train effectively.

![Data parallelism: every GPU holds a full model replica, trains on a disjoint slice of the batch, and the gradients are averaged with an all-reduce.](../paper/figures/distributed/ddp.png)

*Every GPU holds a complete replica. Because each rank starts from identical weights and receives the identical averaged gradient, the replicas never diverge — so only gradients ever cross the wire.*

### Model parallelism
Model parallelism adresses the case where the model is too large to fit into the memory of a single GPU. In this approach, we split the model into smaller parts and distribute them across multiple GPUs. Each GPU computes the forward and backward passes for its own part of the model, and then the gradients are communicated between GPUs to update the model parameters. This approach is suitable for very large models that cannot fit into the memory of a single GPU.

![Model parallelism: the model is split along its depth, activations flow forward from stage to stage and gradients flow back.](../paper/figures/distributed/model_parallel.png)

*The model is cut along its depth. Activations move forward from stage to stage and gradients come back the same way; no single device ever holds the whole model.*

The batch is cut into micro-batches so the stages can work at the same time instead of waiting for
each other — GPU 0 starts μ₂ while GPU 1 is still on μ₁. The idle time that remains at the start and
end of each step is the *pipeline bubble*, roughly `(S−1)/(M+S−1)` for `S` stages and `M`
micro-batches.

![Pipeline schedule for 3 stages and 4 micro-batches: 12 busy slots out of 18 total, leaving a 33% bubble.](../paper/figures/distributed/pipeline_bubble.png)

*With `S = 3` stages and `M = 4` micro-batches, 12 of the 18 slots do useful work — the remaining 33% is the bubble. Raising `M` shrinks it.*



## Measuring the starting point

Both of the diagrams above assume the GPUs are the thing you are short of. Before writing a single
line of `torch.distributed`, I wanted to check whether that was actually true for my trainer. It
turned out not to be — and the way I found out is, I think, the most useful part of this whole
project.

My setup: one RTX 5070 Ti (16 GB, roughly 89 TFLOPS of achievable bf16), 16 CPU cores, 30 GB of RAM.
The model is small — 11.7M parameters, 16 layers of width 256, and a context of only 67 tokens
(64 squares plus castling, en passant and side-to-move). The dataset is 1.53M elite games in a
372 MB HDF5 file, 148M positions in total.

I wrote a profiling script that measures three things separately: how fast the dataloader can
produce batches with no GPU involved, how fast the GPU can consume batches that are already
resident in device memory, and how fast the two go when actually run together.

The first version of that script told me something dramatic:

| | eager | `torch.compile` |
|---|---:|---:|
| Compute only, synthetic batches | 2,576 samples/s | **5,664 samples/s** |
| Dataloader only | 5,132 samples/s | 5,132 samples/s |
| **Real training loop** | **2,276 samples/s** | **2,272 samples/s** |

`torch.compile` made the GPU 2.2× faster and bought *exactly zero* end-to-end throughput. A perfect
illustration of a data-bound pipeline — I had been paying the compile warm-up on every run and
getting nothing back.

It was also completely wrong.

### The measurement tool was the bug

Those two end-to-end numbers agree to within 0.2 %. At the time I read that as a clean result. It
should have read as suspicious: two configurations that differ by a 2.2× change in GPU speed do not
land four samples per second apart by coincidence.

The cause was in my own profiler. It computed MFU using `torch.utils.flop_counter.FlopCounterMode`,
and it did so *before* running the end-to-end benchmark. `FlopCounterMode` is a `TorchDispatchMode`:
entering it forces a compiled model down a fallback path, and that de-optimisation persists for
every subsequent call on that model. So my "compiled" end-to-end benchmark was quietly measuring the
eager model. The two runs matched because they were, in fact, the same run.

Moving one function call to the end of the script gives the real picture:

| | eager | `torch.compile` |
|---|---:|---:|
| Compute only, synthetic batches | 2,551 samples/s | **5,643 samples/s** |
| Model FLOPs utilisation | 13.8 % | **30.8 %** |
| Dataloader only, 8 workers | 3,797 samples/s | 3,801 samples/s |
| **Real training loop** | **2,307 samples/s** | **4,422 samples/s** |

`torch.compile` is worth 1.9× end-to-end. The pipeline *is* data-bound once compiled — the loader
tops out at 3,801 samples/s against a 5,643 compute ceiling — so the conclusion that I needed to fix
the input pipeline first survived. But the loader was costing me about 22 % of each step, not 60 %,
and the headline I was ready to put in this post was fiction.

I am keeping this in the write-up because it is the most transferable thing I learned. The failure
mode was not a crash or an obviously silly number. It was a plausible table that pointed at a real
problem for a fake reason, produced by a tool I had written myself an hour earlier and already
trusted. Profilers are code. They get the same scepticism as the code they measure, and the cheapest
form of that scepticism is asking why a number is *as clean as it is*.

### An honest correction, and a bug I found on the way

My original plan asserted the trainer was CPU-bound and cited a class in my own codebase as proof:

```python
class RepeatingDataloader:
    """Repeats the batch N times to avoid underloaded GPU."""
```

Past me had written a band-aid that yields each batch twice to keep the GPU fed, and left a comment
saying exactly why. Damning evidence — except it was only half right. In eager mode the trainer is
genuinely *compute*-bound. It only flips to data-bound once `torch.compile` is on, which happens to
be what my training script actually uses.

That `RepeatingDataloader` is also a straightforward bug: yielding the same batch twice means every
optimiser step trains on a duplicated gradient, and its `__len__` lies to the learning-rate schedule
about how many steps there are. It is going in the bin.

### Benchmarks in isolation still overstate things

Even with the profiler fixed, the parts do not add up to the whole. Compute alone runs at 5,643
samples/s and the loader alone at 3,801, and the naive prediction for a pipeline that overlaps them
is the slower of the two: 3,801. What I actually get is 4,422 — better than that, because the
prefetch queue absorbs some of the variance, but well short of the 5,643 the GPU is capable of.

The reason the two numbers cannot simply be composed is that the dataloader workers and the main
training process compete for the same 16 cores. While the main process is busy launching CUDA
kernels it is taking a core away from the workers, so the loader never delivers in production what
it delivered on its own. Neither benchmark is wrong; they just cannot both be true at once.

You can see the same contention directly in how throughput scales with worker count:

| workers | samples/s | per worker |
|---:|---:|---:|
| 1 | 799 | 799 |
| 2 | 1,544 | 772 |
| 4 | 2,860 | 715 |
| 8 | 4,040 | 505 |
| 12 | 5,132 | 428 |
| 16 | 6,708 | 419 |

Per-worker productivity halves between 1 and 16 workers. Throwing processes at the problem was
already hitting diminishing returns.

### Where the milliseconds actually go

So the loader is the wall. The next question is which part of it. Every sample costs 1.19 ms of CPU
time — about 843 samples per second per core. Here is the breakdown:

| what happens per sample | ms | share |
|---|---:|---:|
| open the HDF5 file, read and gunzip the game | 0.352 | 30 % |
| **open the HDF5 file a second time for metadata** | **0.300** | **25 %** |
| enumerate legal moves into three tensors | 0.169 | 14 % |
| replay the game move-by-move with python-chess | 0.143 | 12 % |
| tokenise the position | 0.011 | 1 % |
| build the tensors and the sample dict | ~0.21 | 18 % |

I expected the replay to dominate. My dataset stores games as move sequences and reconstructs the
board by pushing moves one at a time, which felt like the obviously expensive thing. It is 12 %.

The actual top item is I/O, and the second item is I/O that does nothing at all. That second file
open fetches three scalars — the two players' Elo ratings and the game result — and the trainer
reads none of the Elos. The result is a single `int8` per game: the entire column is 1.5 MB and
could just live in RAM. A quarter of my data pipeline's CPU budget was spent re-opening a file to
fetch 1.5 MB that I could have loaded once at startup.

There is a similar story hiding in the first row. HDF5 stores this dataset gzip-compressed in chunks
of 10,000 games, so reading one game means decompressing the chunk it lives in:

| access pattern | ms per game |
|---|---:|
| random, reopening the file each time | 0.352 |
| random, keeping the handle open | 0.220 |
| **sequential, keeping the handle open** | **0.017** |

Sequential reads are 13× cheaper than random ones, because consecutive games share a chunk and the
decompression is amortised. Training needs random access, so this is not something I can fix in the
loader — but it is a hard constraint on the preprocessing script I am about to write, which must
walk the games in order.

### Waste I could see once I looked

Three of the tensors built for every sample are never used. The dataset returns a 64×64 grid of legal
moves, a 1,968-entry legal-move mask over the full UCI vocabulary, and a 64×73 tensor of legal
AlphaZero action planes. The trainer only reads the third one. The other two cost CPU time to build,
then get collated, pinned, and copied to the GPU so they can be ignored.

Adding up everything in a batch of 512:

| | MB |
|---|---:|
| what gets transferred to the GPU | 5.81 |
| what the model actually reads | 2.69 |
| what it needs if legal moves are sent as indices instead of a dense mask | 0.10 |

The legal-move mask is the size problem. As a dense 64×73 boolean it is 4,672 bytes per sample, and
it dwarfs everything else combined — the position itself is 64 tokens. But a real chess position has
about 34 legal moves (I measured: mean 33.9, maximum 53 across a random sample). Sending 34 16-bit
indices and scattering them into a mask on the GPU costs 68 bytes instead of 4,672, and the scatter
is free relative to everything around it.

### What this means for multiple GPUs

The last calculation is the one that reframed the project for me.

Feeding eight GPUs of this class needs roughly 45,000 samples/s. At the best per-worker rate I ever
measured — 799 samples/s, with a single uncontended worker — that is about 57 cores doing nothing
but replaying chess games. Realistically more, because per-worker throughput degrades under
contention, and because a rented 8-GPU box has A100s or H100s in it, which are hungrier than my
5070 Ti.

No 8-GPU node I can afford has the CPU to feed this dataset in its current form. Fixing the input
pipeline is not an optimisation I can do later if I have time; it is a precondition for the
distributed work having any point at all. Which is a much better thing to have discovered on my own
desk, for free, than on a rented machine at several euros an hour.

## Dataset optimisation 

** Explain how we transformed our single .h5 dataset to sharded dataset and how we used the sharded dataset to train the model in a distributed manner. **
