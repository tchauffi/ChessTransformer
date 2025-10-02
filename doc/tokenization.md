
## Tokenization strategy

The tokenization strategy for chess moves is a crucial aspect of this project. The goal is to convert chess moves into a format that can be processed by the transformer model. 

Let's look a basic move: `e4`. 

![e2e4 move](images/e4.png)

In this position, the pawn on e2 moves to e4. One solution for this position could be to encode the chess board actual position using ascii chess notation. 

```
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R
```

This representation is easy to understand for humans, but not optimal for a transformer model.

An other solution is to use a more compact representation, such as the following:

```rnbqkbnr/pppppppp/8/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1```

This representation is more compact, but still not optimal for a transformer model as it capture the full board position on a T time, which will not capture the sequential nature of chess moves.

A better solution is to use the Algebraic notation for chess moves, which is a standard way of representing chess moves using a combination of letters and numbers. For example, the move `e4` can be represented as `e4`, and the move `Nf3` can be represented as `Nf3`.

This representation is more compact and captures the sequential nature of chess moves, making it more suitable for a transformer model.


