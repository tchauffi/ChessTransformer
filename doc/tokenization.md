
# Tokenization strategy

The tokenization strategy for chess moves is a crucial aspect of this project. The goal is to convert chess moves into a format that can be processed by the transformer model.

## How to represent chess moves?
Let's look a basic starting move: `e4`.

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

This representation is more compact, but still not optimal for a transformer model as it capture the full board position on a T step of the game, which will not capture the sequential nature of chess moves.

A better solution is to use the Algebraic notation for chess moves, which is a standard way of representing chess moves using a combination of letters and numbers. For example, the move `e4` can be represented as `e4`, and the move `Nf3` can be represented as `Nf3`.

This representation is more compact and captures the sequential nature of chess moves, making it more suitable for a transformer model.

Another possibility is to use the UCI (Universal Chess Interface) notation, which is a standard way of representing chess moves using a combination of letters and numbers. For example, the move `e4` can be represented as `e2e4`, and the move `Nf3` can be represented as `g1f3`.

This representation is more compact and captures the sequential nature of chess moves, making it more suitable for a transformer model.

## Other considerations

**Special moves**: The tokenization strategy should also consider special moves such as castling, en passant, and pawn promotion. These moves have specific notations in both Algebraic and UCI notation that should be correctly tokenized. For example, castling kingside is represented as `O-O` in Algebraic notation and `e1g1` in UCI notation.

**Handling ambiguities**: In cases where multiple pieces can move to the same square, the notation includes additional information to disambiguate the move (e.g., `Nbd2` indicates that the knight from the b-file moves to d2). The tokenization strategy should account for these cases to ensure accurate representation of moves.

**Endgame scenarios**: The tokenization strategy should also consider how to represent endgame scenarios, such as check, checkmate, and stalemate. These scenarios have specific notations (e.g., `+` for check, `#` for checkmate) that should be correctly tokenized.

**Party outcome**: The tokenization strategy should also consider how to represent the outcome of the game, such as win, loss, or draw. This information can be useful for training the model to understand the consequences of certain moves. The outcome is typically represented as `1-0` for a white win, `0-1` for a black win, and `1/2-1/2` for a draw. In order, to compelment this ending token, we would need a special tocken to indigate the start of the game, such as `<START>` and `<END>` to indicate the end of the game.
