# System design strategy for ChessTransformer

Their is to main design strategy we can think for this project:
- the move sequence to move sequence strategy
- the game state to game state strategy

## 1. Move sequence to move sequence strategy

This strategy is the closest to the way language models are trained. The idea is to train a transformer model to predict the next move in a sequence of moves, given the previous moves. This can be done by treating the sequence of moves as a sequence of tokens, similar to how words are treated in natural language processing.

The model can be trained on a large dataset of chess games, where each game is represented as a sequence of moves. The input to the model would be the sequence of moves up to a certain point, and the output would be the next move in the sequence. The model can be trained using a cross-entropy loss function, similar to how language models are trained.

Once the model is trained, it can be used to generate moves in a chess game by providing it with the current sequence of moves and sampling from the predicted next move distribution. 

### First results 
This approach has been implemented and tested on a small dataset of chess games. The model was able to learn some basic patterns in the data, but it struggled to generate legal moves consistently. This is likely due to the fact that the model was not explicitly trained to understand the rules of chess, and it may have learned some incorrect patterns from the data.

## 2. Game state to game state strategy

This strategy involves training a transformer model to predict the next game state, given the current game state. The game state is the represntation of the chess board at a given point in time, it can be modeled as a 64 tokens sequence, where each token represents a square on the board and its content (empty, white piece, black piece). The model can be trained on a large dataset of chess games, where each game is represented as a sequence of game states. The input to the model would be the current game state + the turn indicator (white or black to play), and the output would be the next game state. The model can be trained using a cross-entropy loss function, similar to how language models are trained.

Once the model is trained, it can be used to generate moves in a chess game by providing it with the current game state and sampling from the predicted next game state distribution. The move can be inferred by comparing the current game state and the predicted next game state.
