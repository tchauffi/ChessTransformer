from pathlib import Path
import json
import torch
import chess
from safetensors import safe_open
from chesstransformer.models.transformer.position2move import Position2MoveModel
from chesstransformer.models.tokenizer import PostionTokenizer, MoveTokenizer

data_foler = Path(__file__).parents[1] / "data"

class Position2MoveBot:
    def __init__(self, model_path: str = str((data_foler / "models/V2_model/model.safetensors").resolve()), device: str = "cpu"):
        self.device = device
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
        
        config_path = Path(model_path).parent / "config.json"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load model
        self.model = Position2MoveModel(**self.config).to(device)
        
        with safe_open(model_path, framework="pt", device=device) as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
            
            # Handle compiled model prefix (_orig_mod.)
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
            self.model.load_state_dict(state_dict)

        self.model.eval()

    @torch.no_grad()
    def predict(self, board: chess.Board):
        tokens_ids = self.position_tokenizer.encode(board)
        torch_input = torch.tensor(tokens_ids).unsqueeze(0).long().to(self.device)  # Add batch dimension

        is_white = board.turn
        is_white = torch.tensor([is_white]).bool().to(self.device)  # Add batch dimension

        logits = self.model(torch_input, is_white)

        legal_moves = [m.uci() for m in board.legal_moves]

        mask = torch.full((logits.size(-1),), float('-inf')).to(self.device)
        for move in legal_moves:
            move_id = self.move_tokenizer.encode(move)
            mask[move_id] = 0.0

        masked_logits = logits[0] + mask  # Assuming batch size of 1
        probs = torch.softmax(masked_logits, dim=-1)
        # draw the move from the distribution
        predicted_move_id = torch.multinomial(probs * 0.2, num_samples=1).item()
        proba = probs[predicted_move_id].item()

        predicted_move = self.move_tokenizer.decode(predicted_move_id)
        return predicted_move, proba

if __name__ == "__main__":
    bot = Position2MoveBot()

    board = chess.Board()

    print("Initial board:")
    print(board)

    move, proba = bot.predict(board)
    print(f"Bot suggests move: {move} with probability {proba:.4f}")

    board.push_uci(move)
    print("Board after bot move:")
    print(board)

    move, proba = bot.predict(board)
    print(f"Bot suggests move: {move} with probability {proba:.4f}")

    board.push_uci(move)
    print("Board after bot move:")
    print(board)