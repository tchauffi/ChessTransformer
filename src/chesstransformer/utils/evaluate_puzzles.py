#!/usr/bin/env python3
"""
Evaluation script for chess puzzle solving with Position2Move model.

This script evaluates how well a trained model can solve chess puzzles by:
1. Loading a trained model checkpoint
2. Testing it on puzzles from the Lichess puzzle database
3. Measuring accuracy for first move, full solution, and by rating/theme
4. Generating detailed reports

Usage:
    python src/chesstransformer/utils/evaluate_puzzles.py --model data/models/puzzle_training/run_001/best_model.pth
    
Options:
    --model: Path to model checkpoint (required)
    --puzzle-data: Path to puzzle .csv.zst file (default: data/lichess_db_puzzle.csv.zst)
    --num-puzzles: Number of puzzles to evaluate (default: 1000)
    --min-rating: Minimum puzzle rating (default: None)
    --max-rating: Maximum puzzle rating (default: None)
    --themes: Comma-separated puzzle themes to filter by (default: None)
    --output: Output JSON file for results (default: results/puzzle_eval_{timestamp}.json)
"""

from pathlib import Path
from datetime import datetime
import argparse
import json

import torch
import chess
from tqdm.auto import tqdm

from chesstransformer.datasets.puzzle_dataset import LichessPuzzleFullSolutionDataset
from chesstransformer.models.transformer.position2move import Position2MoveModel
from chesstransformer.models.tokenizer.position_tokenizer import PostionTokenizer
from chesstransformer.models.tokenizer.move_tokenizer import MoveTokenizer


class PuzzleEvaluator:
    """Evaluates a Position2Move model on chess puzzles."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.position_tokenizer = PostionTokenizer()
        self.move_tokenizer = MoveTokenizer()
    
    def predict_move(self, board: chess.Board, top_k=5):
        """
        Predict the best move for a given board position.
        
        Args:
            board: chess.Board object
            top_k: Return top-k predictions
            
        Returns:
            List of (move_uci, probability, is_legal) tuples
        """
        # Encode position
        position_tokens = self.position_tokenizer.encode(board)
        position_tensor = torch.tensor(position_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        is_white = torch.tensor([board.turn == chess.WHITE], dtype=torch.long).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(position_tensor, is_white)
            probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=top_k)
        
        predictions = []
        legal_moves = [m.uci() for m in board.legal_moves]
        
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            try:
                move_uci = self.move_tokenizer.decode(int(idx))
                is_legal = move_uci in legal_moves
                predictions.append((move_uci, float(prob), is_legal))
            except ValueError:
                continue
        
        return predictions
    
    def evaluate_puzzle(self, puzzle_data, max_moves=10):
        """
        Evaluate a single puzzle.
        
        Args:
            puzzle_data: Dictionary with puzzle information
            max_moves: Maximum number of moves to try in the solution
            
        Returns:
            Dictionary with evaluation results
        """
        board = chess.Board(puzzle_data['fen'])
        solution_moves = puzzle_data['moves_uci']
        
        results = {
            'puzzle_id': puzzle_data['puzzle_id'],
            'rating': puzzle_data['rating'],
            'themes': puzzle_data['themes'],
            'solution_length': len(solution_moves),
            'moves_tried': [],
            'first_move_correct': False,
            'fully_solved': False,
            'moves_correct': 0,
            'predictions': []
        }
        
        # Try to solve the puzzle move by move
        for move_idx in range(0, len(solution_moves), 2):  # Only predict our moves (every other move)
            if move_idx >= max_moves:
                break
            
            # Apply opponent's move first (if not the first move)
            if move_idx > 0:
                opponent_move = chess.Move.from_uci(solution_moves[move_idx - 1])
                if opponent_move in board.legal_moves:
                    board.push(opponent_move)
                else:
                    # Opponent move is illegal - puzzle data might be corrupted
                    results['error'] = 'Illegal opponent move in solution'
                    break
            
            # Predict our move
            expected_move = solution_moves[move_idx]
            predictions = self.predict_move(board, top_k=5)
            
            if not predictions:
                results['error'] = 'Model produced no valid predictions'
                break
            
            predicted_move = predictions[0][0]
            predicted_prob = predictions[0][1]
            is_correct = (predicted_move == expected_move)
            
            results['moves_tried'].append({
                'move_number': move_idx // 2 + 1,
                'expected': expected_move,
                'predicted': predicted_move,
                'probability': predicted_prob,
                'correct': is_correct,
                'top_5_predictions': [
                    {'move': m, 'prob': p, 'legal': l} 
                    for m, p, l in predictions
                ]
            })
            
            if is_correct:
                results['moves_correct'] += 1
                if move_idx == 0:
                    results['first_move_correct'] = True
                
                # Apply our correct move
                our_move = chess.Move.from_uci(predicted_move)
                board.push(our_move)
            else:
                # Wrong move - puzzle failed
                break
        
        # Check if fully solved
        if results['moves_correct'] == (len(solution_moves) + 1) // 2:
            results['fully_solved'] = True
        
        return results
    
    def evaluate_dataset(self, dataset, num_puzzles=None):
        """
        Evaluate the model on a puzzle dataset.
        
        Args:
            dataset: LichessPuzzleFullSolutionDataset
            num_puzzles: Number of puzzles to evaluate (None = all)
            
        Returns:
            Dictionary with aggregate results
        """
        num_puzzles = min(num_puzzles or len(dataset), len(dataset))
        
        print(f"Evaluating {num_puzzles} puzzles...")
        
        results = {
            'num_puzzles': num_puzzles,
            'first_move_accuracy': 0,
            'full_solution_accuracy': 0,
            'average_moves_correct': 0,
            'by_rating': {},
            'by_theme': {},
            'puzzle_results': []
        }
        
        first_move_correct = 0
        fully_solved = 0
        total_moves_correct = 0
        total_moves = 0
        
        # Track by rating buckets
        rating_buckets = {
            '0-1000': {'first': 0, 'full': 0, 'total': 0},
            '1000-1500': {'first': 0, 'full': 0, 'total': 0},
            '1500-2000': {'first': 0, 'full': 0, 'total': 0},
            '2000-2500': {'first': 0, 'full': 0, 'total': 0},
            '2500+': {'first': 0, 'full': 0, 'total': 0},
        }
        
        # Track by theme
        theme_stats = {}
        
        for i in tqdm(range(num_puzzles), desc="Evaluating puzzles"):
            puzzle_data = dataset[i]
            puzzle_result = self.evaluate_puzzle(puzzle_data)
            
            # Aggregate statistics
            if puzzle_result['first_move_correct']:
                first_move_correct += 1
            if puzzle_result['fully_solved']:
                fully_solved += 1
            
            total_moves_correct += puzzle_result['moves_correct']
            total_moves += puzzle_result['solution_length'] // 2 + 1
            
            # Rating bucket
            rating = puzzle_result['rating']
            if rating < 1000:
                bucket = '0-1000'
            elif rating < 1500:
                bucket = '1000-1500'
            elif rating < 2000:
                bucket = '1500-2000'
            elif rating < 2500:
                bucket = '2000-2500'
            else:
                bucket = '2500+'
            
            rating_buckets[bucket]['total'] += 1
            if puzzle_result['first_move_correct']:
                rating_buckets[bucket]['first'] += 1
            if puzzle_result['fully_solved']:
                rating_buckets[bucket]['full'] += 1
            
            # Theme statistics
            for theme in puzzle_result['themes']:
                if theme not in theme_stats:
                    theme_stats[theme] = {'first': 0, 'full': 0, 'total': 0}
                theme_stats[theme]['total'] += 1
                if puzzle_result['first_move_correct']:
                    theme_stats[theme]['first'] += 1
                if puzzle_result['fully_solved']:
                    theme_stats[theme]['full'] += 1
            
            results['puzzle_results'].append(puzzle_result)
        
        # Calculate aggregate metrics
        results['first_move_accuracy'] = 100.0 * first_move_correct / num_puzzles
        results['full_solution_accuracy'] = 100.0 * fully_solved / num_puzzles
        results['average_moves_correct'] = total_moves_correct / num_puzzles
        results['move_accuracy'] = 100.0 * total_moves_correct / total_moves if total_moves > 0 else 0
        
        # Rating bucket statistics
        for bucket, stats in rating_buckets.items():
            if stats['total'] > 0:
                results['by_rating'][bucket] = {
                    'count': stats['total'],
                    'first_move_accuracy': 100.0 * stats['first'] / stats['total'],
                    'full_solution_accuracy': 100.0 * stats['full'] / stats['total'],
                }
        
        # Theme statistics (top 20 themes by frequency)
        sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:20]
        for theme, stats in sorted_themes:
            results['by_theme'][theme] = {
                'count': stats['total'],
                'first_move_accuracy': 100.0 * stats['first'] / stats['total'],
                'full_solution_accuracy': 100.0 * stats['full'] / stats['total'],
            }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Position2Move model on chess puzzles')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--puzzle-data', type=str, default='data/lichess_db_puzzle.csv.zst',
                        help='Path to puzzle .csv.zst file')
    parser.add_argument('--num-puzzles', type=int, default=1000,
                        help='Number of puzzles to evaluate')
    parser.add_argument('--min-rating', type=int, default=None,
                        help='Minimum puzzle rating')
    parser.add_argument('--max-rating', type=int, default=None,
                        help='Maximum puzzle rating')
    parser.add_argument('--themes', type=str, default=None,
                        help='Comma-separated puzzle themes to filter by')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Parse themes
    themes = args.themes.split(',') if args.themes else None
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model_path = Path(args.model)
    
    # Support both .pth and .safetensors formats
    if model_path.suffix == '.safetensors':
        # Load config from same directory
        from safetensors import safe_open
        import json
        
        config_path = model_path.parent / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        model = Position2MoveModel(**config)
        
        # Load weights from safetensors
        with safe_open(str(model_path), framework="pt", device=str(device)) as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
            
            # Handle compiled model prefix (_orig_mod.)
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
        
        print(f"Model loaded from safetensors")
    else:
        # Load from .pth checkpoint
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        model = Position2MoveModel(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load puzzle dataset
    print(f"\nLoading puzzle dataset...")
    dataset = LichessPuzzleFullSolutionDataset(
        puzzle_path=args.puzzle_data,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        themes=themes,
        max_puzzles=args.num_puzzles
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("Starting evaluation")
    print("="*70)
    
    evaluator = PuzzleEvaluator(model, device=device)
    results = evaluator.evaluate_dataset(dataset, num_puzzles=args.num_puzzles)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total puzzles: {results['num_puzzles']}")
    print(f"First move accuracy: {results['first_move_accuracy']:.2f}%")
    print(f"Full solution accuracy: {results['full_solution_accuracy']:.2f}%")
    print(f"Move accuracy: {results['move_accuracy']:.2f}%")
    print(f"Average moves correct: {results['average_moves_correct']:.2f}")
    
    print("\n" + "-"*70)
    print("By Rating:")
    print("-"*70)
    for rating, stats in sorted(results['by_rating'].items()):
        print(f"  {rating:>12}: {stats['count']:4} puzzles | "
              f"First: {stats['first_move_accuracy']:5.2f}% | "
              f"Full: {stats['full_solution_accuracy']:5.2f}%")
    
    print("\n" + "-"*70)
    print("Top Themes:")
    print("-"*70)
    for theme, stats in list(results['by_theme'].items())[:10]:
        print(f"  {theme:>20}: {stats['count']:4} puzzles | "
              f"First: {stats['first_move_accuracy']:5.2f}% | "
              f"Full: {stats['full_solution_accuracy']:5.2f}%")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/puzzle_eval_{timestamp}.json"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Don't save individual puzzle results to keep file size manageable
    results_summary = {k: v for k, v in results.items() if k != 'puzzle_results'}
    results_summary['num_puzzles_detailed'] = len(results['puzzle_results'])
    
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
