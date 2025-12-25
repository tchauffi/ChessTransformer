from .lichess import LichessDataset
from .lichess_simple_uci import LichessSimpleUciDataset
from .puzzle_dataset import LichessPuzzleDataset, LichessPuzzleFullSolutionDataset
from .puzzle_preference_dataset import PuzzlePreferenceDataset, PuzzlePreferenceDatasetWithMargin

__all__ = [
    "LichessDataset",
    "LichessSimpleUciDataset",
    "LichessPuzzleDataset",
    "LichessPuzzleFullSolutionDataset",
    "PuzzlePreferenceDataset",
    "PuzzlePreferenceDatasetWithMargin",
]

