import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class HDF5ChessDataset(Dataset):
    def __init__(self, hdf5_path: str, cache_size: int = 1000):
        """
        Efficient dataset that reads from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            cache_size: Number of samples to cache in memory
        """
        self.hdf5_path = hdf5_path
        self.cache_size = cache_size
        self.cache = {}
        
        # Open file to get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.length = f['positions'].shape[0]
            self.num_games = f.attrs['num_games']
            print(f"Loaded dataset: {self.length} positions from {self.num_games} games")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Read from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            position = torch.from_numpy(f['positions'][idx]).long()
            move = torch.tensor(f['moves'][idx], dtype=torch.long)
            is_white = bool(f['is_white'][idx])
            game_id = int(f['game_ids'][idx])
        
        sample = {
            'position': position,
            'move': move,
            'is_white': is_white,
            'game_id': game_id
        }
        
        # Update cache (simple LRU-like behavior)
        if len(self.cache) >= self.cache_size:
            # Remove random item
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = sample
        
        return sample
    
if __name__ == "__main__":
    # Example usage
    dataset = HDF5ChessDataset('data/Lichess2017.h5', cache_size=500)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample)