import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.file = hdf5_file
        with h5py.File(self.file, 'r') as hf:
            self.length = hf['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file, 'r') as hf:
            # Load a single sample lazily
            data = hf['data'][idx]
            label = hf['labels'][idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.int16)