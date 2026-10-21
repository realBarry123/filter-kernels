import torch
import random
import numpy as np
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchaudio.functional import resample
import datasets
import io
import soundfile as sf

import time
from tqdm import tqdm

class SeedRandomSampler(Sampler):
    """Adapted from https://discuss.pytorch.org/t/setting-seed-for-data-samplers-torch-utils-data-sampler/40252/2"""
    def __init__(self, indices, seed):
        self.indices = indices
        random.Random(seed).shuffle(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class DatasetWrapper(Dataset):
    def get_loaders(self, subset=None, valid_split=0.2, batch_size=1, seed=0) -> tuple[DataLoader, DataLoader]:
        """
        Adapted from https://stackoverflow.com/a/50544887 2026-06-10
        """
        subset = self if subset is None else Subset(self, range(subset))
        dataset_size = len(subset)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_split * dataset_size))
        
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_sampler = SubsetRandomSampler(indices[split:])
        valid_sampler = SubsetRandomSampler(indices[:split])
        
        train_loader = DataLoader(subset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(subset, batch_size=batch_size, sampler=valid_sampler)
        
        return train_loader, valid_loader

    def get_minimal_loader(self, subset=None, batch_size=1, seed=0):
        """Create loader with no SubsetRandomSampler, mainly for reproducible experiments"""
        dataset_size = len(self) if subset is None else min(subset, len(self))
        indices = list(range(dataset_size))
        sampler = SeedRandomSampler(indices, seed=seed)
        loader = DataLoader(self, batch_size=batch_size, sampler=sampler)
        return loader


class AudioSetDataset(DatasetWrapper):
    """
    Dataset adapter for AudioSet dataset
    """
    def __init__(self, chunk_duration=5.0, device="cuda"):
        self.dataset = datasets.load_dataset("agkphysics/AudioSet", "balanced", streaming=False, split="train")
        self.dataset = self.dataset.cast_column("audio", datasets.Audio(decode=False))
        self.chunk_duration = chunk_duration
        self.fs = 44100
        self.device = device

        self.chunk_index = []
        
        # Pre-compute chunk indices
        for idx, sample in enumerate(self.dataset):
            try:
                audio_bytes = sample["audio"]["bytes"]
                info = sf.info(io.BytesIO(audio_bytes))
                n_frames_resampled = int(info.frames * self.fs / info.samplerate)
                chunk_size = int(round(self.chunk_duration * self.fs))
                n_chunks = n_frames_resampled // chunk_size
                for i in range(n_chunks):
                    self.chunk_index.append((idx, i * chunk_size, chunk_size))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

    def __len__(self) -> int:
        return len(self.chunk_index)

    def __getitem__(self, item: int) -> torch.Tensor:
        dataset_idx, start, chunk_size = self.chunk_index[item]
        sample = self.dataset[dataset_idx]
        
        audio_bytes = sample["audio"]["bytes"]
        audio_array, fs = sf.read(io.BytesIO(audio_bytes))
        audio_array = torch.tensor(audio_array.astype(np.float32)).to(self.device)
        if fs != self.fs:
            audio_array = resample(audio_array, orig_freq=fs, new_freq=self.fs)
    
        chunk = audio_array[start:start + chunk_size]
        if len(list(chunk.shape)) == 1: # mono
            chunk = repeat(chunk, "t -> 2 t")
        elif chunk.shape[1] == 1: # sneaky mono
            chunk = repeat(chunk.squeeze(1), "t -> 2 t")
        else: 
            chunk = rearrange(chunk[:, :2], "t c -> c t")

        return chunk


if __name__ == "__main__":

    dataset = AudioSetDataset()
    train_loader, valid_loader = dataset.get_loaders(valid_split=0.2, batch_size=32)
    for x in valid_loader:
        x = rearrange(x, "1 t c -> t c")
        sf.write(f"test_audioset.wav", x, 44100)
        break
    