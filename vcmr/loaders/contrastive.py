"""
Wrapper for Dataset class to enable contrastive training
"""
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio_augmentations import Compose
from typing import Tuple, List


class Contrastive(Dataset):
    def __init__(self, dataset: Dataset, input_shape: List[int], transform: Compose):
        self.dataset = dataset
        self.transform = transform
        self.input_shape = input_shape
        self.ignore_idx = []

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        # check if ignored
        if idx in self.ignore_idx:
            return self[idx + 1]
        # load the sample
        audio, label = self.dataset[idx]
        # check if you should ignore
        if audio.shape[1] < self.input_shape[1]:
            self.ignore_idx.append(idx)
            return self[idx + 1]
        # apply transformation
        if self.transform:
            audio = self.transform(audio)
        return audio, label

    def __len__(self) -> int:
        return len(self.dataset)

    def concat_clip(self, n: int, audio_length: float) -> Tensor:
        audio, _ = self.dataset[n]
        # split into samples of input audio length
        batch = torch.split(audio, audio_length, dim=1)
        # discard the last shorter sample
        batch = torch.cat(batch[:-1])
        # return batch in compatible shape
        return batch.unsqueeze(dim=1).unsqueeze(dim=1)


class MultiContrastive(Dataset):
    def __init__(self, dataset: Dataset, input_shape: List[int], transform: Compose):
        self.dataset = dataset
        self.transform = transform
        self.input_shape = input_shape

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        audio, video, label = self.dataset[idx]
        if self.transform:
            audio = self.transform(audio).squeeze(dim=1)
        return audio, video, label

    def __len__(self) -> int:
        return len(self.dataset)
