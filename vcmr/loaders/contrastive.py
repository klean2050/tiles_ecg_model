"""Contains PyTorch (wrapper) dataset classes to enable contrastive learning."""


import torch
from torch.utils.data import Dataset
import numpy as np
import random


class Contrastive(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.ignore_idx = []

    def __getitem__(self, idx):
        audio, label = self.dataset[idx]
        if self.transform:
            audio = self.transform(audio)
        return audio, label

    def __len__(self) -> int:
        return len(self.dataset)

    def concat_clip(self, n: int, audio_length: float):
        audio, _ = self.dataset[n]
        # split into samples of input audio length
        batch = torch.split(audio, audio_length, dim=1)
        # discard the last shorter sample
        batch = torch.cat(batch[:-1])
        # return batch in compatible shape
        return batch.unsqueeze(dim=1).unsqueeze(dim=1)


class MultiContrastive(Dataset):
    def __init__(self, dataset: Dataset, n_samples: int, sr: int = 16000, transform=None):
        # save parameters:
        self.dataset = dataset
        self.n_samples = n_samples
        self.sample_rate = sr
        self.transform = transform

        # duration of video crops in seconds (rounded to nearest second):
        self.n_seconds = int(np.around(self.n_samples / self.sample_rate))
        # number of video features:
        audio, video, _ = self.dataset[0]
        self.video_n_features = video.size(dim=-1)

        # sanity checks:
        assert self.n_samples <= audio.size(dim=-1), "n_samples is too large."
        assert self.n_seconds <= video.size(dim=0), "n_seconds is too large."
    
    def __getitem__(self, idx):
        # get audio and video:
        audio, video, label = self.dataset[idx]

        # randomly crop audio (crop length = self.n_samples):
        audio_crop, audio_start_idx = self.random_crop(audio)
        # apply transform to audio if provided:
        if self.transform:
            audio_crop = self.transform(audio_crop).squeeze(dim=1)
        
        # crop video to be (approximately) aligned with video:
        video_start_sec = int(np.around(audio_start_idx / self.sample_rate))
        video_end_sec = video_start_sec + self.n_seconds
        video_crop = video[video_start_sec : video_end_sec]

        return audio_crop, video_crop, label
    
    def __len__(self):
        return len(self.dataset)
    
    def random_crop(self, audio):
        start_idx = random.randint(0, audio.shape[-1] - self.n_samples)
        audio = audio[..., start_idx : start_idx + self.n_samples]

        return audio, start_idx

