"""
Wrapper for Dataset class to enable contrastive training
"""
import torch, random
from torch.utils.data import Dataset


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
    def __init__(self, dataset, n_samples, sr=16000, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.n_samples = n_samples
        self.sample_rate = sr

    def random_crop(self, audio):
        start_idx = random.randint(0, audio.shape[-1] - self.n_samples)
        audio = audio[..., start_idx : start_idx + self.n_samples]
        return audio, start_idx

    def __getitem__(self, idx):
        audio, video, label = self.dataset[idx]
        audio, start_idx = self.random_crop(audio)
        if self.transform:
            audio = self.transform(audio).squeeze(dim=1)

        duration = self.n_samples // self.sample_rate
        # consider a larger duration for the video
        video_start = max(0, start_idx // self.sample_rate - 2)
        video_end = min(len(video), video_start + duration + 2)

        return audio, video[video_start : video_end], label

    def __len__(self):
        return len(self.dataset)
