"""Contains PyTorch (wrapper) dataset class for splitting songs into segments."""


from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
from typing import Tuple, Any


class SongSplitter(Dataset):
    """Wrapper dataset class for splitting songs into (possibly overlapping) segments.

    Attributes:
        dataset (torch.utils.data.Dataset): Dataset.
        audio_length (int): Length of each audio segment (in samples).
        overlap_ratio (float): Overlap ratio between adjacent segments (in range [0, 0.9]).
    """

    def __init__(
        self, dataset: Any, audio_length: int, overlap_ratio: float = 0
    ) -> None:
        """Initialization.

        Args:
            dataset (torch.utils.data.Dataset): Dataset.
            audio_length (int): Length of each audio segment (in samples).
            overlap_ratio (float): Overlap ratio between adjacent segments (in range [0, 0.9]).

        Returns: None
        """

        # validate overlap ratio:
        if overlap_ratio < 0 or overlap_ratio > 0.9:
            raise ValueError("Invalid overlap ratio value.")

        # save parameters:
        self.dataset = dataset
        self.audio_length = audio_length
        self.overlap_ratio = overlap_ratio

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Gets item:

        Args:
            idx (int): Item index.

        Returns:
            audio_segments (torch.Tensor): Song split into audio segments.
                size: (n_segments, 1, self.audio_length)
            label: Song label (tags).
                size: (n_tags, )
        """

        # get audio and label of song:
        audio_song, label = self.dataset[idx]
        audio_song = audio_song.squeeze(dim=0)
        assert self.audio_length <= audio_song.size(
            dim=0
        ), "audio_length is too large for song."

        # split song into segments:
        step = int(np.around((1 - self.overlap_ratio) * self.audio_length))
        audio_segments = audio_song.unfold(
            dimension=0, size=self.audio_length, step=step
        )
        # sanity check shape:
        assert (
            audio_segments.dim() == 2
            and audio_segments.size(dim=-1) == self.audio_length
        ), "Error with shape of split song."

        # add size 1 dimension to be compatible for future use:
        audio_segments = audio_segments.unsqueeze(dim=1)

        return audio_segments, label

    def __len__(self) -> int:
        """Returns length of dataset.

        Args: None

        Returns:
            Length of dataset (int).
        """

        return len(self.dataset)
