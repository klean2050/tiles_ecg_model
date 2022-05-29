import os, torch, numpy as np
import torchaudio, math
from glob import glob
from torch import Tensor
from typing import Tuple
from torch.utils import data


class AUDIOVISUAL(data.Dataset):
    """Create a Dataset for music video-clip files.
    Args:
        root (str): Path to the directory where the dataset is found.
    """

    def __init__(
        self,
        root: str,
        n_classes: int = 1,
    ) -> None:
        super(AUDIOVISUAL, self).__init__(root)

        self.audio_path = os.path.join(root, "audios")
        self.video_path = os.path.join(root, "videos")
        self.n_classes = n_classes

        self.fl1 = glob(
            os.path.join(self.audio_path, "**", "*.wav"),
            recursive=True,
        )
        self.fl1 = sorted(self.fl1)
        self.fl2 = glob(
            os.path.join(self.video_path, "*.npy"),
            recursive=True,
        )
        self.fl2 = sorted(self.fl2)

        if len(self.fl1) == 0:
            raise RuntimeError(
                "Dataset not found. Please place files in the {} folder.".format(
                    self.audio_path
                )
            )

    def file_path(self, n: int) -> str:
        n = n - 1 if n else n
        return self.fl1[n]

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        filepath = self.file_path(n)
        audio, _ = torchaudio.load(filepath)

        matching_video, interval = filepath[-19:-8], int(filepath[-7:-4])
        matching_video = os.path.join(self.video_path, f"video-{matching_video}.npy")
        video = torch.from_numpy(np.load(matching_video)).float()

        # 10 sec from features of 1.5 sec
        start = math.ceil(interval*10/1.5)
        aligned_segment = video[start : start + 6]
        return audio, aligned_segment, []

    def __len__(self) -> int:
        return len(self.fl1)