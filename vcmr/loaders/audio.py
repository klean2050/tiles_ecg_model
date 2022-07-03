import os, torchaudio, random
from glob import glob
from torch import Tensor
from typing import Tuple
from torch.utils import data


class AUDIO(data.Dataset):
    """Create a Dataset for any folder of audio files.
    Args:
        root (str): Path to the directory where the dataset is found.
        src_ext_audio (str): The extension of the audio files.
    """

    def __init__(
        self,
        root: str,
        subset: str,
        src_ext_audio: str = ".wav",
        n_classes: int = 1,
    ) -> None:
        super(AUDIO, self).__init__()

        self._path = root
        self._src_ext_audio = src_ext_audio
        self.n_classes = n_classes

        self.fl = glob(
            os.path.join(self._path, "**", "*{}".format(self._src_ext_audio)),
            recursive=True,
        )
        random.Random(42).shuffle(self.fl)
        bound = int(0.9 * len(self.fl))
        self.fl = self.fl[:bound] if subset == "train" else self.fl[bound:]

        if len(self.fl) == 0:
            raise RuntimeError(
                "Dataset not found. Please place the audio files in the {} folder.".format(
                    self._path
                )
            )

    def file_path(self, n: int) -> str:
        n = n - 1 if n else n
        return self.fl[n]

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        filepath = self.file_path(n)
        audio, _ = torchaudio.load(filepath)
        return audio, []

    def __len__(self) -> int:
        return len(self.fl)
