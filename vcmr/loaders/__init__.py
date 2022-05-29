import os
from .contrastive import Contrastive, MultiContrastive
from .audio import AUDIO
from .audio_visual import AUDIOVISUAL
from .mtat import MAGNATAGATUNE
from .mtg import MTG


def get_dataset(dataset, dataset_dir, subset, download=False):

    os.makedirs(dataset_dir, exist_ok=True)
    if dataset == "audio":
        return AUDIO(root=dataset_dir)
    elif dataset == "audio_visual":
        return AUDIOVISUAL(root=dataset_dir)
    elif dataset == "magnatagatune":
        return MAGNATAGATUNE(root=dataset_dir, download=download, subset=subset)
    elif dataset == "mtg-jamendo-dataset":
        return MTG(
            root=dataset_dir,
            audio_root="/data/avramidi/mtg",
            split=0,
            subset="moodtheme",
            mode=subset,
        )
    else:
        raise NotImplementedError("Dataset not implemented")
