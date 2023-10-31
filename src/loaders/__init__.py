from .tiles import TILES_ECG
from .swell_kw import SWELL_KW
from .wesad import WESAD
from .mirise import MIRISE
from .ptb_xl import PTB_XL
from .ludb import LUDB
from .avec16 import AVEC16
from .epic import EPIC, MULTI_EPIC


def get_dataset(dataset, dataset_dir, gtruth, sr=100, split="train", ecg_only=True):

    if dataset == "SWELL_KW":
        return SWELL_KW(root=dataset_dir, sr=sr, gtruth=gtruth)
    elif dataset == "WESAD":
        return WESAD(root=dataset_dir, sr=sr)
    elif dataset == "MIRISE":
        return MIRISE(root=dataset_dir, sr=sr, cat=gtruth)
    elif dataset == "ptb_xl":
        return PTB_XL(root=dataset_dir, sr=sr, split=split)
    elif dataset == "LUDB":
        return LUDB(root=dataset_dir)
    elif dataset == "AVEC16":
        return AVEC16(root=dataset_dir, sr=sr, split=split, category=gtruth)
    elif dataset == "EPIC" and ecg_only:
        return EPIC(
            root=dataset_dir, sr=sr, scenario=1, split=split, category=gtruth, fold=0
        )
    elif dataset == "EPIC" and not ecg_only:
        return MULTI_EPIC(
            root=dataset_dir, sr=sr, scenario=4, split=split, category=gtruth, fold=1
        )
    else:
        raise NotImplementedError("Dataset not implemented")
