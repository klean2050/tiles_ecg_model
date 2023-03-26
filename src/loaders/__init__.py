from .tiles import TILES_ECG
from .drivedb import DriveDB
from .swell_kw import SWELL_KW
from .wesad import WESAD
from .mirise import MIRISE
from .ptb_xl import PTB_XL


def get_dataset(dataset, dataset_dir, sr=100, split="train"):

    if dataset == "DriveDB":
        return DriveDB(root=dataset_dir, sr=sr, streams="ECG")
    elif dataset == "SWELL_KW":
        return SWELL_KW(root=dataset_dir, sr=sr)
    elif dataset == "WESAD":
        return WESAD(root=dataset_dir, sr=sr)
    elif dataset == "MIRISE":
        return MIRISE(root=dataset_dir, sr=sr)
    elif dataset == "PTB_XL":
        return PTB_XL(root=dataset_dir, sr=sr, split=split)
    else:
        raise NotImplementedError("Dataset not implemented")
