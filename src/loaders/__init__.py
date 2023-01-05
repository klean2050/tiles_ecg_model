from .tiles import TILES_ECG

def get_dataset(dataset, dataset_dir, split):
    if dataset == "tiles_ecg":
        return TILES_ECG(root=dataset_dir, split=split)
    else:
        raise NotImplementedError("Dataset not implemented")
