import pdb
import wfdb, ast
import os, pickle
import numpy as np
import pandas as pd
import neurokit2 as nk

from tqdm import tqdm
from pathlib import Path
from torch.utils import data
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler

os.makedirs("data/ptb_xl", exist_ok=True)

label_dict = {
    'NORM': 0, 
    'MI': 1, 
    'STTC': 2, 
    'CD': 3,
    'HYP': 4
}

class PTB_XL(data.Dataset):
    def __init__(self, root, sr, split="train"):
        super().__init__()
        self.root = root
        self.sr = sr
        self.win = sr * 10
        
        print("Loading ECG data...")
        
        if os.path.exists(f"data/ptb_xl/{split}_ecg.npy"):
            ecg_data = np.load(f"data/ptb_xl/{split}_ecg.npy")
            ecg_labels = np.load(f"data/ptb_xl/{split}_lab.npy")
        else:
            # Data root folder
            data_path = Path(self.root).joinpath(
                'physionet.org/files/ptb-xl/1.0.3'
            )

            patient_df = pd.read_csv(
                data_path.joinpath('ptbxl_database.csv'), 
                index_col=0
            )

            # Apple code
            patient_df = patient_df.dropna(subset=['site'])
            patient_df.scp_codes = patient_df.scp_codes.apply(lambda x: ast.literal_eval(x))

            def aggregate_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in mapping_df.index and y_dic[key] == 100:
                        tmp.append(mapping_df.loc[key].diagnostic_class)
                return list(set(tmp))
            
            # Read mapping
            mapping_df = pd.read_csv(
                data_path.joinpath('scp_statements.csv'), 
                index_col=0
            )
            mapping_df = mapping_df[mapping_df.diagnostic == 1]
            # Apply diagnostic superclass
            patient_df['diagnostic_superclass'] = patient_df.scp_codes.apply(aggregate_diagnostic)
            
            # the suggest split from the data repo:
            # Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. 
            # We therefore propose to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.
            if split == "train":
                data_df = patient_df.loc[patient_df['strat_fold'] <= 8]
            elif split == "dev":
                data_df = patient_df.loc[patient_df['strat_fold'] == 9]
            elif split == "test":
                data_df = patient_df.loc[patient_df['strat_fold'] == 10]
            
            ecg_data, ecg_labels = list(), list()
            for idx in tqdm(range(len(data_df))):
                # Read file name
                file_name = data_df.filename_lr.values[idx]
                if not Path.exists(Path(str(data_path.joinpath(f"{file_name}.dat")))): continue
                # Labels
                raw_labels = data_df.diagnostic_superclass.values[idx]
                label = list(np.zeros(len(label_dict), dtype=int))
                for raw_label in raw_labels: label[label_dict[raw_label]] = 1
                
                data = wfdb.rdsamp(str(data_path.joinpath(file_name)))[0][:, 0]
                data = nk.ecg_clean(data, sampling_rate=self.sr)
                
                mean, std = np.mean(data, axis=0), np.std(data, axis=0)
                ecg = (data - mean) / (std + 1e-5)
                
                ecg_data.append(ecg)
                ecg_labels.append(label)
            
            ecg_data = np.array(ecg_data)
            ecg_labels = np.array(ecg_labels)
            
            np.save(f"data/ptb_xl/{split}_ecg.npy", ecg_data)
            np.save(f"data/ptb_xl/{split}_lab.npy", ecg_labels)

        self.samples = ecg_data
        self.labels = ecg_labels

        print(f"Loaded {len(self.labels)} ECG samples in total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg = self.samples[idx]
        lab = self.labels[idx]
        
        return ecg, lab, 0


if __name__ == "__main__":
    root = "/media/data/public-data/Health/ptb-xl"
    train_dataset   = PTB_XL(root, sr=100, split="train")
    dev_dataset     = PTB_XL(root, sr=100, split="dev")
    test_dataset    = PTB_XL(root, sr=100, split="test")
    print(train_dataset[0][0].shape, train_dataset[0][1])
