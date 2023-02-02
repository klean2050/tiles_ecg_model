import os, wfdb, pandas as pd
import numpy as np, neurokit2 as nk
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import normalize


class DriveDB(Dataset):
    def __init__(self, root, split, sr, streams):
        super().__init__()
        self.root, self.sr = root, sr
        self.streams = streams + ["EDA"]
        self.data = {s: [] for s in self.streams + ["ECG"]}
        self.names = []

        save_ecg = []
        for file in tqdm(os.listdir(self.root)):
            if not file.endswith(".dat"):
                continue
            if file.split(".")[0] not in split:
                continue

            ### data loading
            signals, fields = wfdb.rdsamp(self.root + os.path.splitext(file)[0])
            this_df = pd.DataFrame(signals, columns=fields["sig_name"])
            not_there = [i for i in streams if i not in fields["sig_name"]]
            if not_there != []:
                continue

            ### ECG processing: interpolate, partition and clean
            signal = this_df["ECG"].to_numpy()
            signal = resample(signal, int(100 * len(signal) / 15.5))
            cutoff = len(signal) % (300 * 100)
            signal = signal[: len(signal) - cutoff].reshape(-1, 300 * 100)
            for i in range(len(signal)):
                signal[i] = nk.ecg_clean(signal[i], sampling_rate=100)
            save_ecg.append(signal)

            ### respiration processing
            if "RESP_rate" in self.streams:
                try:
                    signal = this_df["RESP"].to_numpy()
                    _ = this_df["HR"].to_numpy()
                except:
                    continue
                out, _ = nk.rsp_process(signal, sampling_rate=fields["fs"])
                # this_df["RESP_amp"] = out[["RSP_Amplitude"]]
                this_df["RESP_rate"] = out[["RSP_Rate"]]

            ### lowpass filter (0.05Hz) + downsample
            this_df.index = pd.date_range(
                start="1/1/2023", periods=len(this_df), freq="0.065S"
            )
            down = int(1 / self.sr)
            this_df = this_df.apply(self.LP_filter).resample(f"{down}S").mean()

            ### specify and process ground truth
            try:
                gt_signal = this_df["hand GSR"]
            except:
                gt_signal = this_df["foot GSR"]
            this_df["EDA"] = self.LP_filter(gt_signal, freq=sr, cut=0.01)

            ### divide 5-minute segments
            grouper = this_df.groupby(pd.Grouper(freq="5T"))
            for _, s in grouper:
                s = s[self.streams].to_dict(orient="list")
                for stream in s.keys():
                    self.data[stream].append(s[stream])

            self.names.extend([file.split(".")[0]] * (len(grouper) - 1))
            for stream in self.data.keys():
                self.data[stream] = self.data[stream][:-1]

        ### stack input time-series
        self.data["ECG"] = save_ecg
        for stream in self.data.keys():
            self.data[stream] = np.vstack(self.data[stream])

        ### normalize data subject-wise
        for stream in self.data.keys():
            self.data[stream] = np.vstack(self.data[stream])

            this_name = self.names[0]
            temp, final = [], []
            for i in range(len(self.data["ECG"])):
                if self.names[i] == this_name:
                    temp.append(self.data[stream][i])
                else:
                    this_name = self.names[i]
                    final.append(normalize(np.vstack(temp)))
                    temp = [self.data[stream][i]]

            final.append(normalize(np.vstack(temp)))
            self.data[stream] = np.vstack(final)

    def __len__(self):
        return len(self.data["ECG"])

    def __getitem__(self, i):
        out = {}
        for stream in self.data:
            out[stream] = self.data[stream][i]

        out["ECG"] = out["ECG"].reshape(-1, 1000)
        return out, self.names[i]

    def get_modalities(self):
        mods = list(self.data.keys())
        mods.remove("EDA")
        return mods

    def LP_filter(self, ts=None, freq=15.5, cut=0.05):
        b, a = butter(3, cut, fs=freq, btype="low")
        return filtfilt(b, a, ts)


if __name__ == "__main__":
    root = "/home/kavra/Datasets/physionet.org/files/drivedb/1.0.0/"
    subjects = [i.split(".")[0] for i in os.listdir(root)]
    subjects = list(set([i for i in subjects if "drive" in i]))
    dataset = DriveDB(root=root, sr=0.5, streams=["HR"], split=subjects)
    print("Success! Dataset loaded with length:", len(dataset))

    data, name = dataset[10]
    print(name, data.keys(), data["ECG"].shape, data["EDA"].shape)
