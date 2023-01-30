import os, wfdb, pandas as pd
import numpy as np, neurokit2 as nk
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt


class DriveDB(Dataset):
    def __init__(self, root, sr, streams):
        super().__init__()
        self.root, self.sr = root, sr
        self.streams = streams + ["EDA"]
        self.data = {s: [] for s in self.streams + ["ECG"]}
        self.names = []

        for file in tqdm(os.listdir(self.root)):
            if not file.endswith(".dat"):
                continue

            ### data loading
            signals, fields = wfdb.rdsamp(self.root + os.path.splitext(file)[0])
            this_df = pd.DataFrame(signals, columns=fields["sig_name"])
            not_there = [i for i in streams if i not in fields["sig_name"]]
            if not_there != []:
                continue
            
            ### ECG processing
            signal = this_df["ECG"].to_numpy()
            # interpolate to 100 Hz
            # preprocess like TILES
            # segment into 5-min (20x15secx100Hz)
            self.data["ECG"]

            ### respiration processing
            if "RESP_rate" in self.streams:
                try:
                    signal = this_df["RESP"].to_numpy()
                    _ = this_df["HR"].to_numpy()
                except:
                    continue
                out, _ = nk.rsp_process(signal, sampling_rate=fields["fs"])
                #this_df["RESP_amp"] = out[["RSP_Amplitude"]]
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
            for _, s in this_df.groupby(pd.Grouper(freq="5T")):
                s = s[self.streams].to_dict(orient="list")
                for stream in s.keys():
                    self.data[stream].append(s[stream])
            
            self.names.append(file.split(".")[0])
            for stream in self.data.keys():
                self.data[stream] = self.data[stream][:-1]

        ### stack input time-series
        for stream in self.data.keys():
            self.data[stream] = np.vstack(self.data[stream])

    def __len__(self):
        return len(self.data[self.streams[0]])

    def __getitem__(self, stream, i):
        data = self.data[stream]
        return data[i], self.names[i]
    
    def LP_filter(self, ts=None, freq=15.5, cut=0.05):
        b, a = butter(3, cut, fs=freq, btype="low")
        return filtfilt(b, a, ts)
    

if __name__ == "__main__":
    root = "/home/kavra/Datasets/physionet.org/files/drivedb/1.0.0/"
    dataset = DriveDB(root=root, sr=0.5, streams=["HR"])
    print("Success! Dataset loaded with length:", len(dataset))