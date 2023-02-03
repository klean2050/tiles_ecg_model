import numpy as np, pandas as pd
import os, neurokit2 as nk
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def process_prcp(prcp):

    # load ECG of participant
    df = pd.read_csv(os.path.join(path, prcp))
    # make datetime timestamp
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    except:
        df["Timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop(columns=["timestamp", "record_id"])

    # sort based on timestamp and get ECG
    df = df.set_index("Timestamp").sort_index()["raw_ecg"]

    final_samples = []
    prcp_samples = [
        g for _, g in df.groupby([(df.index - df.index[0]).astype("timedelta64[5m]")])
    ]
    for sample in prcp_samples:
        signal = sample.to_numpy()
        if len(signal) == 15 * 250:
            # downsample to 100 Hz
            signal = resample(signal, 1500)
            # clean via bandpass filtering
            sample = nk.ecg_clean(signal, sampling_rate=100)
            # append a copy to the list
            final_samples.append(sample)

    return np.vstack(final_samples)


if __name__ == "__main__":
    path = "/home/kavra/Datasets/tiles-phase1-opendataset/omsignal/ecg/"
    out_path = "data/"

    os.makedirs(out_path, exist_ok=True)
    for prcp in tqdm(os.listdir(path)):
        if f"{prcp.split('.')[0]}.npy" in os.listdir(out_path):
            continue
        samples = process_prcp(prcp)
        samples = StandardScaler().fit_transform(samples)
        np.save(out_path + prcp.split('.')[0], samples)
