import pandas as pd, os

path = "/project/shrikann_35/rajatheb/music"
mp4_path = path + "/videos_00/"
feats_path = path + "/videos_00_feats/"
os.makedirs(feats_path, exist_ok=True)

df = pd.DataFrame(columns=["video_path", "feature_path"])
for video in os.listdir(mp4_path):
    new_row = {
        "video_path": mp4_path + video,
        "feature_path": feats_path + video + ".npy"
    }
    df = df.append(pd.Series(new_row), ignore_index=True)
df.to_csv("video_paths.csv")
