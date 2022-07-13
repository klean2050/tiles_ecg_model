import csv, torchaudio, os, torch, subprocess
from torch.utils.data import Dataset, DataLoader


def preprocess_audio(source, target, sample_rate):
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            source,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            target,
            "-loglevel",
            "quiet",
        ]
    )
    p.wait()


class MTG(Dataset):
    def __init__(self, root, audio_root, split, subset, mode, sr):
        super(MTG, self).__init__()
        self.sr = sr
        self.root = root
        self.mode = "validation" if mode == "valid" else mode

        tag_file = os.path.join(self.root, "data/tags", f"{subset}_split.txt")
        with open(tag_file, "r") as f:
            labels = f.readlines()

        self.label2idx = {}
        for idx, label in enumerate(labels):
            self.label2idx[label.strip()] = idx

        self.n_classes = len(self.label2idx.keys())

        split_file = os.path.join(
            self.root,
            "data/splits",
            f"split-{split}",
            f"autotagging_{subset}-{self.mode}.tsv",
        )
        split_tracks = []
        with open(split_file, "r") as f:
            split_file = csv.reader(f, delimiter="\t")
            for row in split_file:
                split_tracks.append(row)

        self.dataset = []
        for file in split_tracks[1:]:
            filepath = os.path.join(audio_root, file[3].replace("/", "_clip/"))
            # preprocess_audio(filepath, filepath.replace("mp3", "wav"), 22050)
            label = [self.label2idx[l] for l in file[5:]]
            binary_label = [0 if i not in label else 1 for i in range(len(labels))]
            self.dataset.append((filepath.replace("mp3", "wav"), binary_label))

    def __getitem__(self, idx):
        filepath, label = self.dataset[idx]
        audio, sr = torchaudio.load(filepath)
        resample = torchaudio.transforms.Resample(sr, self.sr)
        return resample(audio), torch.FloatTensor(label)

    def __len__(self):
        return len(self.dataset) if self.mode != "train" else int(0.9*len(self.dataset))


if __name__ == "__main__":

    dataset = MTG(
        root="/data/avramidi/mtg-jamendo-dataset",
        audio_root="/data/avramidi/mtg",
        split=0,
        subset="moodtheme",
        mode="train",
    )
    loader = DataLoader(dataset, batch_size=16)
    data = next(iter(loader))
    print(data[0].shape, data[1].shape)
