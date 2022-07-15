import os, yaml, torch.nn as nn, random
import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

class RandomResizedCrop(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, audio):
        max_samples = audio.shape[-1]
        start_idx = random.randint(0, max_samples - self.n_samples)
        audio = audio[..., start_idx : start_idx + self.n_samples]
        return audio, start_idx

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def visualize(dataset, features, labels, name):
        tsne = TSNE(
            n_components=2,
            perplexity=50,
            learning_rate=130,
            metric="cosine",
            init="pca"
        ).fit_transform(features)
        tx = MinMaxScaler().fit_transform(tsne[:, 0].reshape(-1, 1))[:, 0]
        ty = MinMaxScaler().fit_transform(tsne[:, 1].reshape(-1, 1))[:, 0]

        plt.style.use('dark_background')
        fig = plt.figure(dpi=200)
        cm = plt.get_cmap('rainbow')

        ax = fig.add_subplot(111)
        for label in range(labels.shape[1]):
            # find the samples of this class
            indices = [i for (i, l) in enumerate(labels) if l[label]]
            curr_tx, curr_ty = np.take(tx, indices), np.take(ty, indices)
            ax.scatter(curr_tx, curr_ty, color=cm(label*4), marker=".")
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.savefig(f"data/{dataset}_{name}")
        plt.close()