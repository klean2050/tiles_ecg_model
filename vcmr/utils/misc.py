"""Contains miscellaneous utility functions."""


import os
import torch.nn as nn
import random
import yaml


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

