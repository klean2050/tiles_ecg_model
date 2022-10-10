"""Contains functions to perform analysis of music tagging performance."""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict


# supported values of certain parameters:
ALL_MODEL_MODALITIES = ["audio", "multimodal"]
ALL_METRIC_TYPES = ["ROC-AUC", "PR-AUC"]


def find_best_model(global_metrics: Dict, model_modality: str, metric_type: str) -> Tuple[Tuple, float]:
    """Performs analysis of music tagging performance.

    Args:
        global_metrics (dict): Dictionary containing performance metrics on a dataset.
        model_modality (str): Modality of model to search over.
        metric_type (str): Performance metric to optimize with respect to.
    
    Returns:
        best_model (tuple): Tuple describing best model.
        best_score (float): Best score (of specificed metric).
    """

    # validate and set parameters:
    if model_modality not in ALL_MODEL_MODALITIES:
        raise ValueError("Invalid model modality.")
    if metric_type not in ALL_METRIC_TYPES:
        raise ValueError("Invalid performance metric.")
    
    # find best-performing model for a single model modality and a single performance metric:
    best_model = None
    best_score = 0
    for model in global_metrics.keys():
        # for modality in model_modality     # skip loop since we only consider one model modality
        for version in global_metrics[model][model_modality].keys():
            for overlap in global_metrics[model][model_modality][version].keys():
                for method in global_metrics[model][model_modality][version][overlap].keys():
                    # update best model/score:
                    if global_metrics[model][model_modality][version][overlap][method][metric_type] > best_score:
                        best_model = (model, version, f"overlap={overlap}", method)
                        best_score = global_metrics[model][model_modality][version][overlap][method][metric_type]
    
    return best_model, best_score

