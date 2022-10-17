"""Contains functions to create plots of results of supervised models on music tagging."""


import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Dict


# supported values of certain parameters:
ALL_TAG_CATEGORIES = ["all_tags", "instrument", "genre", "mood", "vocals", "other"]
ALL_METRIC_TYPES = ["ROC-AUC", "PR-AUC"]


def plot_tag_categories(metrics_full: Dict, metric_type: str, dataset_name: str, tag_categories: Union[List, str] = "all", save_path: str = None, bar_width: int = 0.25, fig_num: int = 1) -> None:
    """Creates bar graph of music-tagging performance on tag categories.

    Args:
        metrics_full (dict): Dictionary containing tag-category-wise performance metrics on a dataset.
        metric_type (str): Performance metric to plot.
        dataset_name (str): Name of dataset (only used for plot title).
        tag_categories (list | str): Tag categories to plot.
        save_path (str): Path for saving plot.
        bar_width (int): Bar width for bar graph.
        fig_num (int): matplotlib figure number.
    
    Returns: None
    """

    # validate parameters:
    if metric_type not in ALL_METRIC_TYPES:
        raise ValueError("Invalid performance metric.")
    
    # squeeze dictionary:
    metrics = squeeze_metrics_dict(metrics_full, tag_categories=tag_categories, metric_type=metric_type)
    # get category names:
    for modality in metrics.keys():
        category_names = list(metrics[modality].keys())
    
    # set up bar heights (metric values for all tag categories):
    bar_heights = {}
    for modality in metrics.keys():
        bar_heights[modality] = list(metrics[modality].values())
    
    # set up positions of bars:
    n_categories = len(category_names)
    bar_pos = {}
    k = 0
    for modality in metrics.keys():
        bar_pos[modality] = np.arange(n_categories) + k * bar_width
        k += 1
    
    # create bar graph:
    plt.subplots(num=fig_num)
    for modality in metrics.keys():
        plt.bar(bar_pos[modality], bar_heights[modality], width=bar_width, align="edge", label=modality)
    # label bars (pairs) by tag category:
    plt.xticks(np.arange(n_categories) + 0.5 * len(metrics.keys()) * bar_width, category_names)
    # annotate plot:
    plt.title("{} by Tag Category for {} Dataset".format(metric_type, dataset_name))
    plt.ylabel(metric_type)
    plt.legend(loc="upper right")

    # save plot:
    if save_path is not None:
        plt.savefig(save_path)


def squeeze_metrics_dict(metrics_full: Dict, tag_categories: Union[List, str] = "all", metric_type: str = "all") -> Dict:
    """Squeezes (full) metrics dictionary in order to keep only required key levels and values.

    Args:
        metrics_full (dict): Dictionary containing tag-category-wise performance metrics on a dataset.
        tag_categories (list | str): Tag categories to keep.
        metric_type (str): Performance metric(s) to keep.
    
    Returns:
        metrics_squeezed (dict): Squeezed metrics dictionary.
            if metric_type != "all:
                Dictionary format: metrics_squeezed[modality][category]
            else:
                Dictionary format: metrics_squeezed[modality][category][metric]
    """

    # validate and set parameters:
    if metric_type != "all" and metric_type not in ALL_METRIC_TYPES:
        raise ValueError("Invalid performance metric.")
    if type(tag_categories) == str:
        if tag_categories == "all":
            tag_categories = ALL_TAG_CATEGORIES
        elif tag_categories in ALL_TAG_CATEGORIES:
            tag_categories = [tag_categories]
        else:
            raise ValueError("Invalid tag category.")
    elif type(tag_categories) == list:
        for category in tag_categories:
            if category not in ALL_TAG_CATEGORIES:
                raise ValueError("Invalid tag category.")
        tag_categories = tag_categories
    else:
        raise ValueError("tag_categories is of an invalid data type.")
    
    # squeeze dictionary:
    metrics_squeezed = {}
    for model in metrics_full.keys():
        for modality in metrics_full[model].keys():
            metrics_squeezed[modality] = {}
            for version in metrics_full[model][modality].keys():
                for overlap in metrics_full[model][modality][version].keys():
                    for method in metrics_full[model][modality][version][overlap].keys():                        
                        for category in tag_categories:
                            # if keeping only 1 metric, don't include a key level for metric type:
                            if metric_type != "all":
                                metrics_squeezed[modality][category] = metrics_full[model][modality][version][overlap][method][category][metric_type]
                            # else (keeping all metrics), include a key level for metric type:
                            else:
                                metrics_squeezed[modality][category] = {}
                                for metric in metrics_full[model][modality][version][overlap][method][category].keys():
                                    metrics_squeezed[modality][category][metric] = metrics_full[model][modality][version][overlap][method][category][metric]
    
    return metrics_squeezed

