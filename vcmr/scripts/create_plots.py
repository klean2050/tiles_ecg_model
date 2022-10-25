"""Script to create plots of results of supervised models on music tagging."""


import os
import matplotlib.pyplot as plt
import json

from vcmr.utils import plot_tag_categories


# script options:
plots_dir = "analysis/plots"
datasets = ["magnatagatune", "mtg-jamendo-dataset"]
dataset_names = ["MagnaTagATune", "MTG-Jamendo"]
metric_types = ["ROC-AUC", "PR-AUC"]
verbose = 1
# for plots by tag category:
global_metric_files = ["results/final_evaluation/magnatagatune/global_metrics.json"]
tag_categories = ["instrument", "genre", "mood", "vocals"]
include_plot_title = False


if __name__ == "__main__":
    print("\n")

    # create plots directory:
    os.makedirs(plots_dir, exist_ok=True)
    # initialize matplotlib figure number:
    fig_num = 1


    # ------------------
    # TAG-CATEGORY PLOTS
    # ------------------

    if verbose:
        print("\nCreating tag-category plots...")
    
    # create tag-category plots directory:
    tag_category_plots_dir = os.path.join(plots_dir, "tag_categories", "")
    os.makedirs(tag_category_plots_dir, exist_ok=True)
    
    # create tag-category plots:
    for i in range(len(global_metric_files)):
        # load global metrics dictionary:
        with open(global_metric_files[i], "r") as json_file:
            global_metrics = json.load(json_file)
        # create save directory:
        os.makedirs(os.path.join(tag_category_plots_dir, datasets[i], ""), exist_ok=True)

        # loop over metric types:
        for metric_type in metric_types:
            if include_plot_title:
                plot_title = "{} by Tag Category for {} Dataset".format(metric_type, dataset_names[i])
            else:
                plot_title = None
            plot_tag_categories(
                global_metrics,
                metric_type=metric_type,
                tag_categories=tag_categories,
                plot_title=plot_title,
                save_path=os.path.join(tag_category_plots_dir, datasets[i], "tag_category_" + metric_type + ".png"),
                fig_num=fig_num
            )
            fig_num += 1
    

    print("\n\n")
    # show plots:
    plt.show()

