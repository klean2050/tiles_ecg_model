"""Script to run analysis of supervised models on music tagging."""


import matplotlib.pyplot as plt
import json

from vcmr.utils import find_best_model


# script options:
dataset_names = ["MagnaTagATune", "MTG-Jamendo"]
metric_files = ["results/magnatagatune/global_metrics.json", "results/mtg-jamendo-dataset/global_metrics.json"]
model_modalities = ["audio", "multimodal"]
metric_types = ["ROC-AUC", "PR-AUC"]
verbose = 1


if __name__ == "__main__":

    # -----------
    # BEST MODELS
    # -----------

    for (dataset, metric_file) in zip(dataset_names, metric_files):
        if verbose:
            print("\n\n\n\nFinding best models for {} dataset...\n".format(dataset))
        
        # load metrics:
        with open(metric_file, "r") as json_file:
            global_metrics = json.load(json_file)
        # find best-performing models:
        for modality in model_modalities:
            for metric_type in metric_types:
                best_model, best_score = find_best_model(
                    global_metrics,
                    model_modality=modality,
                    metric_type=metric_type
                )
                if verbose:
                    print("\nBest {} model, wrt {}:".format(modality, metric_type))
                    print("Best model:  {}".format(best_model))
                    print("Best {}: {:.2f} % ".format(metric_type, 100 * best_score))
    
    """
    # --------------
    # VISUALIZATIONS
    # --------------

    if verbose:
        print("\nMaking visualizations...")
    
    # create directories for saving plots:
    plots_dir = os.path.join(main_results_dir, "plots", "")
    os.makedirs(plots_dir, exist_ok=True)

    # create bar graphs for tag-wise performance:
    with open(os.path.join(audio_results_dir, "classes_dict.pickle"), "rb") as fp:
        a = pickle.load(fp)
        a0 = {k: v[0] for k, v in a.items()}
        a1 = {k: v[1] for k, v in a.items()}
    with open(os.path.join(multimodal_results_dir, "classes_dict.pickle"), "rb") as fp:
        b = pickle.load(fp)
        b0 = {k: v[0] for k, v in b.items()}
        b1 = {k: v[1] for k, v in b.items()}
    _ = make_graphs(
        mus=a0,
        vid=b0,
        name="prs",
        save_path=os.path.join(plots_dir, "PR-AUC.png")
    )
    tops = make_graphs(
        mus=a1,
        vid=b1,
        name="rcs",
        save_path=os.path.join(plots_dir, "ROC-AUC.png")
    )
    """


    print("\n\n")

