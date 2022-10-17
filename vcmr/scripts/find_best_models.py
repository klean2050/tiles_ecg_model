"""Script to find best supervised models (on music tagging)."""


import os
import json

from vcmr.utils import find_best_model


# script options:
results_dir = "results"
datasets = ["magnatagatune", "mtg-jamendo-dataset"]
model_modalities = ["audio", "multimodal"]
metric_types = ["ROC-AUC", "PR-AUC"]
verbose = 1


if __name__ == "__main__":

    best_models = {}
    for dataset in datasets:
        if verbose:
            print("\n\n\n\nFinding best models for {} dataset...\n".format(dataset))
        best_models[dataset] = {}

        # load metrics:
        with open(
            os.path.join(results_dir, dataset, "global_metrics.json"), "r"
        ) as json_file:
            global_metrics = json.load(json_file)
        # find best-performing models:
        for modality in model_modalities:
            best_models[dataset][modality] = {}
            for metric_type in metric_types:
                best_model, best_score = find_best_model(
                    global_metrics, model_modality=modality, metric_type=metric_type
                )
                best_models[dataset][modality][metric_type] = {
                    "best_model": best_model,
                    "best_score": best_score,
                }
                if verbose:
                    print("\nBest {} model, wrt {}:".format(modality, metric_type))
                    print("Best model:  {}".format(best_model))
                    print("Best {}: {:.2f} % ".format(metric_type, 100 * best_score))
        # save dictionary containing best models for single dataset (all modalities and all metric types) to json file:
        with open(
            os.path.join(results_dir, dataset, "best_models.json"), "w"
        ) as json_file:
            json.dump(best_models[dataset], json_file, indent=3)

    # save dictionary containing best models for all datasets (all modalities and all metric types) to json file:
    with open(os.path.join(results_dir, "best_models.json"), "w") as json_file:
        json.dump(best_models, json_file, indent=3)

    print("\n\n")

