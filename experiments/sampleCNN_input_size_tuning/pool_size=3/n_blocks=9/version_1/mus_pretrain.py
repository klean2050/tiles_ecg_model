"""Script to perform music (audio) pretraining."""


from vcmr.utils.mus_pretrain_function import mus_pretrain


if __name__ == "__main__":

    # config file:
    config_file = "experiments/sampleCNN_input_size_tuning/pool_size=3/n_blocks=9/version_1/config_mus.yaml"
    # log directory naming:
    exp_name = "sampleCNN_input_size_tuning"
    exp_run_name = "pool_size=3/n_blocks=9/version_1"
    # GPUs to use:
    gpus_to_use = "1, 2, 3"
    # verbosity:
    verbose = 1

    # perform music pretraining:
    print("\n")
    mus_pretrain(config_file=config_file, exp_name=exp_name, exp_run_name=exp_run_name, gpus_to_use=gpus_to_use, verbose=verbose)
    print("\n\n")

