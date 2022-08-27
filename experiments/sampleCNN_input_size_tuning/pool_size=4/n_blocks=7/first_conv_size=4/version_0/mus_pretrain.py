"""Script to perform music (audio) pretraining."""


from vcmr.utils.mus_pretrain_function import mus_pretrain


if __name__ == "__main__":

    # config file:
    config_file = "experiments/sampleCNN_input_size_tuning/pool_size=4/n_blocks=7/first_conv_size=4/version_0/config_mus.yaml"
    # log directory naming:
    exp_name = "sampleCNN_input_size_tuning"
    exp_run_name = "pool_size=4/n_blocks=7/first_conv_size=4/version_0"
    # verbosity:
    verbose = 1

    # perform music pretraining:
    print("\n")
    mus_pretrain(config_file=config_file, exp_name=exp_name, exp_run_name=exp_run_name, verbose=verbose)
    print("\n\n")

