"""Script for testing configurable SampleCNN PyTorch class."""


import numpy as np, torch
from vcmr.models.sample_cnn import SampleCNN


if __name__ == "__main__":
    print("\n")

    # sampling frequency:
    Fs = 22050
    # parameters for all test iterations:
    n_iters = 100
    test_forward = True
    if test_forward:
        max_input_size_sec = 100
        batch_size = 1
    verbose = 2

    # choices of model architecture parameters:
    n_blocks_choices = np.arange(1, 15 + 1, dtype=int)
    n_channels_choices = np.arange(1, 512 + 1, dtype=int)
    output_size_choices = np.arange(1, 1024 + 1, dtype=int)
    conv_kernel_size_choices = np.arange(1, 9 + 1, step=2, dtype=int)
    pool_size_choices = np.arange(1, 9 + 1, dtype=int)
    activation_choices = ["relu", "leaky_relu"]
    first_block_params_out_channels_choices = np.arange(1, 512 + 1, dtype=int)
    first_block_params_conv_size_choices = np.arange(1, 9 + 1, dtype=int)

    for i in range(n_iters):
        if verbose >= 1:
            print("\nTesting iteration {}...".format(i + 1))

        # randomly choose model architecture hyperparameters:
        n_blocks = np.random.choice(n_blocks_choices)
        n_channels = list(np.random.choice(n_channels_choices, size=n_blocks))
        output_size = np.random.choice(output_size_choices)
        conv_kernel_size = np.random.choice(conv_kernel_size_choices)
        pool_size = np.random.choice(pool_size_choices)
        activation = np.random.choice(activation_choices)
        first_block_params = {
            "out_channels": np.random.choice(first_block_params_out_channels_choices),
            "conv_size": np.random.choice(first_block_params_conv_size_choices),
        }
        input_size = first_block_params["conv_size"] * np.power(pool_size, n_blocks)

        # save hyperparameters choices to dictionary:
        hyperparams = {
            "n_blocks": n_blocks,
            "n_channels": n_channels,
            "output_size": output_size,
            "conv_kernel_size": conv_kernel_size,
            "pool_size": pool_size,
            "activation": activation,
            "first_block_params": first_block_params,
            "input_size": input_size,
        }
        if verbose == 2:
            print("Hyperparameters: {}".format(hyperparams))

        # create SampleCNN object:
        model = SampleCNN(
            n_blocks=n_blocks,
            n_channels=n_channels,
            output_size=output_size,
            conv_kernel_size=conv_kernel_size,
            pool_size=pool_size,
            activation=activation,
            first_block_params=first_block_params,
            input_size=input_size,
        )

        # test input size:
        assert model.input_size == input_size, "Error with input_size."

        # test lengths of nn.Sequential() objects:
        assert len(model.all_blocks) == n_blocks + 2, "Error with number of blocks."
        assert len(model.all_blocks[0]) == 3, "Error with length of first block."
        assert (
            len(model.all_blocks[n_blocks + 2 - 1]) == 3
        ), "Error with length of last block."
        for i in range(1, n_blocks + 1):
            assert len(model.all_blocks[i]) == 4, "Error with length of middle block."

        # test forward pass:
        if test_forward and input_size <= Fs * max_input_size_sec:
            model.eval()
            x = torch.rand((batch_size, 1, input_size))
            output = model.forward(x)
            output_shape = tuple(output.size())
            assert output_shape == (
                batch_size,
                output_size,
            ), "Error with shape of forward pass output."

    print("\n")
