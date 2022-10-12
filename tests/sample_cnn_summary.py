"""Script for sanity testing of SampleCNN PyTorch classes."""


from torchinfo import summary
from vcmr.models.sample_cnn import SampleCNN


if __name__ == "__main__":
    print("\n")

    # model architecture hyperparameters:
    n_blocks = 9
    n_channels = [128, 128, 256, 256, 256, 256, 256, 256, 512]
    output_size = 512
    conv_kernel_size = 3
    pool_size = 3
    activation = "relu"
    first_block_params = {"out_channels": 128, "conv_size": 3}
    # input dimensions:
    input_size = 59049
    batch_size = 1
    # depth of nested layers to display:
    summary_depth = 3

    # testing SampleCNN class:
    sample_cnn = SampleCNN(
        n_blocks=n_blocks,
        n_channels=n_channels,
        output_size=output_size,
        conv_kernel_size=conv_kernel_size,
        pool_size=pool_size,
        activation=activation,
        first_block_params=first_block_params,
        input_size=input_size,
    )
    summary(sample_cnn, input_size=(batch_size, 1, input_size), depth=summary_depth)
    print("\n")
