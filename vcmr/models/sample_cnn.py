"""Contains a configurable PyTorch class for the SampleCNN model."""


from torch import nn
import numpy as np
from typing import List, Dict


class SampleCNN(nn.Module):
    """Configurable PyTorch class for SampleCNN.

    Attributes:
        n_blocks: Number of middle convolutional blocks (equal to n_total_blocks - 2).
        output_size: Number of output channels for last block.
        pool_size: Size of pooling kernel and pooling stride for middle blocks.
        input_size: Size (length) of input.
        all_blocks: nn.Sequential() object for all blocks.
    """

    def __init__(self, n_blocks: int, n_channels: List, output_size: int, conv_kernel_size: int, pool_size: int, activation: str = "relu", first_block_params: Dict = None, input_size: int = None) -> None:
        """Initializes SampleCNN object.

        Args:
            n_blocks: Number of middle convolutional blocks (equal to n_total_blocks - 2).
            n_channels: List of number of (output) channels for middle blocks.
                length: n_blocks
            output_size: Number of output channels for last block.
            conv_kernel_size: Size of convolutional kernel for middle blocks.
                Convolution stride is equal to 1 for middle blocks.
            pool_size: Size of pooling kernel and pooling stride for middle blocks.
                Kernel size is equal to stride to ensure even division of input size.
            activation: Type of activation to use for all blocks.
                Supported values: "relu", "leaky_relu"
            first_block_params: Dictionary describing first block, with keys/values:
                out_channels: Number of output channels.
                conv_size: Size of convolutional kernel and convolution stride (kernel size is equal to stride to ensure even division of input size).
            input_size: Size (length) of input.
        
        Returns: None
        """

        super(SampleCNN, self).__init__()

        # validate parameters:
        assert len(n_channels) == n_blocks, "Length of n_channels doesn't match n_blocks."
        assert activation == "relu" or activation == "leaky_relu", "Invalid activation type."

        # save attributes:
        self.n_blocks = n_blocks
        self.output_size = output_size
        self.pool_size = pool_size

        # if items of first_block are unspecified, set to default values:
        if first_block_params is None:
            first_block_params = {}
        if first_block_params.get("out_channels") is None:
            first_block_params["out_channels"] = n_channels[0]
        if first_block_params.get("conv_size") is None:
            first_block_params["conv_size"] = conv_kernel_size
        
        # if input_size is not None, validate input_size:
        if input_size is not None:
            assert input_size == first_block_params["conv_size"] * np.power(pool_size, n_blocks), "Input size is incompatible with network architecture."
        # else infer from network architecture:
        else:
            input_size == first_block_params["conv_size"] * np.power(pool_size, n_blocks)
        self.input_size = input_size


        # create first block:
        first_block = nn.Sequential(
            nn.Conv1d(1, first_block_params["out_channels"], kernel_size=first_block_params["conv_size"], stride=first_block_params["conv_size"], padding=0),
            nn.BatchNorm1d(first_block_params["out_channels"])
        )
        if activation == "relu":
            first_block.append(nn.ReLU())
        elif activation == "leaky_relu":
            first_block.append(nn.LeakyReLU())
        
        # create middle blocks:
        middle_blocks = []
        for i in range(n_blocks):
            # get number of input and output channels for convolutional layer:
            if i == 0:
                in_channels = first_block_params["out_channels"]
            else:
                in_channels = n_channels[i-1]
            out_channels = n_channels[i]
            
            # create block:
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=1, padding="same"),
                nn.BatchNorm1d(out_channels)
            )
            if activation == "relu":
                block.append(nn.ReLU())
            elif activation == "leaky_relu":
                block.append(nn.LeakyReLU())
            block.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0))
            # append block to list:
            middle_blocks.append(block)
        
        # create last block:
        last_block = nn.Sequential(
            nn.Conv1d(n_channels[n_blocks-1], output_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_size)
        )
        if activation == "relu":
            last_block.append(nn.ReLU())
        elif activation == "leaky_relu":
            last_block.append(nn.LeakyReLU())
        
        # concatenate all blocks and convert list to nn.Sequential() object:
        all_blocks_list = [first_block] + middle_blocks + [last_block]
        self.all_blocks = nn.Sequential(*all_blocks_list)
    
    def forward(self, x):
        # forward pass through all blocks:
        output = self.all_blocks(x)
        # remove temporal dimension (since it is size 1):
        output = output.squeeze(dim=-1)

        return output

