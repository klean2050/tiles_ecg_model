"""Script for generating valid input sizes of SampleCNN."""


import numpy as np
import pandas as pd


# csv file for saving results:
save_file = "experiments/sampleCNN_input_size_tuning/sample_cnn_input_sizes_all.csv"
# sampling rate:
Fs = 16000
# range of input sizes to try:
input_size_range_sec = np.array([3.0, 7.1])
input_size_range = np.around(Fs * input_size_range_sec).astype(int)
# choices of (relevant) model architecture hyperparameters:
pool_size_choices = np.arange(2, 4+1, dtype=int)
first_block_params_conv_size_choices = np.arange(2, 5+1, dtype=int)


if __name__ == "__main__":
    print("\n\n")

    # find valid input sizes in specified range:
    print("Finding valid input sizes in range [{} s, {} s] = [{} samples, {} samples]...".format(input_size_range_sec[0], input_size_range_sec[1], input_size_range[0], input_size_range[1]))
    hyperparam_choices = {}
    count = 1
    for pool_size in pool_size_choices:
        for first_conv_size in first_block_params_conv_size_choices:
            # find minimum number of blocks for which input size is in range:
            n_blocks = 0
            input_size = first_conv_size * np.power(pool_size, n_blocks)
            while input_size < input_size_range[0]:
                n_blocks += 1
                input_size = first_conv_size * np.power(pool_size, n_blocks)
            
            # find valid input sizes in specified range:
            while input_size <= input_size_range[1]:
                # save model architecture hyperparameters:
                hyperparam_choices["config_" + str(count)] = {
                    "pool_size": pool_size,
                    "n_blocks": n_blocks,
                    "first_conv_size": first_conv_size,
                    "input_size": input_size,
                    "input_size_sec": "{:.3}".format(input_size / Fs)
                }
                count += 1
                # increase number of blocks:
                n_blocks += 1
                input_size = first_conv_size * np.power(pool_size, n_blocks)
    
    # print all valid model architecture configurations:
    pool_size_curr = 0
    first_conv_size_curr = 0
    for config_num in hyperparam_choices.keys():
        # for nice display:
        pool_size_new = hyperparam_choices[config_num]["pool_size"]
        if pool_size_new != pool_size_curr:
            print("\n\n\npool_size = {}:\n".format(pool_size_new))
            pool_size_curr = pool_size_new
        first_conv_size_new = hyperparam_choices[config_num]["first_conv_size"]
        if first_conv_size_new != first_conv_size_curr:
            print()
            first_conv_size_curr = first_conv_size_new
        
        print("{}: ".format(config_num), end="")
        config = hyperparam_choices[config_num]
        print(config)
    
    # save all valid model architecture configurations to csv file:
    df = pd.DataFrame.from_dict(hyperparam_choices, orient="index")
    df.to_csv(path_or_buf=save_file)

    print("\n\n")

