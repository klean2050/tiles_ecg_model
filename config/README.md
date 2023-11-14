# How to configure available modes

The following variables are set for logging/admin purposes:
```
seed: 42
val_freq: 1     # how often to validate (runs validation every val_freq epochs)
log_freq: 10    # how often to log (logs every log_freq steps)
workers: 4
n_cuda: "0"
bit_precision: 16
log_dir: "runs"
experiment_version: null
```

The following variables should be set for each dataset separately:
```
dataset: "DATASET_NAME"
dataset_dir: "/FULL/PATH/TO/DATASET/"
sr: sampling-rate 
```

### Pre-training the proposed S4 model

The proposed configuration is the following. Any variable that is not mentioned here should not be altered.
```
model_type: s4

```

### Pre-training of a ResNet model

### pre-training using contrastive learning

### Fine-tuning Setup

When using a pre-trained checkpoint, make sure the variables defined in pre-training for the specific experiment remain the same.