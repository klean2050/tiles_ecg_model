# How to use pre-trained models

### Attention

Please access the latest model checkpoints [here](https://drive.google.com/drive/folders/1L7T-fsCHiyh5XWaA7VyLHxyl_pcEK-Ar?usp=sharing) and copy to `ckpt` folder.

One can load a pre-trained model for fine-tuning or testing as follows:

```python
from src.models import ResNet1D
from src.trainers import ContrastiveLearning

# create args parser and link to Lightning trainer:
parser = argparse.ArgumentParser(description="demo")
parser = Trainer.add_argparse_args(parser)

# extract args from config file and add to parser:
config = yaml_config_hook("some_config.yaml")
for key, value in config.items():
    parser.add_argument(f"--{key}", default=value, type=type(value))
args = parser.parse_args()

# set random seed if selected:
if args.seed:
    pl.seed_everything(args.seed, workers=True)

encoder = ResNet1D(
    in_channels=args.in_channels,
    base_filters=args.base_filters,
    kernel_size=args.kernel_size,
    stride=args.stride,
    groups=args.groups,
    n_block=args.n_block,
    n_classes=args.n_classes,
)

ckpt_path = "ckpt/main_epoch=49-step=57850.ckpt"
model = ContrastiveLearning.load_from_checkpoint(ckpt_path, encoder)
```