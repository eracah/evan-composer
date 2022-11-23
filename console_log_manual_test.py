from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from composer import Trainer
from composer.models import mnist_model
from composer.algorithms import  ChannelsLast, ColOut, CutOut, SqueezeExcite, BlurPool, Factorize
import argparse

transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
val_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_dataloader = DataLoader(train_set, batch_size=128)
eval_dataloader = DataLoader(val_set, batch_size=64)
model=mnist_model(num_classes=10)

parser = argparse.ArgumentParser()

parser.add_argument('--eval', action='store_true')
parser.add_argument('--eval-interval', type=str, default='1ep')
parser.add_argument('--log-to-console', action='store_true')
parser.add_argument('--progress-bar', action='store_true')
parser.add_argument('--console-log-interval', type=str, default='1ep')
parser.add_argument('--max-duration', type=str, default='9ep')

args = parser.parse_args()


trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader if args.eval else None,
    max_duration=args.max_duration,
    train_subset_num_batches=4,
    algorithms=[
        ChannelsLast(),
        ColOut(),
        CutOut(),
        SqueezeExcite(),
        BlurPool(),
        Factorize()
        ],
    eval_interval=args.eval_interval if args.eval else 1,
    progress_bar=args.progress_bar,
    log_to_console=args.log_to_console,
    console_log_interval=args.console_log_interval,
)

trainer.fit()
