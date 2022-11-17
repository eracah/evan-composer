import argparse
import logging
import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from composer import DataSpec, Time, Trainer
from composer.algorithms import EMA, SAM, ChannelsLast, MixUp
from composer.callbacks import CheckpointSaver, ImageVisualizer, LRMonitor, SpeedMonitor
from composer.datasets.ade20k import (ADE20k, PadToSize, PhotometricDistoration, RandomCropPair, RandomHFlipPair,
                                      RandomResizePair)
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.loggers import CometMLLogger, WandBLogger
from composer.loss import DiceLoss, soft_cross_entropy
from composer.metrics import CrossEntropy, MIoU
from composer.models import ComposerClassifier
from composer.models.deeplabv3.model import deeplabv3
from composer.optim import CosineAnnealingScheduler, DecoupledSGDW
from composer.utils import dist

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

IMAGENET_CHANNEL_MEAN = (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255))
IMAGENET_CHANNEL_STD = (int(0.229 * 255), int(0.224 * 255), int(0.225 * 255))

ADE20K_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
ADE20K_FILE = 'ADEChallengeData2016.zip'

parser = argparse.ArgumentParser()

parser.add_argument('--train_batch_size', help='Train dataloader per-device batch size', type=int, default=128)
parser.add_argument('--eval_batch_size', help='Validation dataloader per-device batch size', type=int, default=128)

args = parser.parse_args()

def _main():
    # Divide batch size by number of devices
    if dist.get_world_size() > 1:
        args.train_batch_size = args.train_batch_size // dist.get_world_size()
        args.eval_batch_size = args.eval_batch_size // dist.get_world_size()

    # Train dataset code
    logging.info('Building train dataloader')

    data_dir = './'
    data_dir = os.path.join(data_dir, 'ADEChallengeData2016')
    if not os.path.exists(data_dir):
        torchvision.datasets.utils.download_and_extract_archive(url=ADE20K_URL,
                                                                download_root=data_dir,
                                                                filename=ADE20K_FILE,
                                                                remove_finished=True)
    # Adjust the data_dir to include the extracted directory
    data_dir = os.path.join(data_dir, 'ADEChallengeData2016')
    #os.makedirs(data_dir, exist_ok=True)

    # Training transforms applied to both the image and target
    train_both_transforms = torch.nn.Sequential(
        RandomResizePair(
            min_scale=0.5,
            max_scale=2.0,
            base_size=(512, 512),
        ),
        RandomCropPair(
            crop_size=(512, 512),
            class_max_percent=0.75,
            num_retry=10,
        ),
        RandomHFlipPair(),
    )

    # Training transforms applied to the image only
    train_image_transforms = torch.nn.Sequential(
        PhotometricDistoration(
            brightness=32. / 255,
            contrast=0.5,
            saturation=0.5,
            hue=18. / 255,
        ),
        PadToSize(
            size=(512, 512),
            fill=IMAGENET_CHANNEL_MEAN,
        ),
    )

    # Training transforms applied to the target only
    train_target_transforms = PadToSize(size=(512, 512), fill=0)

    # Create ADE20k train dataset
    train_dataset = ADE20k(
        datadir=data_dir,
        split='training',
        image_transforms=train_image_transforms,
        target_transforms=train_target_transforms,
        both_transforms=train_both_transforms,
    )

    # Create ADE20k train dataloader

    train_sampler = None
    if dist.get_world_size():
        # Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup
        train_sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        drop_last=True,  # Prevents using a smaller batch at the end of an epoch
        sampler=train_sampler,
        collate_fn=pil_image_collate,
        persistent_workers=True,
    )

    # DataSpec enables image normalization to be performed on-GPU, marginally relieving dataloader bottleneck
    train_dataspec = DataSpec(dataloader=train_dataloader,
                              device_transforms=NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                                                std=IMAGENET_CHANNEL_STD,
                                                                ignore_background=True))
    logging.info('Built train dataloader\n')

    # Validation dataset code
    logging.info('Building evaluation dataloader')

    # Validation image and target transformations
    image_transforms = transforms.Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR)
    target_transforms = transforms.Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)

    # Create ADE20k validation dataset
    val_dataset = ADE20k(datadir=data_dir,
                         split='validation',
                         both_transforms=None,
                         image_transforms=image_transforms,
                         target_transforms=target_transforms)

    #Create ADE20k validation dataloader

    val_sampler = None
    if dist.get_world_size():
        # Nifty function to instantiate a PyTorch DistributedSampler based on your hardware
        val_sampler = dist.get_sampler(val_dataset, drop_last=False, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
        collate_fn=pil_image_collate,
        persistent_workers=True,
    )

    # DataSpec enables image normalization to be performed on-GPU, marginally relieving dataloader bottleneck
    val_dataspec = DataSpec(dataloader=val_dataloader,
                            device_transforms=NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                                              std=IMAGENET_CHANNEL_STD,
                                                              ignore_background=True))
    logging.info('Built validation dataset\n')

    logging.info('Building Composer DeepLabv3+ model')

    # Create a DeepLabv3+ model
    model = deeplabv3(
        num_classes=150,
        backbone_arch='resnet50',
        backbone_weights='IMAGENET1K_V2',
        sync_bn=False,
        use_plus=True,
    )

    # Initialize the classifier head only since the backbone uses pre-trained weights
    def weight_init(module: torch.nn.Module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight)
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    model.classifier.apply(weight_init)  # type: ignore Does not recognize classifier as a torch.nn.Module

    # Loss function to use during training
    # This ignores index -1 since the NormalizationFn transformation sets the background class to -1
    dice_loss_fn = DiceLoss(softmax=True, batch=True, ignore_absent_classes=True)

    def combo_loss(output, target):
        loss = {}
        loss['cross_entropy'] = soft_cross_entropy(output, target, ignore_index=-1)
        loss['dice'] = dice_loss_fn(output, target)
        loss['total'] = 0.375 * loss['cross_entropy'] + 1.125 * loss['dice']
        return loss

    # Training and Validation metrics to log throughout training
    train_metrics = MetricCollection([CrossEntropy(ignore_index=-1), MIoU(num_classes=150, ignore_index=-1)])
    val_metrics = MetricCollection([CrossEntropy(ignore_index=-1), MIoU(num_classes=150, ignore_index=-1)])

    # Create a ComposerClassifier using the model, loss function, and metrics
    composer_model = ComposerClassifier(module=model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=combo_loss)

    logging.info('Built Composer DeepLabv3+ model\n')

    logging.info('Building optimizer and learning rate scheduler')
    # Optimizer
    optimizer = DecoupledSGDW(composer_model.parameters(), lr=0.08, momentum=0.9, weight_decay=5.0e-05)

    lr_scheduler = CosineAnnealingScheduler()

    logging.info('Built optimizer and learning rate scheduler')

    logging.info('Building callbacks: SpeedMonitor, LRMonitor, and CheckpointSaver')
    speed_monitor = SpeedMonitor(window_size=50)  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate

    # Callback for checkpointing
    checkpoint_saver = CheckpointSaver(folder='./checkpoints', save_interval='1ep')
    logging.info('Built callbacks: SpeedMonitor, LRMonitor, and CheckpointSaver\n')

    logging.info('Built algorithm recipes\n')

    # Weight and Biases logger if specified in commandline
    wandb_logger = WandBLogger()
    comet_logger = CometMLLogger()
    image_viz = ImageVisualizer(interval='10ba', mode='segmentation', num_images=2)
    logging.info('Built Weights and Biases logger')

    # Create the Trainer!
    logging.info('Building Trainer')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = 'amp' if device == 'gpu' else 'fp32'  # Mixed precision for fast training when using a GPU
    grad_accum = 'auto' if device == 'gpu' else 1  # If on GPU, use 'auto' gradient accumulation
    trainer = Trainer(
        run_name='evan-test-log-images-15',
        model=composer_model,
        train_dataloader=train_dataspec,
        eval_dataloader=val_dataspec,
        eval_interval='1ep',
        optimizers=optimizer,
        schedulers=lr_scheduler,
        #   algorithms=algorithms,
        loggers=[wandb_logger, comet_logger],
        max_duration='1ep',
        callbacks=[speed_monitor, lr_monitor, checkpoint_saver, image_viz],
        #   load_path=args.load_checkpoint_path,
        device=device,
        precision=precision,
        grad_accum=grad_accum,
        seed=12)
    logging.info('Built Trainer\n')

    # Start training!
    logging.info('Train!')
    trainer.fit()


if __name__ == '__main__':
    _main()
