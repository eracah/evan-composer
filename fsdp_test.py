import torch
from torch.utils.data import DataLoader, Dataset
from typing import Sequence

from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from composer.models import ComposerClassifier
import torch.distributed

class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]


class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 1, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, 5, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(5, num_classes, bias=False),
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net, num_classes=num_classes)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        # self.fc1 = fc1
        # self.fc2 = fc2



def get_trainer(save_folder=None,
                save_filename='ba{batch}-rank{rank}.pt',
                num_features=2,
                num_classes=2,
                fsdp_state_dict_type='full',
                load_path=None,
                autoresume=False,
                run_name=None,
                python_log_level=None,
                max_duration='2ba'
                ):
    model = SimpleModel(num_features=num_features, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_features, 1, 1), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'min_params': 1,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': 'FULL_SHARD'
        },
        save_folder=save_folder,
        max_duration=max_duration,
        save_interval='2ba',
        save_filename=save_filename,
        save_overwrite=False,
        load_path=load_path,
        progress_bar=False,
        log_to_console=False,
        autoresume=autoresume,
        run_name=run_name,
        python_log_level=python_log_level,
        save_latest_filename=None
    )
    return trainer

if __name__ == '__main__':
    from torch.distributed import checkpoint
    ## Save
    trainer = get_trainer(fsdp_state_dict_type='local')
    msd = trainer.state.model.state_dict()
    if dist.get_global_rank() == 0:
        print(msd)
    fsw = checkpoint.FileSystemWriter(f"./checkpoint/")
    tsd = trainer.state.state_dict()['model']
    md = checkpoint.save_state_dict(tsd, fsw)
    print(md)
    
    
    # # trainer.fit()


    ## Load
    # trainer2 = get_trainer(fsdp_state_dict_type='local')
    fsl = checkpoint.FileSystemReader(f"./checkpoint/")
    # with torch.no_grad():
    #     checkpoint.load_state_dict(state_dict=trainer2.state.state_dict()['model'], storage_reader=fsl)
   