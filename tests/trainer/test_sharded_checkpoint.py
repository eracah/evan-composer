# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import textwrap
from typing import Dict, List

import numpy as np
import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.algorithms import EMA
from composer.models import ComposerClassifier
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from composer.utils import dist
from composer.utils.file_helpers import get_file
from composer.utils.reproducibility import get_rng_state
from tests.common import RandomClassificationDataset
from tests.common.compare import deep_compare
from tests.common.markers import world_size
from composer.utils.misc import using_torch_2

# This model is to be used explicitly for this unit test because some old reference checkpoints
# were saved using it exactly as it is. Changing this model will break test_fsdp_load_old_checkpoint.
class SimpleMLP(ComposerClassifier):

    def __init__(self, num_features: int = 32, num_classes: int = 8):
        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_classes, bias=False),
        )
        super().__init__(module=net, num_classes=num_classes)


def get_trainer(save_folder=None,
                save_filename='ba{batch}-rank{rank}.pt',
                save_overwrite=False,
                num_features=2,
                num_classes=2,
                fsdp_state_dict_type='full',
                fsdp_sharded_ckpt_prefix_dir='ba{batch}',
                load_path=None,
                autoresume=False,
                run_name=None,
                max_duration='2ba',
                precision='amp_fp16',
                sharding_strategy='FULL_SHARD',
                save_interval='2ba',
                save_weights_only=False,
                load_weights_only=False,
                algorithms=None,
                optimizer='adam'):
    model = SimpleMLP(num_features=num_features, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_features,), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=8)
    if optimizer == 'adam':
        optim = torch.optim.Adam(params=model.parameters())
    elif optimizer == 'adamw':
        optim = DecoupledAdamW(model.parameters())
    else:
        raise ValueError(f'Unsupported optimizer name {optimizer}')
    trainer = Trainer(algorithms=algorithms,
                      model=model,
                      optimizers=optim,
                      train_dataloader=dataloader,
                      fsdp_config={
                          'min_params': 16,
                          'state_dict_type': fsdp_state_dict_type,
                          'sharding_strategy': sharding_strategy,
                          'sharded_ckpt_prefix_dir': fsdp_sharded_ckpt_prefix_dir,
                      },
                      save_folder=save_folder,
                      max_duration=max_duration,
                      save_interval=save_interval,
                      save_filename=save_filename,
                      save_overwrite=save_overwrite,
                      precision=precision,
                      load_path=load_path,
                      progress_bar=False,
                      log_to_console=False,
                      autoresume=autoresume,
                      run_name=run_name,
                      save_latest_filename='latest-rank{rank}.pt',
                      save_weights_only=save_weights_only,
                      load_weights_only=load_weights_only)
    return trainer


def _compare_optims_between_state_dicts(state_dict1, state_dict2):
    # Check that optim params are equal between checkpoint and in memory optimizer
    assert len(list(state_dict1['optimizers'].keys())) == 1
    assert len(list(state_dict2['optimizers'].keys())) == 1
    optim_key1 = list(state_dict1['optimizers'].keys()).pop()
    optim_key2 = list(state_dict2['optimizers'].keys()).pop()
    assert optim_key1 == optim_key2
    state_dict1_optim_params = state_dict1['optimizers'][optim_key1]['state']
    state_dict2_optim_params = state_dict2['optimizers'][optim_key2]['state']
    state_dict1_keys = set(state_dict1_optim_params.keys())
    state_dict2_keys = set(state_dict2_optim_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""")

    for param_name in state_dict2_optim_params.keys():
        state_dict1_param_moment_dict = state_dict1_optim_params[param_name]
        state_dict2_param_moment_dict = state_dict2_optim_params[param_name]
        for moment_name in state_dict2_param_moment_dict.keys():
            state_dict1_moment = state_dict1_param_moment_dict[moment_name].cpu()
            state_dict2_moment = state_dict2_param_moment_dict[moment_name].cpu()
            assert torch.equal(
                state_dict1_moment,
                state_dict2_moment), f'Moment {moment_name} for parameter {param_name} not the same between state dicts'


def _compare_model_params_between_state_dicts(state_dict1, state_dict2):
    # Check that model params are equal between in memory mode and checkpoint
    state_dict1_model_params = state_dict1['model']
    state_dict2_model_params = state_dict2['model']

    state_dict1_keys = set(state_dict1_model_params.keys())
    state_dict2_keys = set(state_dict2_model_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys that belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""")

    for param_name in state_dict2_model_params.keys():
        state_dict1_model_tensor = state_dict1_model_params[param_name]
        state_dict2_model_tensor = state_dict2_model_params[param_name]
        assert torch.equal(state_dict1_model_tensor,
                           state_dict2_model_tensor), f'Weight named {param_name} not the same between state_dicts'


def _compare_rng_states_between_trainers(rng_state1: List[Dict], rng_state2: List[Dict]):
    assert len(rng_state1) == len(rng_state2)
    for rank, rank_state1, rank_state2 in zip(range(len(rng_state1)), rng_state1, rng_state2):
        rank_state1_keys = set(rank_state1.keys())
        rank_state2_keys = set(rank_state2.keys())
        assert len(rank_state1_keys.symmetric_difference(rank_state2_keys)) == 0, textwrap.dedent(
            f"""The two rank rng state dicts being compared for rank {rank} must have the exact same set of keys,
            but instead these keys that belong to one, but not the other:
            {rank_state1_keys.symmetric_difference(rank_state2_keys)}""")
        python_state1 = rank_state1['python']
        python_state2 = rank_state2['python']
        assert python_state1 == python_state2, f'Python rng state not the same between state_dicts for rank {rank}'

        numpy_state1 = rank_state1['numpy']
        numpy_state2 = rank_state2['numpy']
        _, keys1, pos1, has_gauss1, cached_gaussian1 = numpy_state1
        _, keys2, pos2, has_gauss2, cached_gaussian2 = numpy_state2
        assert np.allclose(keys1, keys2,
                           equal_nan=True), f'Numpy rng keys state not the same between state_dicts for rank {rank}'
        assert pos1 == pos2, f'Numpy rng pos state not the same between state_dicts for rank {rank}'
        assert has_gauss1 == has_gauss2, f'Numpy rng has_gauss state not the same between state_dicts for rank {rank}'
        assert cached_gaussian1 == cached_gaussian2, f'Numpy rng cached_gaussian state not the same between state_dicts for rank {rank}'

        torch_state1 = rank_state1['torch']
        torch_state2 = rank_state2['torch']
        assert torch.equal(torch_state1,
                           torch_state2), f'Torch rng state not the same between state_dicts for rank {rank}'

        if 'cuda' in rank_state1_keys:
            cuda_state1 = rank_state1['cuda']
            cuda_state2 = rank_state2['cuda']
            torch.equal(cuda_state1, cuda_state2), f'Cuda rng state not the same between state_dicts for rank {rank}'


def _compare_metrics_between_state_dicts(state_dict1, state_dict2):
    # Check that metric states are equal between in memory mode and checkpoint
    state_dict1_train_metrics = state_dict1['train_metrics']
    state_dict2_train_metrics = state_dict2['train_metrics']

    state_dict1_eval_metrics = state_dict1['eval_metrics']
    state_dict2_eval_metrics = state_dict2['eval_metrics']

    deep_compare(state_dict1_train_metrics, state_dict2_train_metrics)
    deep_compare(state_dict1_eval_metrics, state_dict2_eval_metrics)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
@pytest.mark.parametrize('autoresume', [True, False])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load(world_size, tmp_path: pathlib.Path, autoresume: bool, precision: str,
                                   optimizer: str):
    if autoresume:
        run_name = 'my-cool-autoresume-run'
    else:
        run_name = None
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        optimizer=optimizer,
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        load_path=load_path,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        max_duration='4ba',
        optimizer=optimizer,
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    # Continue to fit to make sure we can continue training.
    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@pytest.mark.remote
@world_size(2)
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD', 'SHARD_GRAD_OP'])
@pytest.mark.parametrize('state_dict_type', ['full', 'sharded', 'local'])
@pytest.mark.parametrize('composer_version', [
    pytest.param(
        '0.13.5',
        marks=pytest.mark.filterwarnings(
            r'ignore:ShardedGradScaler is not in the state_dict. Its state will not be restored.:UserWarning')),
    '0.14.0', '0.14.1'
])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_load_old_checkpoint(world_size, tmp_path: pathlib.Path, precision: str, sharding_strategy: str,
                                  state_dict_type: str, s3_bucket: str, s3_read_only_prefix: str,
                                  composer_version: str):
    if version.parse(torch.__version__) >= version.parse('2.0.0') and state_dict_type == 'local':
        pytest.xfail(
            'Loading a torch 1.13 checkpoint with torch 2.0 for state_dict_type local is not backwards compatible. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )

    rank = 0 if state_dict_type == 'full' else '{rank}'
    load_path = f's3://{s3_bucket}/{s3_read_only_prefix}/backwards_compatibility/{composer_version}/{sharding_strategy.lower()}_{state_dict_type}_{precision}/ba2_rank{rank}.pt'

    trainer2 = get_trainer(
        fsdp_state_dict_type=state_dict_type,
        num_features=32,  # This parameter setting is very important. Don't change or the test will fail.
        num_classes=8,  # This parameter setting is very important. Don't change or the test will fail.
        sharding_strategy=sharding_strategy,
        load_path=load_path,
        precision=precision,
        max_duration='4ba',
    )
    state_dict2 = trainer2.state.state_dict()

    if (dist.get_global_rank() == 0 and state_dict_type == 'full') or state_dict_type in ['sharded', 'local']:
        filled_load_path = load_path.format(rank=dist.get_global_rank())
        destination = str(tmp_path / pathlib.Path(filled_load_path).name)
        get_file(filled_load_path, destination=destination)
        with open(destination, 'rb') as f:
            state_dict1 = torch.load(f)['state']
        _compare_model_params_between_state_dicts(state_dict1, state_dict2)

        _compare_optims_between_state_dicts(state_dict1, state_dict2)

    # Continue to fit to make sure we can continue training.
    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load_with_ema(world_size, tmp_path: pathlib.Path, precision: str, optimizer: str):
    save_folder = tmp_path
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        algorithms=EMA(smoothing=0.9999, half_life=None, update_interval='1ba'),
        save_interval='1ba',
        max_duration='5ba',
        optimizer=optimizer,
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()

    load_path = str(save_folder / pathlib.Path('ba4-rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        load_path=load_path,
        sharding_strategy='SHARD_GRAD_OP',
        algorithms=EMA(smoothing=0.9999, half_life=None, update_interval='1ba'),
        save_interval='1ba',
        save_overwrite=True,
        optimizer=optimizer,
    )
    trainer2.fit(duration='1ba')
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('weights_only', [True, False])
@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('autoresume', [False])  # True commented out for now
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
def test_fsdp_partitioned_state_dict_load(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool,
                                          precision: str, optimizer: str, weights_only: bool):
    if state_dict_type == 'local' and using_torch_2():
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )
    if autoresume:
        run_name = 'my-autoresume-run'
    else:
        run_name = None
    save_folder = '/tmp/test_checkpoints'
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(save_folder=str(save_folder),
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           run_name=run_name,
                           precision=precision,
                           autoresume=autoresume,
                           optimizer=optimizer,
                           save_weights_only=weights_only,
                           fsdp_sharded_ckpt_prefix_dir='ba{batch}')
    trainer1.fit()
    rng1 = get_rng_state()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('ba2'))
    trainer2 = get_trainer(save_folder=str(save_folder),
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           load_path=load_path,
                           precision=precision,
                           autoresume=autoresume,
                           run_name=run_name,
                           max_duration='4ba',
                           optimizer=optimizer,
                           load_weights_only=weights_only)
    state_dict_from_trainer2 = trainer2.state.state_dict()
    rng2 = trainer2._rng_state
    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
    if not weights_only:
        _compare_rng_states_between_trainers(rng1, rng2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('autoresume', [True])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_mismatch_timestamp_error(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool):
    run_name = 'my-run-ar' if autoresume else 'my-run'
    save_folder = str(tmp_path / pathlib.Path(run_name))
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(save_folder=save_folder,
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           run_name=run_name,
                           autoresume=autoresume,
                           max_duration='2ba',
                           save_interval='1ba')
    trainer1.fit()
    trainer1.close()
    # Corrupt latest checkpoint symlink for rank1 by changing it from batch 2 checkpoint to the batch 1 one
    # and removing batch 2 checkpoint.
    if dist.get_global_rank() == 1:
        latest_symlink = str(pathlib.Path(save_folder) / pathlib.Path('latest-rank1.pt'))
        latest_checkpoint_path = pathlib.Path(save_folder) / pathlib.Path('ba2') / pathlib.Path(
            save_filename.format(batch=2, rank=1))
        assert os.readlink(latest_symlink) == str(
            pathlib.Path('ba2') / pathlib.Path(save_filename.format(batch=2, rank=1)))
        oldest_checkpoint_relative_path = str(pathlib.Path('ba1') / pathlib.Path(save_filename.format(batch=1, rank=1)))
        os.remove(latest_symlink)
        os.symlink(src=oldest_checkpoint_relative_path, dst=latest_symlink)
        os.remove(latest_checkpoint_path)
        assert os.readlink(latest_symlink) == oldest_checkpoint_relative_path

    expected_error = pytest.raises(RuntimeError, match='Timestamp mismatch error:*')

    with expected_error:
        get_trainer(
            save_folder=save_folder,
            save_filename=save_filename,
            fsdp_state_dict_type=state_dict_type,
            autoresume=autoresume,
            run_name=run_name,
        )


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('use_remote', [pytest.param(True, marks=pytest.mark.remote), False])
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_sharded_folder(world_size, use_remote, tmp_path: pathlib.Path, state_dict_type: str, s3_bucket):
    run_name = 'my-cool-s3-run'
    if use_remote:
        save_folder = 's3://' + str(pathlib.Path(s3_bucket) / pathlib.Path(run_name))
    else:
        save_folder = str(tmp_path / pathlib.Path(run_name))
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(save_folder=save_folder,
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           fsdp_sharded_ckpt_prefix_dir='ba{batch}',
                           run_name=run_name,
                           max_duration='2ba',
                           save_interval='1ba',
                           save_overwrite=True)
    trainer1.fit()
    trainer1.close()
    if not use_remote:
        expected_checkpoint_path = os.path.join(save_folder, 'ba1', f'ba1-rank{dist.get_global_rank()}.pt')
        assert os.path.exists(expected_checkpoint_path)

    load_path = os.path.join(save_folder, 'ba1', 'ba1-rank{rank}.pt')
    trainer2 = get_trainer(
        fsdp_state_dict_type=state_dict_type,
        load_path=load_path,
        max_duration='2ba',
    )
    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('weights_only', [True, False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.0.0'),
                    reason='requires PyTorch 2.0.0 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
def test_new_sharded_save(world_size, tmp_path: pathlib.Path, state_dict_type: str, weights_only: bool):
    if state_dict_type == 'local':
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )
    save_folder = '/tmp/test_checkpoints'
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type=state_dict_type,
        save_weights_only=weights_only,
        save_interval='2ba',
        fsdp_sharded_ckpt_prefix_dir='ba{batch}',
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()

    trainer2 = get_trainer(fsdp_state_dict_type=state_dict_type,)
    load_path_dir = str(save_folder / pathlib.Path('ba2'))
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    storage_reader = dist_cp.FileSystemReader(load_path_dir)
    model_state_dict = {'state': {'model': trainer2.state.state_dict()['model']}}
    dist_cp.load_state_dict(model_state_dict, storage_reader)
    trainer2.state.load_model_state(model_state_dict['state'], strict=True, logger=None)
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, trainer2.state.state_dict())
    if not weights_only:
        optim_state = load_sharded_optimizer_state_dict(model_state_dict=trainer2.state.state_dict()['model'],
                                                        optimizer_key='optimizers',
                                                        storage_reader=storage_reader)
        trainer2.state.load_optim_state(optim_state)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, trainer2.state.state_dict())


@pytest.mark.gpu
@world_size(2)
@pytest.mark.remote
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('weights_only', [True, False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.0.0'),
                    reason='requires PyTorch 2.0.0 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
def test_new_remote_sharded_save(world_size, tmp_path: pathlib.Path, state_dict_type: str, weights_only: bool,
                                 s3_bucket, s3_ephemeral_prefix):
    if state_dict_type == 'local':
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )
    save_folder = f's3://{s3_bucket}/{s3_ephemeral_prefix}/checkpoints/{{run_name}}'
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type=state_dict_type,
        save_weights_only=weights_only,
        save_interval='2ba',
        fsdp_sharded_ckpt_prefix_dir='ba{batch}',
    )
    run_name = trainer1.state.run_name
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()

    from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
    from composer.utils.object_store.s3_object_store import S3ObjectStore
    s3os = S3ObjectStore(bucket=s3_bucket)
    remote_folder = save_folder.format(run_name=run_name).replace(f's3://{s3_bucket}/', '')
    assert remote_folder == f'ephemeral/checkpoints/{run_name}'
    local_folder = f'/tmp/checkpoints/{run_name}'
    os.makedirs(local_folder, exist_ok=True)
    s3os.download_object(object_name=os.path.join(remote_folder, 'ba2', _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME),
                         filename=os.path.join(local_folder, _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME))

    if dist.get_global_rank() == 0:
        s3os.download_object(object_name=os.path.join(remote_folder, 'ba2', '.metadata'),
                             filename=os.path.join(local_folder, '.metadata'))
    dist.barrier()

    trainer2 = get_trainer(fsdp_state_dict_type=state_dict_type,)
    load_path_dir = local_folder
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
    storage_reader = dist_cp.FileSystemReader(load_path_dir)
    model_state_dict = {'state': {'model': trainer2.state.state_dict()['model']}}
    dist_cp.load_state_dict(model_state_dict, storage_reader)
    trainer2.state.load_model_state(model_state_dict['state'], strict=True, logger=None)
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, trainer2.state.state_dict())
    if not weights_only:
        optim_state = load_sharded_optimizer_state_dict(model_state_dict=trainer2.state.state_dict()['model'],
                                                        optimizer_key='optimizers',
                                                        storage_reader=storage_reader)
        trainer2.state.load_optim_state(optim_state)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, trainer2.state.state_dict())
