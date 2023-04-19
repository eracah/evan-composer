# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import textwrap

import numpy as np
import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.markers import world_size


def get_trainer(save_folder=None,
                save_filename='ba{batch}-rank{rank}.pt',
                num_features=2,
                num_classes=2,
                fsdp_state_dict_type='full',
                load_path=None,
                autoresume=False,
                run_name=None,
                max_duration='2ba',
                save_interval='2ba',
                precision='amp_fp16'):
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
        save_interval=save_interval,
        save_filename=save_filename,
        save_overwrite=False,
        precision=precision,
        load_path=load_path,
        progress_bar=False,
        log_to_console=False,
        autoresume=autoresume,
        run_name=run_name,
        save_latest_filename='latest-rank{rank}.pt',
    )
    return trainer


def _compare_optims_between_state_dicts(state_dict1, state_dict2):
    # Check that optim params are equal between checkpoint and in memory optimizer
    state_dict1_optim_params = state_dict1['optimizers']['state']
    state_dict2_optim_params = state_dict2['optimizers']['state']
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
            state_dict1_moment = state_dict1_param_moment_dict[moment_name]
            state_dict2_moment = state_dict2_param_moment_dict[moment_name]
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


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('autoresume', [False])  # True
#@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load(world_size, tmp_path: pathlib.Path, autoresume: bool):  #, precision: str):
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
        # precision=precision,
        autoresume=autoresume)
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(
        #save_folder=str(save_folder),
        #save_filename=save_filename,
        fsdp_state_dict_type='full',
        load_path=load_path,
        run_name=run_name,
        #precision=precision,
        autoresume=autoresume,
        max_duration='4ba')
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    # Continue to fit to make sure we can continue training.
    trainer2.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('autoresume', [False])  #, True])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_partitioned_state_dict_load(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool,
                                          precision: str):
    if autoresume:
        run_name = 'my-autoresume-run'
    else:
        run_name = None

    rank0_tmp_path = dist.all_gather_object(tmp_path)[0]
    save_folder = str(rank0_tmp_path / pathlib.Path('{run_name}'))
    trainer1 = get_trainer(save_folder=save_folder,
                           fsdp_state_dict_type=state_dict_type,
                           run_name=run_name,
                           precision=precision,
                           autoresume=autoresume)
    run_name = trainer1.state.run_name
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(pathlib.Path(save_folder.format(run_name=run_name)) / pathlib.Path('ba2'))

    trainer2 = get_trainer(save_folder=save_folder,
                           fsdp_state_dict_type=state_dict_type,
                           load_path=load_path,
                           precision=precision,
                           autoresume=autoresume,
                           run_name=run_name,
                           max_duration='4ba')
    state_dict_from_trainer2 = trainer2.state.state_dict()

    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    trainer2.fit()


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
        latest_checkpoint_path = pathlib.Path(save_folder) / pathlib.Path(save_filename.format(batch=2, rank=1))
        assert os.readlink(latest_symlink) == latest_checkpoint_path.name
        oldest_checkpoint_path = pathlib.Path(save_folder) / pathlib.Path(save_filename.format(batch=1, rank=1))
        os.remove(latest_symlink)
        os.symlink(src=oldest_checkpoint_path.name, dst=latest_symlink)
        os.remove(latest_checkpoint_path)
        assert os.readlink(latest_symlink) == oldest_checkpoint_path.name

    expected_error = pytest.raises(RuntimeError, match='Timestamp mismatch error:*')

    with expected_error:
        get_trainer(
            save_folder=save_folder,
            save_filename=save_filename,
            fsdp_state_dict_type=state_dict_type,
            autoresume=autoresume,
            run_name=run_name,
        )
