# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""S3-Compatible object store."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from typing import Any, Callable, Dict, Optional, Union, List
import warnings
from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore

__all__ = ['WandBObjectStore']



class WandBObjectStore(ObjectStore):
    """
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import wandb
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='wandb',
                                                conda_package='wandb',
                                                conda_channel='conda-forge') from e

        del wandb  # unused

        if init_kwargs is None:
            init_kwargs = {}

        if project is not None:
            init_kwargs['project'] = project

        if group is not None:
            init_kwargs['group'] = group

        if name is not None:
            init_kwargs['name'] = name

        if entity is not None:
            init_kwargs['entity'] = entity

        if tags is not None:
            init_kwargs['tags'] = tags

        self._init_kwargs = init_kwargs
        self._is_in_atexit = False

        # Set these variable directly to allow fetching an Artifact **without** initializing a WandB run
        # When used as a LoggerDestination, these values are overriden from global rank 0 to all ranks on Event.INIT
        self.entity = entity
        self.project = project


    def get_object_size(self, object_name: str) -> int:
        pass

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        del callback
        import wandb
        # replace all unsupported characters with periods
        # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
        new_object_name = re.sub(r'[^a-zA-Z0-9-_\.]', '.', object_name)
        if new_object_name != object_name:
            warnings.warn(('WandB permits only alpha-numeric, periods, hyphens, and underscores in file names. '
                            f"The file with name '{object_name}' will be stored as '{new_object_name}'."))

        extension = new_object_name.split('.')[-1]

        wandb_artifact = wandb.Artifact(
            name=new_object_name,
            type=extension,
        )
        wandb_artifact.add_file(os.path.abspath(filename))
        wandb.log_artifact(wandb_artifact)
        

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ):
        import wandb
        import wandb.errors

        # using the wandb.Api() to support retrieving artifacts on ranks where
        # artifacts are not initialized
        api = wandb.Api()
        if not self.entity or not self.project:
            raise RuntimeError('get_file_artifact can only be called after running init()')

        # replace all unsupported characters with periods
        # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
        if ':' not in object_name:
            object_name += ':latest'

        new_object_name = re.sub(r'[^a-zA-Z0-9-_\.:]', '.', object_name)
        if new_object_name != object_name:
            warnings.warn(('WandB permits only alpha-numeric, periods, hyphens, and underscores in file names. '
                           f"The file with name '{object_name}' will be stored as '{new_object_name}'."))

        try:
            wandb_artifact = api.artifact('/'.join([self.entity, self.project, new_object_name]))
        except wandb.errors.CommError as e:
            if 'does not contain artifact' in str(e):
                raise FileNotFoundError(f'WandB Artifact {new_object_name} not found') from e
            raise e
        with tempfile.TemporaryDirectory() as tmpdir:
            wandb_artifact_folder = os.path.join(tmpdir, 'wandb_artifact_folder')
            wandb_artifact.download(root=wandb_artifact_folder)
            wandb_artifact_names = os.listdir(wandb_artifact_folder)
            # We only log one file per artifact
            if len(wandb_artifact_names) > 1:
                raise RuntimeError(
                    'Found more than one file in WandB artifact. We assume the checkpoint is the only file in the WandB artifact.'
                )
            wandb_artifact_name = wandb_artifact_names[0]
            wandb_artifact_path = os.path.join(wandb_artifact_folder, wandb_artifact_name)
            if overwrite:
                os.replace(wandb_artifact_path, filename)
            else:
                os.rename(wandb_artifact_path, filename)
    
    def initialize_wandb_module(self, run_name: Optional[str] = None):
        import wandb

        # Use the state run name if the name is not set.
        if 'name' not in self._init_kwargs or self._init_kwargs['name'] is None:
            self._init_kwargs['name'] = run_name

        if wandb.run is None:
            wandb.init(**self._init_kwargs)
        else:
            if run_name is not None and wandb.run.name != run_name:
                warnings.warn(f'WandB run name is already set, so using {wandb.run.name} instead of {run_name}')
        


