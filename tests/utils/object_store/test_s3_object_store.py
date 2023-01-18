# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from pathlib import Path
import threading
import moto
import pytest
from unittest.mock import MagicMock
from composer.utils.object_store import S3ObjectStore


def _worker(bucket: str, tmp_path: pathlib.Path, tid: int):
    object_store = S3ObjectStore(bucket=bucket)
    os.makedirs(tmp_path / str(tid))
    with pytest.raises(FileNotFoundError):
        object_store.download_object('this_key_should_not_exist', filename=tmp_path / str(tid) / 'dummy_file')


# This test requires properly configured aws credentials; otherwise the s3 client would hit a NoCredentialsError
# when constructing the Session, which occurs before the bug this test checks
@pytest.mark.remote
def test_s3_object_store_multi_threads(tmp_path: pathlib.Path, s3_bucket: str):
    """Test to verify that we do not hit https://github.com/boto/boto3/issues/1592."""
    pytest.importorskip('boto3')
    threads = []
    # Manually tried fewer threads; it seems that 100 is needed to reliably re-produce the bug
    for i in range(100):
        t = threading.Thread(target=_worker, kwargs={'bucket': s3_bucket, 'tid': i, 'tmp_path': tmp_path})
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

@pytest.fixture
def mock_bucket_name():
    return 'my_bucket'
    # bucket_uri = 's3://my-bucket'
    # kwargs = {'bucket': 'my-bucket', 'prefix': 'folder/subfolder'}

@pytest.fixture
def mock_prefix_name():
    return 'folder/subfolder'

@pytest.fixture
def mock_s3_obj_store(mock_bucket_name, mock_prefix_name, monkeypatch):
    pytest.importorskip('boto3')
    import boto3
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
    monkeypatch.setenv('AWS_SECURITY_TOKEN', 'testing')
    monkeypatch.setenv('AWS_SESSION_TOKEN', 'testing')
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
    with moto.mock_s3():
        # create the dummy bucket
        s3 = boto3.client('s3')
        s3.create_bucket(Bucket=mock_bucket_name)
    return S3ObjectStore(bucket=mock_bucket_name, prefix=mock_prefix_name )
    
@pytest.fixture
def dummy_file(tmp_path: pathlib.Path):
    file_to_upload = str(tmp_path / Path('my_upload.bin'))
    with open(file_to_upload, 'wb') as f:
        f.write(bytes(range(20)))
    return file_to_upload


def test_upload_object(mock_s3_obj_store, monkeypatch, dummy_file, mock_bucket_name,
                       mock_prefix_name, tmp_path):
    pytest.importorskip('boto3')
    s3_os = mock_s3_obj_store
    mock_object_name = 'my_object'

    mock_upload_file = MagicMock()
    monkeypatch.setattr(s3_os.client, 'upload_file', mock_upload_file)


    s3_os.upload_object(object_name=mock_object_name, filename=dummy_file)
    mock_upload_file.assert_called_once_with(Bucket=mock_bucket_name,
                                             Key=mock_prefix_name + mock_object_name,
                                             Filename=dummy_file,
                                             Callback=None,
                                             Config={})

def test_download(
    mock_s3_obj_store,
    dummy_file,
    tmp_path,
    overwrite: bool,
):
    del remote  # unused
    object_name = 'tmpfile_object_name'
    mock_s3_obj_store.upload_object(object_name, str(dummy_file))
    filepath = str(tmp_path / 'destination_path')
    mock_s3_obj_store.download_object(object_name, filepath)

def test_download_not_found(self, mock_s3_obj_store: ObjectStore, remote: bool):
    with pytest.raises(FileNotFoundError):
        mock_s3_obj_store.download_object('not_found_object', filename='not used')

def test_get_uri(mock_s3_obj_store, mock_prefix_name, mock_bucket_name):
    mock_object_name = 'tmpfile_object_name'
    uri = mock_s3_obj_store.get_uri(mock_object_name)
    if isinstance(mock_s3_obj_store, S3ObjectStore):
        assert uri == f's3://{mock_bucket_name}/{mock_prefix_name}/{mock_object_name}'

def test_get_file_size(self, object_store: ObjectStore, dummy_obj: pathlib.Path, remote: bool):
    object_name = 'tmpfile_object_name'
    object_store.upload_object(object_name, str(dummy_obj))
    assert object_store.get_object_size(object_name) == dummy_obj.stat().st_size

def test_get_file_size_not_found(self, object_store: ObjectStore, remote: bool):
    del remote  # unused
    with pytest.raises(FileNotFoundError):
        object_store.get_object_size('not found object')

@pytest.mark.parametrize('overwrite', [True, False])
def test_download(
    self,
    object_store: ObjectStore,
    dummy_obj: pathlib.Path,
    tmp_path: pathlib.Path,
    overwrite: bool,
    remote: bool,
):
    del remote  # unused
    object_name = 'tmpfile_object_name'
    object_store.upload_object(object_name, str(dummy_obj))
    filepath = str(tmp_path / 'destination_path')
    cb = MockCallback(dummy_obj.stat().st_size)
    object_store.download_object(object_name, filepath, callback=cb)
    ctx = contextlib.nullcontext() if overwrite else pytest.raises(FileExistsError)
    with ctx:
        object_store.download_object(object_name, filepath, callback=cb, overwrite=overwrite)
    cb.assert_all_data_transferred()

def test_download_not_found(self, object_store: ObjectStore, remote: bool):
    with pytest.raises(FileNotFoundError):
        object_store.download_object('not_found_object', filename='not used')
