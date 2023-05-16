# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import threading
import contextlib
import os
import pathlib
from typing import Any, Dict, Type

import mockssh
import moto
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import composer.utils.object_store
import composer.utils.object_store.sftp_object_store
from composer.utils.object_store import LibcloudObjectStore, ObjectStore, OCIObjectStore, S3ObjectStore, SFTPObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore
from tests.common import get_module_subclasses
import pytest

from composer.utils.object_store import S3ObjectStore

@contextlib.contextmanager
def mock_s3_object_store(remote: bool, s3_bucket:str, test_session_name: str, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip('boto3')
    if remote:
        yield S3ObjectStore(bucket=s3_bucket, prefix=test_session_name)
    else:
        bucket = 'my-bucket'
        prefix = 'folder/subfolder'
        import boto3
        monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
        monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
        monkeypatch.setenv('AWS_SECURITY_TOKEN', 'testing')
        monkeypatch.setenv('AWS_SESSION_TOKEN', 'testing')
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        with moto.mock_s3():
            # create the dummy bucket
            s3 = boto3.client('s3')
            s3.create_bucket(Bucket=bucket)
            yield S3ObjectStore(bucket=bucket, prefix=prefix)


@pytest.fixture
def dummy_obj(tmp_path: pathlib.Path):
    tmpfile_path = tmp_path / 'file_to_upload'
    with open(tmpfile_path, 'w+') as f:
        f.write('dummy content')
    return tmpfile_path


class MockCallback:

    def __init__(self, total_num_bytes: int) -> None:
        self.total_num_bytes = total_num_bytes
        self.transferred_bytes = 0
        self.num_calls = 0

    def __call__(self, transferred: int, total: int):
        self.num_calls += 1
        assert transferred == 0 or transferred >= self.transferred_bytes, 'transferred should be monotonically increasing'
        self.transferred_bytes = transferred
        assert total == self.total_num_bytes

    def assert_all_data_transferred(self):
        assert self.total_num_bytes == self.transferred_bytes

@pytest.mark.parametrize('remote', [False, pytest.param(True, marks=pytest.mark.remote)])
def test_s3_upload(dummy_obj: pathlib.Path, s3_bucket: str, test_session_name: str, remote: bool, monkeypatch: pytest.MonkeyPatch):
    object_name = 'tmpfile_object_name'
    cb = MockCallback(dummy_obj.stat().st_size)
    with mock_s3_object_store(remote, s3_bucket, test_session_name, monkeypatch) as s3_object_store:
        s3_object_store.upload_object(object_name, str(dummy_obj), callback=cb)

    cb.assert_all_data_transferred()

def test_get_uri(monkeypatch: pytest.MonkeyPatch):
    with mock_s3_object_store(monkeypatch) as s3_object_store:
        uri = s3_object_store.get_uri('tmpfile_object_name')
        assert uri == 's3://my-bucket/folder/subfolder/tmpfile_object_name'

def test_get_file_size(self, dummy_obj: pathlib.Path, remote: bool, s3_bucket: str, test_session_name: str, monkeypatch: pytest.MonkeyPatch):
    object_name = 'tmpfile_object_name'
    cb = MockCallback(dummy_obj.stat().st_size)
    with mock_s3_object_store(remote, s3_bucket, test_session_name, monkeypatch) as s3_object_store:
        s3_object_store.upload_object(object_name, str(dummy_obj), callback=cb)
    assert s3_object_store.get_object_size(object_name) == dummy_obj.stat().st_size

# def test_get_file_size_not_found(self, object_store: ObjectStore, remote: bool):
#     del remote  # unused
#     with pytest.raises(FileNotFoundError):
#         object_store.get_object_size('not found object')

# @pytest.mark.parametrize('overwrite', [True, False])
# def test_download(
#     self,
#     object_store: ObjectStore,
#     dummy_obj: pathlib.Path,
#     tmp_path: pathlib.Path,
#     overwrite: bool,
#     remote: bool,
# ):
#     del remote  # unused
#     object_name = 'tmpfile_object_name'
#     object_store.upload_object(object_name, str(dummy_obj))
#     filepath = str(tmp_path / 'destination_path')
#     cb = MockCallback(dummy_obj.stat().st_size)
#     object_store.download_object(object_name, filepath, callback=cb)
#     ctx = contextlib.nullcontext() if overwrite else pytest.raises(FileExistsError)
#     with ctx:
#         object_store.download_object(object_name, filepath, callback=cb, overwrite=overwrite)
#     cb.assert_all_data_transferred()

# def test_download_not_found(self, object_store: ObjectStore, remote: bool):
#     with pytest.raises(FileNotFoundError):
#         object_store.download_object('not_found_object', filename='not used')


# @pytest.mark.parametrize('bucket_uri_and_kwargs', object_stores, indirect=True)
# @pytest.mark.parametrize('remote', [False, pytest.param(True, marks=pytest.mark.remote)])
# class TestS3ObjectStore:




#     def test_get_file_size(self, object_store: ObjectStore, dummy_obj: pathlib.Path, remote: bool):
#         del remote  # unused
#         object_name = 'tmpfile_object_name'
#         object_store.upload_object(object_name, str(dummy_obj))
#         assert object_store.get_object_size(object_name) == dummy_obj.stat().st_size

#     def test_get_file_size_not_found(self, object_store: ObjectStore, remote: bool):
#         del remote  # unused
#         with pytest.raises(FileNotFoundError):
#             object_store.get_object_size('not found object')

#     @pytest.mark.parametrize('overwrite', [True, False])
#     def test_download(
#         self,
#         object_store: ObjectStore,
#         dummy_obj: pathlib.Path,
#         tmp_path: pathlib.Path,
#         overwrite: bool,
#         remote: bool,
#     ):
#         del remote  # unused
#         object_name = 'tmpfile_object_name'
#         object_store.upload_object(object_name, str(dummy_obj))
#         filepath = str(tmp_path / 'destination_path')
#         cb = MockCallback(dummy_obj.stat().st_size)
#         object_store.download_object(object_name, filepath, callback=cb)
#         ctx = contextlib.nullcontext() if overwrite else pytest.raises(FileExistsError)
#         with ctx:
#             object_store.download_object(object_name, filepath, callback=cb, overwrite=overwrite)
#         cb.assert_all_data_transferred()

#     def test_download_not_found(self, object_store: ObjectStore, remote: bool):
#         with pytest.raises(FileNotFoundError):
#             object_store.download_object('not_found_object', filename='not used')


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
