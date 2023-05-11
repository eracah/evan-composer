# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.utils.object_store import LibcloudObjectStore


class GCSObjectStore(LibcloudObjectStore):

    def __init__(self, bucket: str):
        super().__init__(
            provider='google_storage',
            container=bucket,
            key_environ='GCS_KEY',  # Name of env variable for HMAC access id.
            secret_environ='GCS_SECRET',  # Name of env variable for HMAC secret.
        )
