from composer.utils import S3ObjectStore

s3os = S3ObjectStore(bucket='mosaicml-internal-checkpoints-test')

objs = s3os.list_objects(
    'evan-test/checkpoints/',
    show_full_paths=False, recurse=False)
print(objs)
