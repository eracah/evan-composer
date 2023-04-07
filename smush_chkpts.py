import torch.distributed as dist
import os
# print(os.environ)
from composer.utils.object_store import S3ObjectStore
import sys
from urllib.parse import urlparse
import tempfile
from pathlib import Path
import torch

s3_path = sys.argv[1]
output_path = sys.argv[2]

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
node_rank = int(os.environ['NODE_RANK'])

dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

print(dist.is_initialized())
parsed_path = urlparse(s3_path)
bucket, path = parsed_path.netloc, parsed_path.path.lstrip('/')
s3os = S3ObjectStore(bucket=bucket)
object_name=path.format(rank=rank)
with tempfile.TemporaryDirectory() as tempdir:
    local_chkpt_path = os.path.join(tempdir, Path(object_name).name)
    s3os.download_object(object_name=object_name, filename=local_chkpt_path)
    sd = torch.load(local_chkpt_path, map_location=torch.device('cpu'))
    print(sd['state'].keys())



