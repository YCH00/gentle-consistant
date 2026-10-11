# Shared artifact schema ported from the user's FOCAL/focal_shared (2026-10-11).
import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = 'shared-main-v1'
REFERENCE_COMMIT = '075df392bbaad8177339c41b700d567ef550e94f'
ENVIRONMENTS = ('point-robot', 'ant-dir', 'cheetah-vel', 'cheetah-dir',
                'hopper-rand-params', 'walker-rand-params')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      default=jsonable, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

