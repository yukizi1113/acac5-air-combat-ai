"""Rollout file I/O without torch (actors run in the simulator's Python 3.10 env, which has no torch)."""
import os
import gzip
import pickle


def save_rollout(res, path):
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wb', compresslevel=3) as f:
        pickle.dump(res, f, protocol=4)
    os.replace(tmp, path)


def load_rollout(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def dumps(res):
    return gzip.compress(pickle.dumps(res, protocol=4), compresslevel=3)


def loads(b):
    return pickle.loads(gzip.decompress(b))
