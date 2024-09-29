"""
The file is adopted from
https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/src/utils.py
"""
import os


def expand_path(path, unique=True):
    """
    Resolve a path that may contain variables and user home directory references.
    """
    return os.path.expandvars(os.path.expanduser(path))
