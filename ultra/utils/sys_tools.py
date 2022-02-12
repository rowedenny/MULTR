import os
import sys
import traceback
import inspect
import random
import datetime

import numpy as np
import torch


def find_class(class_str):
    """Find the corresponding class based on a string of class name.

      Args:
        class_str: a string containing the name of the class
      Raises:
        ValueError: If there is no class with the name.
    """
    mod_str, _sep, class_str = class_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


def create_object(class_str, *args, **kwargs):
    """Find the corresponding class based on a string of class name and create an object.

      Args:
        class_str: a string containing the name of the class
      Raises:
        ValueError: If there is no class with the name.
    """
    return find_class(class_str)(*args, **kwargs)


def list_recursive_concrete_subclasses(base):
    """List all concrete subclasses of `base` recursively.

      Args:
        base: a string containing the name of the class
    """

    return _filter_concrete(_bfs(base))


def _filter_concrete(classes):
    return list(filter(lambda c: not inspect.isabstract(c), classes))


def _bfs(base):
    return base.__subclasses__() + sum([
        _bfs(subclass)
        for subclass in base.__subclasses__()
    ], [])


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
