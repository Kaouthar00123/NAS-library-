__all__ = ['verify_duplication', 'retry_sampling']

import logging
import random
from typing import Any, Callable

import numpy as np
import torch

_logger = logging.getLogger(__name__)

line_fmt = '{:<40}  {:<8}'

def verify_duplication(history: set, sample: Any, raise_on_dup: bool = False) -> bool:
    """
    Check if the new sample has been seen before.
    #TODO this current method only check instance duplication  
    Parameters
    ----------
    history
        A set containing the history of samples.
    sample
        The new sample to check.
    raise_on_dup
        Whether to raise an exception if a duplicate is found.

    Returns
    -------
    bool
        True if the sample is not a duplicate, False otherwise.
    """
    if sample in history:
        _logger.debug('Duplicated sample found: %s', sample)
        if raise_on_dup:
            raise ValueError(f'Duplicated sample found: {sample}')
        return False
    history.add(sample)
    return True


def retry_sampling(func: Callable, retries: int = 5) -> Any:
    """
    Retry a function until it succeeds.

    Parameters
    ----------
    func
        The function to retry.
    retries
        Number of retries.

    Returns
    -------
    Any
        The result of the function, if successful. None otherwise.
    """
    for retry in range(retries):
        try:
            return func()
        except Exception as e:
            if retry in [0, 10, 100, 1000]:
                _logger.debug('Sampling failed. %d retries so far. Exception caught: %r', retry, e)

    _logger.warning('Sampling failed after %d retries. Giving up and returning None.', retries)
    return None

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)