import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set the seed to ensure reproducibility of experiments.
    https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed (int): The seed value to set.
    """

    # Set the seed for different libraries
    random.seed(seed)   # Set the seed for Python's built-in random module
    os.environ['PYTHONHASHSEED'] = str(seed)    # Set Python's hash random seed
    np.random.seed(seed)    # Set the seed for the numpy module
    torch.manual_seed(seed)   # Set the seed for PyTorch's CPU operations
    torch.cuda.manual_seed(seed)    # Set the seed for PyTorch's CUDA operations on a single GPU
    torch.cuda.manual_seed_all(seed)    # Set the seed for all GPUs

    # Disable CuDNN's auto-tuner to prevent uncertainty in algorithm selection.
    # The auto-tuner automatically searches for the best convolution implementation algorithm,
    # but it does not guarantee reproducibility.
    torch.backends.cudnn.benchmark = False
    # Enable deterministic algorithms to prevent uncertainty in the algorithm itself.
    torch.backends.cudnn.deterministic = True
    # Enable CuDNN's deep convolutional neural network to improve training speed,
    # but it may affect reproducibility.
    torch.backends.cudnn.enabled = True

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    # If the CUDA version is 10.2 or higher, a few CUDA operations are non-deterministic.
    # Set the environment variable to address this.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # # Use deterministic algorithms
    # torch.use_deterministic_algorithms(True)
