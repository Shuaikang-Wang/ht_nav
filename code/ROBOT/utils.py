import os
import sys

sys.path.append(os.getcwd())

import numpy as np


# some general functions

def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point1 - point2)
