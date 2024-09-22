import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import math

# some general functions


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)