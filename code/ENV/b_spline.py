import numpy as np
from scipy.interpolate import BSpline


class KBSpline(object):
    def __init__(self, k):
        self.k = k

    def traj_smoothing(self, path):
        t = []
        num = len(path)
        for i in range(num + self.k + 1):
            if i <= self.k:
                t.append(0)
            elif i >= num:
                t.append(num - self.k)
            else:
                t.append(i - self.k)
        spl_x = BSpline(t, path[:, 0], self.k)
        spl_y = BSpline(t, path[:, 1], self.k)
        xx = np.linspace(0.0, num - self.k, 10 * num)
        return spl_x(xx), spl_y(xx)
