import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import yaml
from typing import Tuple

from NF.utils import distance


class World(object):
    def __init__(self, config='config.yaml'):
        self.load_world_config(config)
        self.construct_world()
        self.global_lidar_points = []
        self.cluster_points = []
        self.all_cluster_segments = []
        self.squircle_data = []

    @property
    def config(self) -> dict:
        return self._config

    @property
    def obstacles(self):
        return self._obstacles

    @property
    def workspace(self):
        return self._workspace

    def load_world_config(self, config):
        with open(config, "r") as stream:
            try:
                self._config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def construct_world(self):
        self._obstacles = []
        # add obstacles by param
        obs_param = self.config['obstacles']
        for obs in obs_param:
            stars_in_obs = []
            obs_shape = obs['shape']
            if obs_shape == 'StarTree':
                for star in obs['stars']:
                    star_cls = star['type']
                    klass = globals()[star_cls]
                    star_obj = klass(**star)
                    stars_in_obs.append(star_obj)
            else:
                star = obs['star'][0]
                star_cls = star['type']
                klass = globals()[star_cls]
                star_obj = klass(**star)
                stars_in_obs.append(star_obj)
            self._obstacles.append(stars_in_obs)
            # self._obstacles = stars_in_obs

        # add workspace
        self._workspace = []
        ws_parm = self.config['workspace']
        for ws in ws_parm:
            ws_shape = ws['shape']
            star_cls = 'Workspace'
            stars_in_ws = []
            if ws_shape == 'StarTree':
                star_ws = ws['stars'][0]
                klass = globals()[star_cls]
                star_obj = klass(**star_ws)
                stars_in_ws.append(star_obj)
                for star in ws['stars'][1:]:
                    star_cls_obs = star['type']
                    klass = globals()[star_cls_obs]
                    star_obj = klass(**star)
                    stars_in_ws.append(star_obj)
            else:
                star = ws['star'][0]
                klass = globals()[star_cls]
                star_obj = klass(**star)
                stars_in_ws.append(star_obj)
            self._workspace.append(stars_in_ws)
        # ws = self.config['workspace'][0]
        # ws_shape = ws['shape']
        # star_cls = 'Workspace'
        # stars_in_ws = []
        # if ws_shape == 'StarTree':
        #     star_ws = ws['stars'][0]
        #     klass = globals()[star_cls]
        #     star_obj = klass(**star_ws)
        #     stars_in_ws.append(star_obj)
        #     for star in ws['stars'][1:]:
        #         star_cls_obs = star['type']
        #         klass = globals()[star_cls_obs]
        #         star_obj = klass(**star)
        #         stars_in_ws.append(star_obj)
        # else:
        #     star = ws['star'][0]
        #     klass = globals()[star_cls]
        #     star_obj = klass(**star)
        #     stars_in_ws.append(star_obj)
        # self._workspace = stars_in_ws

    def check_point_in_free_space(self, q: np.ndarray, threshold=0.0):
        if not self.workspace[0][0].check_point_inside(q, threshold=0.9 * threshold):
            return False
        for ws in self.workspace:
            for ws_i in ws[1:]:
                if ws_i.check_point_inside(q, threshold):
                    return False
        for obs in self.obstacles:
            for obs_i in obs:
                if obs_i.check_point_inside(q, threshold):
                    return False
        return True

    def check_point_in_obs_free_space(self, q: np.ndarray, threshold=0.0):
        for ws in self.workspace:
            for ws_i in ws[1:]:
                if ws_i.check_point_inside(q, threshold):
                    return False
        for obs in self.obstacles:
            for obs_i in obs:
                if obs_i.check_point_inside(q, threshold):
                    return False
        return True

    def check_obs_exist(self, new_obs):
        for obs in self.obstacles:
            for obs_i in obs:
                if np.allclose(obs_i.center, new_obs.center, rtol=0.1):
                    return True
        return False

    def add_obstacles(self, obstacles):
        for obs in obstacles:
            if not self.check_obs_exist(obs):
                self._obstacles.append([obs])
                return True
        return False


class Rectangular(object):
    def __init__(self, type, center, width, height, theta, s):
        self.type = type
        self._center = center
        self._width = width + 0.1
        self._height = height + 0.1
        self._theta = theta
        self._s = s
        self._radius = 0.1 * min(width, height)  # 0.9 0.3
        self.local_points = []
        self.accumulated_local_points = []

    @property
    def center(self) -> np.ndarray:
        return np.array(self._center)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def s(self) -> float:
        return self._s

    def potential(self, q: np.ndarray) -> float:
        theta = self.theta
        s = self.s
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width / 2, self.height / 2
        rotated_x = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta) + x_0
        rotated_y = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta) + y_0
        x, y = rotated_x, rotated_y
        return ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                    (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (1 - s ** 2) * (
                                                           a ** 2 + b ** 2)) ** 0.5) - (a ** 2 + b ** 2)

    def compute_v(self, q: np.ndarray, beta: float) -> float:
        maxVal = (self.width / 2) ** 2 + (self.height / 2) ** 2
        if distance(q, self.center) < 1.0e-3:
            return self.radius * (1.0 + beta / maxVal) * 1e3
        else:
            return self.radius * (1.0 + beta / maxVal) / distance(q, self.center)

    def compute_T(self, q: np.ndarray, beta: float) -> np.ndarray:
        return self.compute_v(q, beta) * (q - self.center) + self.center

    def x_limits(self, threshold=0.2):
        x_min = self.center[0] - self.width / 2 - threshold
        x_max = self.center[0] + self.width / 2 + threshold
        return x_min, x_max

    def y_limits(self, threshold=0.2):
        y_min = self.center[1] - self.height / 2 - threshold
        y_max = self.center[1] + self.height / 2 + threshold
        return y_min, y_max

    def workspace_meshgrid(self, resolution=0.01, threshold=0.0) -> Tuple[np.ndarray, np.ndarray]:
        # 0.05
        x_min, x_max = self.x_limits()
        y_min, y_max = self.y_limits()
        x = np.arange(x_min - threshold, x_max + threshold, resolution)
        y = np.arange(y_min - threshold, y_max + threshold, resolution)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def potential_accurate(self, q, s=0.99):
        rotation = 0.5
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width, self.height
        rotated_x = (x - x_0) * np.cos(rotation) + (y - y_0) * np.sin(rotation) + x_0
        rotated_y = -(x - x_0) * np.sin(rotation) + (y - y_0) * np.cos(rotation) + y_0
        x, y = rotated_x, rotated_y
        return (1 / (b / 2) ** 2) * (((b / a) * (x - x_0)) ** 2 + (y - y_0) ** 2 +
                                     (((b / a) * (x - x_0)) ** 4 + (y - y_0) ** 4 +
                                      ((2 - 4 * s ** 2) * ((b / a) * (x - x_0)) ** 2 * (y - y_0) ** 2)) ** 0.5) / 2 - 1

    def check_point_inside(self, q: np.ndarray, threshold=0.0):
        theta = self.theta
        s = self.s
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width + threshold, self.height + threshold
        rotated_x = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta) + x_0
        rotated_y = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta) + y_0
        x, y = rotated_x, rotated_y
        potential_point = (1 / (b / 2) ** 2) * (((b / a) * (x - x_0)) ** 2 + (y - y_0) ** 2 +
                                                (((b / a) * (x - x_0)) ** 4 + (y - y_0) ** 4 +
                                                 ((2 - 4 * s ** 2) * ((b / a) * (x - x_0)) ** 2 * (
                                                         y - y_0) ** 2)) ** 0.5) / 2 - 1
        if potential_point <= 0.0:
            return True
        # if self.potential(q) <= threshold:
        #     return True
        return False


class Workspace(Rectangular):
    def __init__(self, type, center, width, height, theta, s):
        super().__init__(type, center, width - 0.0, height - 0.0, theta, s)
        self._radius = 10.0 * max(width, height)  # 8.0 2.0

    def potential(self, q):
        theta = self.theta
        s = self.s
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width / 2, self.height / 2
        rotated_x = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta) + x_0
        rotated_y = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta) + y_0
        x, y = rotated_x, rotated_y
        return (a ** 2 + b ** 2) - ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                                        (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (
                                                                               1 - s ** 2) * (
                                                                               a ** 2 + b ** 2)) ** 0.5)

    def compute_v(self, q: np.ndarray, beta: float) -> float:
        maxVal = (self.width / 2) ** 2 + (self.height / 2) ** 2
        if distance(q, self.center) < 1.0e-3:
            return self.radius * (1 - beta / maxVal) * 1.0e4
        else:
            return self.radius * (1 - beta / maxVal) / distance(q, self.center)

    def compute_T(self, q: np.ndarray, beta: float) -> np.ndarray:
        return self.compute_v(q, beta) * (q - self.center) + self.center

    def check_point_inside(self, q: np.ndarray, threshold=0.0):
        s = self.s
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width - threshold, self.height - threshold
        potential_point = 1 - (1 / (b / 2) ** 2) * (((b / a) * (x - x_0)) ** 2 + (y - y_0) ** 2 +
                                                    (((b / a) * (x - x_0)) ** 4 + (y - y_0) ** 4 +
                                                     ((2 - 4 * s ** 2) * ((b / a) * (x - x_0)) ** 2 * (
                                                             y - y_0) ** 2)) ** 0.5) / 2
        if potential_point >= 0.0:
            return True
        # if self.potential(q) <= threshold:
        #     return True
        return False


class Circle(object):
    def __init__(self, type: str, center: np.ndarray, radius: float):
        self.type = type
        self._center = center
        self._radius = radius

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def potential(self, q: np.ndarray) -> float:
        return distance(q, self.center) ** 2 - self.radius ** 2

    def grad_potential(self, q: np.ndarray) -> float:
        return 2 * (q - self.center)

    def compute_v(self, q: np.ndarray, beta: float) -> float:
        return self.radius * (1 + beta) / distance(q, self.center)

    def compute_T(self, q: np.ndarray, beta: float) -> np.ndarray:
        return self.compute_v(q, beta) * (q - self.center) + self.center

    def check_point_inside(self, q: np.ndarray):
        if self.potential(q) < 0.0:
            return True
        return False
