import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import random
# from sympy import symbols, Eq, solve



class Lidar(object):
    def __init__(self, radius=1.0, resolution=np.pi / 18, noise_gain=0.0):
        self._radius = radius
        self._resolution = resolution
        self._noise_gain = noise_gain

    # def __init__(self, radius=1.0, resolution=np.pi / 100):
    #     self._radius = radius
    #     self._resolution = resolution

    @property
    def radius(self):
        return self._radius

    @property
    def noise_gain(self):
        return self._noise_gain

    @property
    def resolution(self):
        return self._resolution

    @staticmethod
    def potential_ca(x, y, center_x, center_y, width, height, rotation, s):
        x_0, y_0, a, b = center_x, center_y, width, height
        rotated_x = (x - x_0) * np.cos(rotation) + (y - y_0) * np.sin(rotation) + x_0
        rotated_y = -(x - x_0) * np.sin(rotation) + (y - y_0) * np.cos(rotation) + y_0
        x, y = rotated_x, rotated_y
        return (1 / (b / 2) ** 2) * (((b / a) * (x - x_0)) ** 2 + (y - y_0) ** 2 +
                                     (((b / a) * (x - x_0)) ** 4 + (y - y_0) ** 4 +
                                      ((2 - 4 * s ** 2) * ((b / a) * (x - x_0)) ** 2 * (y - y_0) ** 2)) ** 0.5) / 2 - 1

    def find_furthest_free_point(self, pose, theta, world, obs=False):
        if obs:
            # x, y = symbols('x y')
            # x1, y1 = pose[0], pose[1]
            # x2, y2 = pose[0] + self.radius * np.cos(theta), pose[1] + self.radius * np.sin(theta)
            # print("x1, y1, x2, y2", x1, y1, x2, y2)
            # line_equation = Eq((y2 - y1) * x - (x2 - x1) * y + x1 * y2 - x2 * y1, 0)
            # near_intersect = []
            # for obs_group in world.obstacles:
            #     for obs in obs_group:
            #         if not obs.check_point_inside_accurate(np.array([x2, y2])):
            #             continue
            #         squircle_equation = Eq(self.potential_ca(x, y, obs.center[0], obs.center[1], obs.width, obs.height, 0.5, 0.99), 0)
            #         print("squircle", obs.center[0], obs.center[1], obs.width, obs.height)
            #         y_expr = solve(line_equation, y)[0]
            #         squircle_equation_substituted = squircle_equation.subs(y, y_expr)
            #         x_values = solve(squircle_equation_substituted, x, dict=True)
            #         y_values = [y_expr.subs(x, x_val[x]) for x_val in x_values]
            #
            #         valid_intersections = []
            #         for x_val, y_val in zip(x_values, y_values):
            #             if min(x1, x2) <= x_val[x] <= max(x1, x2):
            #                 valid_intersections.append((x_val[x], y_val))
            #         print("valid_intersections", valid_intersections)
            #         if len(valid_intersections) == 2:
            #             intersect_1 = valid_intersections[0]
            #             intersect_2 = valid_intersections[1]
            #             if np.sqrt((pose[0] - intersect_1[0])**2 + (pose[1] - intersect_1[1])**2) < \
            #                     np.sqrt((pose[0] - intersect_2[0])**2 + (pose[1] - intersect_2[1])**2):
            #                 near_intersect = intersect_1
            #             else:
            #                 near_intersect = intersect_2
            #         elif len(valid_intersections) == 1:
            #             near_intersect = valid_intersections[0]
            # print('near_intersect', near_intersect)
            # return near_intersect
            for dist in np.arange(0, self.radius, 0.001):
                point = np.array(pose[0:2]) + np.array([dist * np.cos(theta), dist * np.sin(theta)])
                if not world.workspace[0][0].check_point_inside(point):
                    return
                for ws in world.workspace:
                    if len(ws) == 1:
                        continue
                    ws_1 = ws[1]
                    for ws_i in ws[1:]:
                        if ws_i.check_point_inside(point):
                            point_with_robot = np.array([point[0], point[1], pose[0], pose[1], ws_i.theta]) # x, y, x_r, y_r, theta
                            ws_1.local_points.append(point_with_robot)
                            ws_1.accumulated_local_points.append(point_with_robot)
                            return
                for obs in world.obstacles:
                    obs_0 = obs[0]
                    for obs_i in obs:
                        if obs_i.check_point_inside(point):
                            point_with_robot = np.array([point[0], point[1], pose[0], pose[1], obs_i.theta])
                            obs_0.local_points.append(point_with_robot)
                            obs_0.accumulated_local_points.append(point_with_robot)
                            return
            # return point
        else:
            point = []
            for dist in np.arange(0, self.radius, 0.01):
                point = np.array(pose[0:2]) + np.array([dist * np.cos(theta), dist * np.sin(theta)])
                if not world.check_point_in_free_space(point):
                    point = [point[0] + self.noise_gain * random.random(), point[1] + self.noise_gain * random.random()]
                    return point
            return point

    def get_points_in_range(self, pose, world, obs=False):
        data_points = []
        for theta in np.arange(0, 2 * np.pi, self.resolution):
            last_point = self.find_furthest_free_point(pose, theta, world, obs=obs)
            if last_point is not None:
                data_points.append(last_point)
        return np.array(data_points)

    def get_measurements(self, pose, world, obs=False):
        if obs:
            for ws in world.workspace:
                if len(ws) == 1:
                    continue
                ws_1 = ws[1]
                ws_1.local_points = []
            for obs in world.obstacles:
                obs_0 = obs[0]
                obs_0.local_points = []
        data_points = self.get_points_in_range(pose, world, obs=obs)
        return data_points
