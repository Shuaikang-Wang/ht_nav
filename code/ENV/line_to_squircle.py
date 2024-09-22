import os
import sys

sys.path.append(os.getcwd())

import math
import numpy as np


class LineToSquircle(object):
    def __init__(self, point_1, point_2, vector, extension=0.4):
        self.start_point = point_1
        self.end_point = point_2
        self.extension = extension
        self.extended_start_point = None
        self.extended_end_point = None
        self.extended_side_start_point = None
        self.extended_side_end_point = None
        self.extend_line_segment()
        self.vector = vector

    def extend_line_segment(self):
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        length = math.sqrt(dx ** 2 + dy ** 2) + 1e-5

        extended_start_point = (
            self.start_point[0] - (dx / length) * self.extension,
            self.start_point[1] - (dy / length) * self.extension
        )
        extended_end_point = (
            self.end_point[0] + (dx / length) * self.extension,
            self.end_point[1] + (dy / length) * self.extension
        )

        self.extended_start_point = extended_start_point
        self.extended_end_point = extended_end_point

    def midpoint(self):
        return np.array([(self.extended_start_point[0] + self.extended_end_point[0]) / 2,
                         (self.extended_start_point[1] + self.extended_end_point[1]) / 2])

    @staticmethod
    def scale_vector(vector, scale):
        return np.array([vector[0] * scale, vector[1] * scale])

    @staticmethod
    def distance_between_points(point_1, point_2):
        return ((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2) ** 0.5

    def find_point_on_perpendicular_line(self, distance):
        mid_point = self.midpoint()

        perpendicular_direction = np.array(self.vector)
        # print(perpendicular_direction)

        magnitude = self.distance_between_points(np.array([0, 0]), perpendicular_direction) + 1e-5
        normalized_direction = np.array(
            [perpendicular_direction[0] / magnitude, perpendicular_direction[1] / magnitude])

        point_on_perpendicular = self.scale_vector(normalized_direction, distance / 2)

        point_on_segment = np.array(
            [mid_point[0] + point_on_perpendicular[0], mid_point[1] + point_on_perpendicular[1]])

        point_on_perpendicular_double = self.scale_vector(normalized_direction, distance)

        self.extended_side_start_point = np.array(
            [self.extended_start_point[0] + point_on_perpendicular_double[0],
             self.extended_start_point[1] + point_on_perpendicular_double[1]])

        self.extended_side_end_point = np.array(
            [self.extended_end_point[0] + point_on_perpendicular_double[0],
             self.extended_end_point[1] + point_on_perpendicular_double[1]])

        return point_on_segment

    def line_to_squircle(self):
        center = self.find_point_on_perpendicular_line(self.extension)
        width = self.distance_between_points(self.extended_start_point, self.extended_end_point)
        height = self.extension
        theta = np.arctan2(self.vector[1],
                           self.vector[0]) + np.pi / 2

        if abs(theta) < np.pi / 4 or abs(theta - np.pi) < np.pi / 4 or abs(theta + np.pi) < np.pi / 4:
            width = width
            height = height
        else:
            temp = width
            width = height
            height = temp
        theta = 0.0
        return center, width, height, theta
