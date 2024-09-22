import math

import numpy as np
from shapely import LineString, Point
import matplotlib.pyplot as plt
from scipy import stats

step = 0
changed = False

angle_threshold = 30
line_distance_threshold = 0.3
ADD_CORNER_THRESHOLD = 0.3


class Line(object):
    def __init__(self):
        self.param = None
        self.endpoint = None
        self.direction = None
        self.points_on_line = None
        self.step = 0
        self.changed = False

    def decompose_line(self):
        line = [self.param, self.endpoint, self.direction, self.points_on_line, self.step, self.changed]
        return line

    def generate_line(self, line):
        self.param = line[0]
        self.endpoint = line[1]
        self.direction = line[2]
        self.points_on_line = line[3]
        self.step = line[4]
        self.changed = line[5]


def projection_line2line(line1, line2):
    endpoint1 = line2[1][0]  # line2 's endpoint1
    endpoint2 = line2[1][1]
    w = np.array([a, b])
    projection1 = endpoint1 - w * (np.dot(endpoint1, w) + c) / (np.linalg.norm(w) ** 2)
    projection2 = endpoint2 - w * (np.dot(endpoint2, w) + c) / (np.linalg.norm(w) ** 2)
    return np.array([projection1, projection2])


def compute_slope_difference(line1, line2):
    a1, b1, c1 = line1[0]
    a2, b2, c2 = line2[0]
    endpoint1, endpoint2 = line1[1][0], line1[1][1]
    endpoint3, endpoint4 = line2[1][0], line2[1][1]
    segment1 = LineString([endpoint1, endpoint2])
    segment2 = LineString([endpoint3, endpoint4])
    distance = segment1.distance(segment2)

    if b1 == 0 and b2 != 0:
        slope2 = -a2 / b2
        angle2 = math.degrees(math.atan(slope2))
        angle1 = 90
        difference = abs(angle1 - angle2)
        if difference >= 90:
            difference = 180 - difference
        if difference < angle_threshold and distance <= line_distance_threshold:
            return True
        else:
            return False

        return False
    if b1 != 0 and b2 == 0:
        slope1 = -a1 / b1
        angle1 = math.degrees(math.atan(slope1))
        angle2 = 90
        difference = abs(angle1 - angle2)
        if difference >= 90:
            difference = 180 - difference
        if difference < angle_threshold and distance <= line_distance_threshold:
            return True
        else:
            return False
        return False

    if b1 == 0 and b2 == 0 and distance <= line_distance_threshold:
        return True

    slope1 = -a1 / b1
    angle1 = math.degrees(math.atan(slope1))
    slope2 = -a2 / b2
    angle2 = math.degrees(math.atan(slope2))
    difference = abs(angle1 - angle2)
    if difference >= 90:
        difference = 180 - difference

    if difference <= angle_threshold and distance <= line_distance_threshold:

        return True
    else:

        return False


def compute_max_distance_point(points):
    max_distance = 0
    max_index = 0
    for i in range(0, 3):
        distance = np.linalg.norm(np.array(points[i]) - np.array(points[2]))
        if distance > max_distance:
            max_index = i
            max_distance = distance
    return np.array([points[max_index], points[2]])


class Line_process():
    def __init__(self):
        self.all_line_segment_list = []
        self.points_unprocessed = np.zeros(0)
        self.all_line_segment_class = []

    def decompose_line_process(self):
        line_list = []
        for line_class in self.all_line_segment_class:
            line = [[], [], [], [], [], []]
            line[0] = line_class.param
            line[1] = line_class.endpoint
            line[2] = line_class.direction
            line[3] = line_class.points_on_line
            line[4] = line_class.step
            line[5] = line_class.changed
            line_list.append([line[0], line[1], line[2], line[3], line[4], line[5]])
        self.all_line_segment_list = line_list

    def generate_line_process(self):
        line_class_list = []
        for line in self.all_line_segment_list:
            line_class = Line()
            line_class.param = line[0]
            line_class.endpoint = line[1]
            line_class.direction = line[2]
            line_class.points_on_line = line[3]
            line_class.step = line[4]
            line_class.changed = line[5]
            line_class_list.append(line_class)
        self.all_line_segment_class = line_class_list

    def whether_merge(self, line1, line2):
        a1, b1, c1 = line1[0]
        a2, b2, c2 = line2[0]
        endpoint1, endpoint2 = line1[1][0], line1[1][1]
        endpoint3, endpoint4 = line2[1][0], line2[1][1]

        segment1 = LineString([endpoint1, endpoint2])
        segment2 = LineString([endpoint3, endpoint4])
        distance = segment1.distance(segment2)
        # print("aaa")
        if b1 == 0 and b2 != 0:
            slope2 = -a2 / b2
            angle2 = math.degrees(math.atan(slope2))
            angle1 = 90
            difference = abs(angle1 - angle2)
            if difference >= 90:
                difference = 180 - difference
            if difference < angle_threshold and distance <= line_distance_threshold:
                return True
            else:
                return False

            return False
        if b1 != 0 and b2 == 0:
            slope1 = -a1 / b1
            angle1 = math.degrees(math.atan(slope1))
            angle2 = 90
            difference = abs(angle1 - angle2)
            if difference >= 90:
                difference = 180 - difference
            if difference < angle_threshold and distance <= line_distance_threshold:
                return True
            else:
                return False
            return False

        if b1 == 0 and b2 == 0 and distance <= line_distance_threshold:
            return True

        slope1 = -a1 / (b1 + 1e-5)
        angle1 = math.degrees(math.atan(slope1))
        slope2 = -a2 / (b2 + 1e-5)
        angle2 = math.degrees(math.atan(slope2))
        difference = abs(angle1 - angle2)
        if difference >= 90:
            difference = 180 - difference

        if difference <= angle_threshold and distance <= line_distance_threshold:
            return True
        else:
            return False

    def get_intersection_point(self, Line1, Line2):
        a1, b1, c1 = Line1
        a2, b2, c2 = Line2
        if a1 * b2 == a2 * b1 and a1 != 0 and b1 != 0 and a2 != 0 and b2 != 0:
            return None
        elif a1 == a2 and a1 == 0 and b1 != b2:
            return None
        elif b1 == b2 and b1 == 0 and a1 != a2:
            return None
        else:
            x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1 + 1e-5)
            y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1 + 1e-5)
            return [x, y]

    def whether_add_corner(self, robot):
        length = len(self.all_line_segment_list)
        for i in range(0, length - 1):
            Line1 = LineString(self.all_line_segment_list[i][1])
            Line1_param = self.all_line_segment_list[i][0]

            for j in range(i + 1, length):
                Line2 = LineString(self.all_line_segment_list[j][1])
                Line2_param = self.all_line_segment_list[j][0]
                intersection_points = self.get_intersection_point(Line1_param, Line2_param)
                if intersection_points is None:
                    continue
                elif Line1.distance(Point(intersection_points)) <= ADD_CORNER_THRESHOLD and Line2.distance(
                        Point(intersection_points)) <= ADD_CORNER_THRESHOLD:
                    points1 = [self.all_line_segment_list[i][1][0], self.all_line_segment_list[i][1][1], intersection_points]
                    points2 = [self.all_line_segment_list[j][1][0], self.all_line_segment_list[j][1][1], intersection_points]
                    new_line1 = compute_max_distance_point(points1)
                    new_line2 = compute_max_distance_point(points2)
                    self.all_line_segment_list[i][1] = new_line1
                    self.all_line_segment_list[j][1] = new_line2

    def merge_2line(self, line1, line2):
        if not self.whether_merge(line1, line2):
            return
        else:
            whole_points = np.concatenate((line1[3], line2[3]))
            line1[3] = np.unique(whole_points, axis=0)

            endpoint = projection_line2line(line1, line2)
            point1 = line1[1][0]
            point2 = line1[1][1]
            point3 = endpoint[0]
            point4 = endpoint[1]
            Final_Endpoint = self.max_distance([point1, point2, point3, point4])
            line_old = line1
            line_new = line2
            if math.sqrt((line_old[2][0])**2 + (line_old[2][1])**2) < 1e-5:
                line_old[2] = line_new[2]

            NEW_LINE = [line1[0], Final_Endpoint, line_old[2], line1[3], step, changed]
            return NEW_LINE

    def merge_myline(self):
        new_all_line_segment_list = []
        for i in range(0, len(self.all_line_segment_list)):

            flag = 0
            if len(new_all_line_segment_list) == 0:
                new_all_line_segment_list.append(self.all_line_segment_list[i])
            for j in range(0, len(new_all_line_segment_list)):
                new_line = self.merge_2line(self.all_line_segment_list[i], new_all_line_segment_list[j])
                if new_line is not None:
                    flag = 1
                    new_all_line_segment_list[j] = new_line
                    break
            if flag == 0:
                new_all_line_segment_list.append(self.all_line_segment_list[i])
        self.all_line_segment_list = new_all_line_segment_list

    def max_distance(self, points):
        max_dist = 0
        point1, point2 = None, None

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = math.sqrt((points[j][0] - points[i][0]) ** 2 + (points[j][1] - points[i][1]) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    point1, point2 = points[i], points[j]

        return np.array([point1, point2])

    def force_mapping(self):
        for i in range(0, len(self.all_line_segment_list)):
            if self.all_line_segment_list[i][0][0] != 0 and self.all_line_segment_list[i][0][1] != 0:
                if abs(self.all_line_segment_list[i][0][0]) > abs(self.all_line_segment_list[i][0][1]):
                    self.all_line_segment_list[i][0][1] = 0

                else:
                    self.all_line_segment_list[i][0][0] = 0