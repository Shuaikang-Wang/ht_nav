import os
import sys

sys.path.append(os.getcwd())


import numpy as np
from shapely.geometry import Polygon
from shapely.ops import nearest_points


from LSE.line_process import Line_process
from LSE.features import FeaturesDetection

from shapely import LineString, Point


class PointToLine(object):
    def __init__(self, robot, world, ANGLE_OF_CHANGE_PARAM=3, threshold=0.1):
        self.ANGLE_OF_CHANGE_PARAM = ANGLE_OF_CHANGE_PARAM
        self.threshold = threshold
        self.robot = robot
        self.world = world

    def points_is_in_polygon(self, polygon, point):
        point = Point(point[0], point[1])
        real_polygon = Polygon(polygon.vertices)

        distance = point.distance(real_polygon)
        # print("distance",distance)
        if distance <= self.threshold:
            return True
        else:
            return False

    def trans_point_on_line(self, point, polygon):
        point = Point(point[0], point[1])
        real_polygon = Polygon(polygon.vertices)
        p1, p2 = nearest_points(real_polygon, point)
        return p1

    def point_approximate(self, point, polygonobs):
        x = point[0]
        y = point[1]
        x_list = np.array([point[0] for point in polygonobs.vertices])
        y_list = np.array([point[1] for point in polygonobs.vertices])
        x_diff = np.abs(x - x_list)
        y_diff = np.abs(y - y_list)
        x_mask = x_diff < 0.02
        y_mask = y_diff < 0.02
        if np.any(x_mask):
            x = x_list[x_mask][0]
        if np.any(y_mask):
            y = y_list[y_mask][0]
        return [x, y]

    @staticmethod
    def compute_normal(robot_point, line):
        end_point = line[0]
        start_point = line[1]
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        point_vector_x = robot_point[0] - start_point[0]
        point_vector_y = robot_point[1] - start_point[1]

        cross_product = dx * point_vector_y - dy * point_vector_x
        vector = np.array([end_point[0] - start_point[0],
                           end_point[1] - start_point[1]])
        if cross_product > 0:
            return np.array([-vector[1], vector[0]])
        else:
            return np.array([vector[1], -vector[0]])

    def update_line(self):
        ws_obs_set = []
        for ws in self.world.workspace:
            if len(ws) == 1:
                continue
            ws_1 = ws[1]
            if len(ws_1.local_points) != 0:
                ws_obs_set.append(ws_1)
        for obs in self.world.obstacles:
            is_sphere = False
            for obs_i in obs:
                if obs_i.s < 0.1:
                    is_sphere = True
                    break
            if not is_sphere:
                obs_0 = obs[0]
                if len(obs_0.local_points) != 0:
                    ws_obs_set.append(obs_0)

        for obs in ws_obs_set:
            current_points = obs.local_points  # lidar points within each obstacle in current time step

            gcs = Line_process()
            FeatureMAP = FeaturesDetection()
            gcs.all_line_segment_class = obs.accumulated_line_class
            gcs.decompose_line_process()

            position = self.robot.pose[0:2]
            current_points = np.array(current_points)
            FeatureMAP.LASERPOINTS = current_points

            BREAK_POINT_IND = 0
            FeatureMAP.NP = len(FeatureMAP.LASERPOINTS) - 1

            while BREAK_POINT_IND < (FeatureMAP.NP - FeatureMAP.PMIN):
                seedSeg = FeatureMAP.seed_segment_detection(position, BREAK_POINT_IND)
                if not seedSeg:
                    break
                else:
                    seedSegment = seedSeg[0]
                    INDICES = seedSeg[1]
                    results = [FeatureMAP.LINE_PARAMS, seedSeg[1][0], seedSeg[1][1]]

                    if not results:
                        BREAK_POINT_IND = INDICES[1]
                        continue
                    else:
                        line_eq = results[0]
                        PB = results[1]
                        PF = results[2]
                        BREAK_POINT_IND = PF
                        ENDPOINTS = FeatureMAP.projection_point2line(line_eq, FeatureMAP.LASERPOINTS[[PB, PF - 1]])

                        if line_eq[0] != 0 and line_eq[1] != 0:
                            midpoint = np.mean(ENDPOINTS, axis=0)
                            if abs(line_eq[0]) - abs(line_eq[1]) > 0:
                                ENDPOINTS = np.array([[midpoint[0], ENDPOINTS[0][1]], [midpoint[0], ENDPOINTS[1][1]]])
                                line_eq = FeatureMAP.odr_fit(ENDPOINTS)
                            else:

                                ENDPOINTS = np.array([[ENDPOINTS[0][0], midpoint[1]], [ENDPOINTS[1][0], midpoint[1]]])
                                line_eq = FeatureMAP.odr_fit(ENDPOINTS)

                        normal_direction = self.compute_normal(position, ENDPOINTS)
                        # print("normal_direction", normal_direction)
                        normal_direction = normal_direction / (np.linalg.norm(normal_direction) + 1e-5)
                        FeatureMAP.FEATURES.append(
                            [line_eq, ENDPOINTS, normal_direction, seedSegment, self.robot.step, True])

            if not gcs.all_line_segment_list:
                gcs.all_line_segment_list = FeatureMAP.FEATURES
                gcs.merge_myline()
            else:
                for i in range(0, len(FeatureMAP.FEATURES)):
                    flag = 0
                    for j in range(0, len(gcs.all_line_segment_list)):
                        New_Line = gcs.merge_2line(gcs.all_line_segment_list[j], FeatureMAP.FEATURES[i])
                        if New_Line:
                            gcs.all_line_segment_list[j] = New_Line
                            flag = 1
                    if flag == 0:
                        gcs.all_line_segment_list.append(FeatureMAP.FEATURES[i])

            gcs.merge_myline()
            gcs.whether_add_corner(self.robot)
            gcs.generate_line_process()

            for i in range(0, len(gcs.all_line_segment_class)):
                flag = 1
                for j in range(0, len(obs.accumulated_line_list)):
                    if np.array_equal(gcs.all_line_segment_class[i].endpoint, obs.accumulated_line_class[j].endpoint):
                        flag = 0
                        gcs.all_line_segment_class[i].changed = False
                        gcs.all_line_segment_class[i].step = obs.accumulated_line_class[j].step
                if flag == 1:
                    gcs.all_line_segment_class[i].changed = True
                    gcs.all_line_segment_class[i].step = self.robot.step

            obs.accumulated_line_class = gcs.all_line_segment_class
            obs.accumulated_line_list = gcs.all_line_segment_list
