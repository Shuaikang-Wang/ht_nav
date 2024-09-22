import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from NM.delaunay import DelaunayTriangulation
from NM.vis_graph import VisGraph, Point
from NM.polygon_extend import Building

from shapely.geometry import Point as Point_spl
from shapely.geometry import Polygon as Polygon_spl


class NavigationMap(object):
    def __init__(self, planner_type='vis', inflated_size=0.0):
        self.planner_type = planner_type
        self.delaunay_planner = None
        self.vis_graph_planner = None
        self.path = None
        self.inflated_size = inflated_size

    @staticmethod
    def point_to_segment_distance(segment_start, segment_end, point):
        def dot_product(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]

        def magnitude(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def distance(v1, v2):
            return magnitude((v1[0] - v2[0], v1[1] - v2[1]))

        segment_vector = (segment_end[0] - segment_start[0], segment_end[1] - segment_start[1])
        point_vector = (point[0] - segment_start[0], point[1] - segment_start[1])

        if dot_product(point_vector, segment_vector) <= 0:
            return distance(point, segment_start)

        segment_length_squared = dot_product(segment_vector, segment_vector)
        if dot_product(point_vector, segment_vector) >= segment_length_squared:
            return distance(point, segment_end)

        projection_length = dot_product(point_vector, segment_vector) / segment_length_squared
        projection = (segment_start[0] + projection_length * segment_vector[0],
                      segment_start[1] + projection_length * segment_vector[1])

        return distance(point, projection)

    def construct_planner_rect(self, start_pose, goal_pose, polygon_list):
        start_point = start_pose[0:2]
        goal_point = goal_pose[0:2]
        self.vis_graph_planner = VisGraph()
        polys_vg = []
        for poly_i in polygon_list:
            poly_vg_i = []
            for point_j in poly_i:
                poly_vg_i.append(Point(point_j[0], point_j[1]))
            polys_vg.append(poly_vg_i)
        self.vis_graph_planner.build(polys_vg)

        symmetric_start_point = start_point
        is_symmetric = False
        for k, polygon in enumerate(polygon_list):
            if self.is_point_inside_polygon(start_point, polygon):
                min_index = None
                min_dis = 100
                is_symmetric = True
                for i, vertex_i in enumerate(polygon):
                    if self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                      start_point) < min_dis:
                        min_index = i
                        min_dis = self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                                 start_point)
                symmetric_start_point = self.symmetric_point(start_point[0: 2],
                                                             [polygon[min_index % len(polygon)],
                                                              polygon[(min_index + 1) % len(polygon)]],
                                                             dis=0.02)
                # print("symmetric_start_point", symmetric_start_point)
                break

        shortest = self.vis_graph_planner.shortest_path(
            Point(symmetric_start_point[0], symmetric_start_point[1]),
            Point(goal_point[0], goal_point[1]))
        if is_symmetric:
            path = [start_point]
        else:
            path = []

        for i, point_i in enumerate(shortest[: -1]):
            if i == 0 and is_symmetric:
                continue
            if i == 0:
                theta_i = 0.0
            else:
                theta_i = np.arctan2(point_i.y - shortest[i - 1].y, point_i.x - shortest[i - 1].x)
            theta_i_1 = np.arctan2(shortest[i + 1].y - point_i.y, shortest[i + 1].x - point_i.x)
            if theta_i < 0:
                theta_i = theta_i + 2 * np.pi
            if theta_i_1 < 0:
                theta_i_1 = theta_i_1 + 2 * np.pi
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%index%%%%%%%%%%%%%%%%%%%%%%%%%%", i)
            # print("theta_i", theta_i)
            # print("theta_i_1", theta_i_1)
            theta_diff = (theta_i_1 - theta_i) / 2
            theta = theta_diff + theta_i
            if abs(theta_i_1 - theta_i) > np.pi:
                theta = theta - np.pi
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            # print("theta", theta)
            path.append([point_i.x, point_i.y,
                         theta])
        path.append([shortest[-1].x, shortest[-1].y, goal_pose[2]])
        self.path = path

    def construct_planner_rect_multi_goal(self, start_pose, goal_pose_list, polygon_list):
        self.vis_graph_planner = VisGraph()
        polys_vg = []
        for poly_i in polygon_list:
            poly_vg_i = []
            for point_j in poly_i:
                poly_vg_i.append(Point(point_j[0], point_j[1]))
            polys_vg.append(poly_vg_i)
        self.vis_graph_planner.build(polys_vg)

        all_path = []
        is_symmetric = False
        for i, goal_pose in enumerate(goal_pose_list):
            goal_point = goal_pose[0:2]
            if i == 0:
                start_point = start_pose[0:2]
                symmetric_start_point = start_point
                
                for polygon in polygon_list:
                    if self.is_point_inside_polygon(start_point, polygon):
                        min_index = None
                        min_dis = 100
                        is_symmetric = True
                        for i, vertex_i in enumerate(polygon):
                            if self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                            start_point) < min_dis:
                                min_index = i
                                min_dis = self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                                        start_point)
                        symmetric_start_point = self.symmetric_point(start_point[0: 2],
                                                                    [polygon[min_index % len(polygon)],
                                                                    polygon[(min_index + 1) % len(polygon)]],
                                                                    dis=0.02)
                        break

                shortest = self.vis_graph_planner.shortest_path(
                    Point(symmetric_start_point[0], symmetric_start_point[1]),
                    Point(goal_point[0], goal_point[1]))
                all_path += shortest
            else:
                start_point = goal_pose_list[i - 1][0:2]
                shortest = self.vis_graph_planner.shortest_path(
                    Point(start_point[0], start_point[1]),
                    Point(goal_point[0], goal_point[1]))
                all_path += shortest[1:]
        
        if is_symmetric:
            path = [start_pose[0:2]]
        else:
            path = []

        shortest = all_path
        for i, point_i in enumerate(shortest[: -1]):
            if i == 0 and is_symmetric:
                continue
            if i == 0:
                theta_i = 0.0
            else:
                theta_i = np.arctan2(point_i.y - shortest[i - 1].y, point_i.x - shortest[i - 1].x)
            theta_i_1 = np.arctan2(shortest[i + 1].y - point_i.y, shortest[i + 1].x - point_i.x)
            if theta_i < 0:
                theta_i = theta_i + 2 * np.pi
            if theta_i_1 < 0:
                theta_i_1 = theta_i_1 + 2 * np.pi
            theta_diff = (theta_i_1 - theta_i) / 2
            theta = theta_diff + theta_i
            if abs(theta_i_1 - theta_i) > np.pi:
                theta = theta - np.pi
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            # print("theta", theta)
            path.append([point_i.x, point_i.y,
                         theta])
            
        path.append([shortest[-1].x, shortest[-1].y, goal_pose[2]])
        self.path = path
            
            

    def construct_planner_no_ws(self, start_point, goal_point, polygon_list):
        if self.planner_type == 'delaunay':
            self.delaunay_planner = DelaunayTriangulation()
        else:
            self.vis_graph_planner = VisGraph()
            inflated_polygon_list = []
            for polygon in polygon_list:
                inflated_polygon = self.polygon_extend(polygon)
                # print("inflated_polygon", inflated_polygon)
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
                inflated_polygon_list.append(inflated_polygon)

            union_polygon = Polygon_spl(inflated_polygon_list[0])
            for polygon_i in inflated_polygon_list[1:]:
                # print("polygon_i", polygon_i)
                # print("union", union_polygon)
                polygon_spl_i = Polygon_spl(polygon_i)
                # print("polygon_spl_i", polygon_spl_i)
                union_polygon = union_polygon.union(polygon_spl_i)

            # print("inflated polygon list before", len(inflated_polygon_list))
            inflated_polygon_list = []
            if type(union_polygon).__name__ == "Polygon":
                inflated_polygon_list.append(list(union_polygon.exterior.coords))
            else:
                for polygon_i in union_polygon.geoms:
                    inflated_polygon_list.append(list(polygon_i.exterior.coords))

            # print("inflated polygon list after", len(inflated_polygon_list))

            polys_vg = []
            for poly_i in inflated_polygon_list:
                poly_vg_i = []
                for point_j in poly_i:
                    poly_vg_i.append(Point(point_j[0], point_j[1]))
                polys_vg.append(poly_vg_i)
            self.vis_graph_planner.build(polys_vg)

            symmetric_start_point = start_point
            is_symmetric = False
            for k, polygon in enumerate(inflated_polygon_list):
                if self.is_point_inside_polygon(start_point, polygon):
                    min_index = None
                    min_dis = 100
                    is_symmetric = True
                    for i, vertex_i in enumerate(polygon):
                        if self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                          start_point) < min_dis:
                            min_index = i
                            min_dis = self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                                     start_point)
                    symmetric_start_point = self.symmetric_point(start_point[0: 2],
                                                                 [polygon[min_index % len(polygon)],
                                                                  polygon[(min_index + 1) % len(polygon)]],
                                                                 dis=0.02)
                    # print("symmetric_start_point", symmetric_start_point)
                    break

            shortest = self.vis_graph_planner.shortest_path(
                Point(symmetric_start_point[0], symmetric_start_point[1]),
                Point(goal_point[0], goal_point[1]))
            if is_symmetric:
                path = [start_point]
            else:
                path = []

            for i, point_i in enumerate(shortest[: -1]):
                if i == 0 and is_symmetric:
                    continue
                if i == 0:
                    theta_i = 0.0
                else:
                    theta_i = np.arctan2(point_i.y - shortest[i - 1].y, point_i.x - shortest[i - 1].x)
                theta_i_1 = np.arctan2(shortest[i + 1].y - point_i.y, shortest[i + 1].x - point_i.x)
                path.append([point_i.x, point_i.y,
                             (theta_i + theta_i_1) / 2])
            path.append([shortest[-1].x, shortest[-1].y, path[-1][2]])
            self.path = path

    def construct_planner(self, start_point, goal_point, polygon_list):
        """"
        planner type:
            delaunay
            vis
        """
        if self.planner_type == 'delaunay':
            self.delaunay_planner = DelaunayTriangulation()
        else:
            self.vis_graph_planner = VisGraph()
            inflated_polygon_list = []
            if len(polygon_list) == 1:
                # print("polygon_list[0]", polygon_list[0])
                inflated_polygon = self.inflate_ws_polygon(polygon_list[0])
                inflated_polygon_list.append(inflated_polygon)
            else:
                inflated_polygon = self.inflate_ws_polygon(polygon_list[0])
                inflated_polygon_list.append(inflated_polygon)
                for polygon in polygon_list[1:]:
                    inflated_polygon = self.inflate_polygon(polygon)
                    inflated_polygon_list.append(inflated_polygon)
            polys_vg = []
            for poly_i in inflated_polygon_list:
                poly_vg_i = []
                for point_j in poly_i:
                    poly_vg_i.append(Point(point_j[0], point_j[1]))
                polys_vg.append(poly_vg_i)
            self.vis_graph_planner.build(polys_vg)

            symmetric_start_point = start_point
            is_symmetric = False
            for k, polygon in enumerate(inflated_polygon_list):
                if k == 0:
                    # print("========test=======")
                    # print("start point", start_point)
                    # print("polygon", polygon)
                    # print("point inside ws", self.is_point_inside_polygon(start_point, polygon))
                    if not self.is_point_inside_polygon(start_point, polygon):
                        min_index = None
                        min_dis = 100
                        is_symmetric = True
                        for i, vertex_i in enumerate(polygon):
                            if self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                              start_point) < min_dis:
                                min_index = i
                                min_dis = self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                                         start_point)
                        symmetric_start_point = self.symmetric_point(start_point[0: 2],
                                                                     [polygon[min_index % len(polygon)],
                                                                      polygon[(min_index + 1) % len(
                                                                          polygon)]], dis=0.02)
                        # print("symmetric_start_point", symmetric_start_point)
                        break
                else:
                    if self.is_point_inside_polygon(start_point, polygon):
                        min_index = None
                        min_dis = 100
                        is_symmetric = True
                        for i, vertex_i in enumerate(polygon):
                            if self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                              start_point) < min_dis:
                                min_index = i
                                min_dis = self.point_to_segment_distance(vertex_i, polygon[(i + 1) % len(polygon)],
                                                                         start_point)
                        symmetric_start_point = self.symmetric_point(start_point[0: 2],
                                                                     [polygon[min_index % len(polygon)],
                                                                      polygon[(min_index + 1) % len(polygon)]],
                                                                     dis=0.02)
                        print("symmetric_start_point", symmetric_start_point)
                        break

            shortest = self.vis_graph_planner.shortest_path(
                Point(symmetric_start_point[0], symmetric_start_point[1]),
                Point(goal_point[0], goal_point[1]))
            if is_symmetric:
                path = [start_point]
            else:
                path = []
            for i, point_i in enumerate(shortest[: -1]):
                if i == 0 and is_symmetric:
                    continue
                theta_i = np.arctan2(point_i.y - shortest[i - 1].y, point_i.x - shortest[i - 1].x)
                theta_i_1 = np.arctan2(shortest[i + 1].y - point_i.y, shortest[i + 1].x - point_i.x)
                path.append([point_i.x, point_i.y,
                             (theta_i + theta_i_1) / 2])
            path.append([shortest[-1].x, shortest[-1].y, path[-1][2]])
            self.path = path

    @staticmethod
    def is_point_inside_polygon(point, polygon):
        point = Point_spl(point)
        polygon = Polygon_spl(polygon)
        return polygon.contains(point)

    @staticmethod
    def unit_normal_vector(point1, point2):
        direction_vector = np.array(point2) - np.array(point1)

        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return None
        unit_vector = direction_vector / norm

        unit_normal = np.array([-unit_vector[1], unit_vector[0]])
        return unit_normal

    @staticmethod
    def project_point_to_segment(point, segment):
        p1 = segment[0]
        p2 = segment[1]
        segment_vector = np.array(p2) - np.array(p1)
        point_vector = np.array(point) - np.array(p1)
        # print("segment_vector", segment_vector)
        t = np.dot(point_vector, segment_vector) / (np.dot(segment_vector, segment_vector) + 1e-5)
        # print("t", t)
        if t < 0:
            projection = p1
        elif t > 1:
            projection = p2
        else:
            projection = p1 + t * segment_vector
        return projection

    def symmetric_point(self, point, line_segment, dis=None):
        if not dis:
            projection = self.project_point_to_segment(point, line_segment)
            vector_to_projection = np.array(projection) - np.array(point)
            symmetric_point = np.array(projection) + vector_to_projection
        else:
            projection = self.project_point_to_segment(point, line_segment)
            vector_to_projection = np.array(projection) - np.array(point)
            symmetric_point = np.array(projection) + dis * vector_to_projection / np.linalg.norm(vector_to_projection)
        return symmetric_point

    def polygon_extend(self, vertices):
        vertices_list = [list(vertex) for vertex in vertices]
        # print("vertices", list(vertices_list))
        b = Building(np.array(vertices_list), self.inflated_size)
        return b.expansion_anchors

    def inflate_polygon(self, vertices):
        inflated_vertices = []
        threshold = 1e-5
        for vertex in vertices:
            inflated_left_minus = [vertex[0] - threshold, vertex[1] - threshold]
            inflated_left_plus = [vertex[0] - threshold, vertex[1] + threshold]
            inflated_right_minus = [vertex[0] + threshold, vertex[1] - threshold]
            inflated_right_plus = [vertex[0] + threshold, vertex[1] + threshold]
            inflated_bottom_minus = [vertex[0] - threshold, vertex[1] - threshold]
            inflated_bottom_plus = [vertex[0] + threshold, vertex[1] - threshold]
            inflated_top_minus = [vertex[0] - threshold, vertex[1] + threshold]
            inflated_top_plus = [vertex[0] + threshold, vertex[1] + threshold]
            inflated_left = [inflated_left_minus, inflated_left_plus]
            inflated_right = [inflated_right_minus, inflated_right_plus]
            inflated_bottom = [inflated_bottom_minus, inflated_bottom_plus]
            inflated_top = [inflated_top_minus, inflated_top_plus]
            all_inflated = [inflated_left, inflated_right, inflated_bottom, inflated_top]

            all_free_list = []
            for k, inflate_pair in enumerate(all_inflated):
                if (not self.is_point_inside_polygon(inflate_pair[0], vertices)) and \
                        (not self.is_point_inside_polygon(inflate_pair[1], vertices)):
                    if k == 0:
                        inflate_pair = [[inflate_pair[0][0] + threshold - self.inflated_size, inflate_pair[0][1]],
                                        [inflate_pair[1][0] + threshold - self.inflated_size, inflate_pair[1][1]]
                                        ]
                    if k == 1:
                        inflate_pair = [[inflate_pair[0][0] - threshold + self.inflated_size, inflate_pair[0][1]],
                                        [inflate_pair[1][0] - threshold + self.inflated_size, inflate_pair[1][1]]
                                        ]
                    if k == 2:
                        inflate_pair = [[inflate_pair[0][0], inflate_pair[0][1] + threshold - self.inflated_size],
                                        [inflate_pair[1][0], inflate_pair[1][1] + threshold - self.inflated_size]
                                        ]
                    if k == 3:
                        inflate_pair = [[inflate_pair[0][0], inflate_pair[0][1] - threshold + self.inflated_size],
                                        [inflate_pair[1][0], inflate_pair[1][1] - threshold + self.inflated_size]
                                        ]
                    inflate = [(inflate_pair[0][0] + inflate_pair[1][0]) / 2,
                               (inflate_pair[0][1] + inflate_pair[1][1]) / 2]
                    all_free_list.append(inflate)

            if len(all_free_list) == 0:
                for k, inflate_pair in enumerate(all_inflated):
                    if (not self.is_point_inside_polygon(inflate_pair[0], vertices)) and \
                            (not self.is_point_inside_polygon(inflate_pair[1], vertices)):
                        continue
                    if not self.is_point_inside_polygon(inflate_pair[0], vertices):
                        inflate_pair_1 = inflate_pair
                        if k == 0:
                            inflate_pair_1 = [[inflate_pair[0][0] + threshold - self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] + threshold - self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 1:
                            inflate_pair_1 = [[inflate_pair[0][0] - threshold + self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] - threshold + self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 2:
                            inflate_pair_1 = [[inflate_pair[0][0], inflate_pair[0][1] + threshold - self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] + threshold - self.inflated_size]
                                              ]
                        if k == 3:
                            inflate_pair_1 = [[inflate_pair[0][0], inflate_pair[0][1] - threshold + self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] - threshold + self.inflated_size]
                                              ]
                        inflate = [(inflate_pair_1[0][0] + inflate_pair_1[1][0]) / 2,
                                   (inflate_pair_1[0][1] + inflate_pair_1[1][1]) / 2]
                        all_free_list.append(inflate)
                    if not self.is_point_inside_polygon(inflate_pair[1], vertices):
                        inflate_pair_2 = inflate_pair
                        if k == 0:
                            inflate_pair_2 = [[inflate_pair[0][0] + threshold - self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] + threshold - self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 1:
                            inflate_pair_2 = [[inflate_pair[0][0] - threshold + self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] - threshold + self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 2:
                            inflate_pair_2 = [[inflate_pair[0][0], inflate_pair[0][1] + threshold - self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] + threshold - self.inflated_size]
                                              ]
                        if k == 3:
                            inflate_pair_2 = [[inflate_pair[0][0], inflate_pair[0][1] - threshold + self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] - threshold + self.inflated_size]
                                              ]
                        inflate = [(inflate_pair_2[0][0] + inflate_pair_2[1][0]) / 2,
                                   (inflate_pair_2[0][1] + inflate_pair_2[1][1]) / 2]
                        all_free_list.append(inflate)
            # print("all_free_list", all_free_list)
            inflated_vertex = self.symmetric_point(vertex, [all_free_list[0], all_free_list[1]])
            inflated_vertices.append(inflated_vertex)
        return inflated_vertices

    @staticmethod
    def scale_vector(vector, scale):
        return np.array([vector[0] * scale, vector[1] * scale])

    def inflate_ws_polygon(self, vertices):
        threshold = 1e-10
        inflated_vertices = []
        for vertex in vertices:
            inflated_left_minus = [vertex[0] - threshold, vertex[1] - threshold]
            inflated_left_plus = [vertex[0] - threshold, vertex[1] + threshold]
            inflated_right_minus = [vertex[0] + threshold, vertex[1] - threshold]
            inflated_right_plus = [vertex[0] + threshold, vertex[1] + threshold]
            inflated_bottom_minus = [vertex[0] - threshold, vertex[1] - threshold]
            inflated_bottom_plus = [vertex[0] + threshold, vertex[1] - threshold]
            inflated_top_minus = [vertex[0] - threshold, vertex[1] + threshold]
            inflated_top_plus = [vertex[0] + threshold, vertex[1] + threshold]
            inflated_left = [inflated_left_minus, inflated_left_plus]
            inflated_right = [inflated_right_minus, inflated_right_plus]
            inflated_bottom = [inflated_bottom_minus, inflated_bottom_plus]
            inflated_top = [inflated_top_minus, inflated_top_plus]
            all_inflated = [inflated_left, inflated_right, inflated_bottom, inflated_top]

            all_free_list = []
            # print("vertex", vertex)
            for k, inflate_pair in enumerate(all_inflated):
                if self.is_point_inside_polygon(inflate_pair[0], vertices) and \
                        self.is_point_inside_polygon(inflate_pair[1], vertices):
                    # print("k", k)
                    # print("before inflate pair", inflate_pair)
                    if k == 0:
                        inflate_pair = [[inflate_pair[0][0] + threshold - self.inflated_size, inflate_pair[0][1]],
                                        [inflate_pair[1][0] + threshold - self.inflated_size, inflate_pair[1][1]]
                                        ]
                    if k == 1:
                        inflate_pair = [[inflate_pair[0][0] - threshold + self.inflated_size, inflate_pair[0][1]],
                                        [inflate_pair[1][0] - threshold + self.inflated_size, inflate_pair[1][1]]
                                        ]
                    if k == 2:
                        inflate_pair = [[inflate_pair[0][0], inflate_pair[0][1] + threshold - self.inflated_size],
                                        [inflate_pair[1][0], inflate_pair[1][1] + threshold - self.inflated_size]
                                        ]
                    if k == 3:
                        inflate_pair = [[inflate_pair[0][0], inflate_pair[0][1] - threshold + self.inflated_size],
                                        [inflate_pair[1][0], inflate_pair[1][1] - threshold + self.inflated_size]
                                        ]
                    inflate = [(inflate_pair[0][0] + inflate_pair[1][0]) / 2,
                               (inflate_pair[0][1] + inflate_pair[1][1]) / 2]
                    all_free_list.append(inflate)
                    # print("end inflate pair", inflate_pair)

            # print("all_free_list 0")
            if len(all_free_list) == 0:
                for k, inflate_pair in enumerate(all_inflated):
                    if self.is_point_inside_polygon(inflate_pair[0], vertices) and \
                            self.is_point_inside_polygon(inflate_pair[1], vertices):
                        continue
                    if self.is_point_inside_polygon(inflate_pair[0], vertices):
                        # print("k", k)
                        # print("before inflate pair", inflate_pair)
                        inflate_pair_1 = inflate_pair
                        if k == 0:
                            inflate_pair_1 = [[inflate_pair[0][0] + threshold - self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] + threshold - self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 1:
                            inflate_pair_1 = [[inflate_pair[0][0] - threshold + self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] - threshold + self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 2:
                            inflate_pair_1 = [[inflate_pair[0][0], inflate_pair[0][1] + threshold - self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] + threshold - self.inflated_size]
                                              ]
                        if k == 3:
                            inflate_pair_1 = [[inflate_pair[0][0], inflate_pair[0][1] - threshold + self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] - threshold + self.inflated_size]
                                              ]
                        inflate = [(inflate_pair_1[0][0] + inflate_pair_1[1][0]) / 2,
                                   (inflate_pair_1[0][1] + inflate_pair_1[1][1]) / 2]
                        all_free_list.append(inflate)
                        # print("end inflate pair", inflate_pair_1)
                    if self.is_point_inside_polygon(inflate_pair[1], vertices):
                        # print("k", k)
                        # print("before inflate pair", inflate_pair)
                        inflate_pair_2 = inflate_pair
                        if k == 0:
                            inflate_pair_2 = [[inflate_pair[0][0] + threshold - self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] + threshold - self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 1:
                            inflate_pair_2 = [[inflate_pair[0][0] - threshold + self.inflated_size, inflate_pair[0][1]],
                                              [inflate_pair[1][0] - threshold + self.inflated_size, inflate_pair[1][1]]
                                              ]
                        if k == 2:
                            inflate_pair_2 = [[inflate_pair[0][0], inflate_pair[0][1] + threshold - self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] + threshold - self.inflated_size]
                                              ]
                        if k == 3:
                            inflate_pair_2 = [[inflate_pair[0][0], inflate_pair[0][1] - threshold + self.inflated_size],
                                              [inflate_pair[1][0], inflate_pair[1][1] - threshold + self.inflated_size]
                                              ]
                        inflate = [(inflate_pair_2[0][0] + inflate_pair_2[1][0]) / 2,
                                   (inflate_pair_2[0][1] + inflate_pair_2[1][1]) / 2]
                        all_free_list.append(inflate)
                        # print("end inflate pair", inflate_pair_2)
            # print("all_free_list", all_free_list)
            inflated_vertex = self.symmetric_point(vertex, [all_free_list[0], all_free_list[1]])
            inflated_vertices.append(inflated_vertex)
        return inflated_vertices

    @staticmethod
    def plot_polygons(ax, polygon_list, edgecolor='black'):
        for polygon in polygon_list:
            poly = Polygon(polygon, edgecolor=edgecolor, linestyle='-', fill=False)
            ax.add_patch(poly)

    def plot_inflated_polygons(self, ax, polygon_list, edgecolor='black'):
        inflated_polygon_list = []
        for polygon in polygon_list:
            inflated_polygon = self.inflate_polygon(polygon)
            inflated_polygon_list.append(inflated_polygon)

        for polygon in inflated_polygon_list:
            poly = Polygon(polygon, edgecolor=edgecolor, linestyle='--', fill=False)
            ax.add_patch(poly)

    def plot_path(self, ax):
        path_x = []
        path_y = []
        path_theta = []
        for path_i in self.path[1:-1]:
            ax.plot(path_i[0], path_i[1], '.k')
            path_x.append(path_i[0])
            path_y.append(path_i[1])
            path_theta.append(path_i[2])
        path_x.append(self.path[-1][0])
        path_y.append(self.path[-1][1])
        path_theta.append(self.path[-1][2])
        # print("final waypoint", self.path[-1])
        # for i in range(len(path_x)):
        #     ax.quiver(path_x[i], path_y[i], np.cos(path_theta[i]), np.sin(path_theta[i]), units='xy', width=0.05,
        #               headwidth=3.3, scale=1 / 0.5, color='red', zorder=2)
        near_path_x = [self.path[0][0], self.path[1][0]]
        near_path_y = [self.path[0][1], self.path[1][1]]
        ax.plot(near_path_x, near_path_y, ':', color='gold', linewidth=2.0, zorder=1)
        ax.plot(path_x, path_y, '-', color='gold', linewidth=2.0, zorder=1)
