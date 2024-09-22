import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import box
import numpy as np
import math
import networkx as nx
from ENV.line_to_squircle import LineToSquircle
from ENV.geometry import Squircle, Star
from ENV.polygon_partition import PolygonPartition
from NM.polygon_extend import Building

from ENV.utils import distance


class SemanticWorld(object):
    def __init__(self):
        self.workspace = []
        self.obstacles = []


class TreeNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []


class ForestWorld(object):
    def __init__(self, workspace, obstacles):
        self.workspace = workspace
        self.obstacles = obstacles

    def check_point_in_free_space(self, q: np.ndarray, threshold=0.0):
        if self.workspace[0][0].check_point_inside(q, threshold):
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


class ConstructForest:
    def __init__(self, squircle_data):
        self.squircle_data = squircle_data
        self.squircles = []
        self.forest_world = ForestWorld([], [])
        self.ws_root = Squircle('workspace', np.array([3.5, 2.0]), 7.0, 4.0, 0.0, 0.9999)
        self.semantic_world = SemanticWorld()
        self.forest_root = None
        self.all_obs_graph = []
        self.all_ws_graph = []

        self.convert_squircle_data()
        self.construct_semantic_world()
        self.construct_forest_world()
        self.create_forest()
        self.node_pos = {self.forest_root: (0, 0)}

    def convert_squircle_data(self):
        for squircle_i in self.squircle_data:
            squircle = Squircle('obstacle', np.array(squircle_i[0]), squircle_i[1], squircle_i[2], squircle_i[3], squircle_i[4])
            self.squircles.append(squircle)

    @staticmethod
    def line_perpendicular(line1, line2, threshold=0.01):
        dx1 = line1[1][0] - line1[0][0]
        dy1 = line1[1][1] - line1[0][1]
        mag1 = (dx1 ** 2 + dy1 ** 2) ** 0.5

        dx2 = line2[1][0] - line2[0][0]
        dy2 = line2[1][1] - line2[0][1]
        mag2 = (dx2 ** 2 + dy2 ** 2) ** 0.5

        dot_product = dx1 * dx2 + dy1 * dy2
        cos_theta = dot_product / (mag1 * mag2)

        return abs(cos_theta) < threshold

    @staticmethod
    def intersect_ray(ray1, ray2):
        def cross_product(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        def subtract(v1, v2):
            return [v1[0] - v2[0], v1[1] - v2[1]]

        p1, v1 = ray1
        p2, v2 = ray2

        determinant = cross_product(v1, v2)

        if determinant == 0:
            return False

        t = cross_product(subtract(p2, p1), v2) / determinant
        u = cross_product(subtract(p2, p1), v1) / determinant

        if t >= 0 and u >= 0:
            return True
        else:
            return False

    @staticmethod
    def line_intersection(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if det == 0:
            return None

        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

        intersection_point = [intersection_x, intersection_y]
        return intersection_point

    @staticmethod
    def is_end_point(line, point):
        for point_i in line:
            dx = point_i[0] - point[0]
            dy = point_i[1] - point[1]
            if (dx ** 2 + dy ** 2) ** 0.5 < 1e-5:
                return True
        return False

    @staticmethod
    def point_on_line(line, point):
        start_point = line[0]
        end_point = line[1]
        segment_length_square = (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2

        dot_product = ((point[0] - start_point[0]) * (end_point[0] - start_point[0]) +
                       (point[1] - start_point[1]) * (end_point[1] - start_point[1]))

        if dot_product < 0 or dot_product > segment_length_square:
            return False

        dot_product = dot_product / segment_length_square

        distance_to_segment_square = (dot_product * (end_point[1] - start_point[1]) + start_point[1] - point[1]) ** 2 + \
                                     (dot_product * (end_point[0] - start_point[0]) + start_point[0] - point[0]) ** 2

        return distance_to_segment_square < 1e-5

    @staticmethod
    def check_ray_intersection(ray_1, ray_2):
        # ray1 = [start_point1, direction1]
        # ray2 = [start_point2, direction2]
        x1, y1 = ray_1[0]
        dx1, dy1 = ray_1[1]
        x2, y2 = ray_2[0]
        dx2, dy2 = ray_2[1]

        t1 = (dx2 * (y1 - y2) + dy2 * (x2 - x1)) / (dx1 * dy2 - dx2 * dy1)
        t2 = (dx1 * (y1 - y2) + dy1 * (x2 - x1)) / (dx1 * dy2 - dx2 * dy1)

        if t1 >= 0 and t2 >= 0:
            return True
        else:
            return False
        
    def merge_group_to_star(self, all_squircle_group):
        select_region = [1.6, 3.5, 0.6, 2.2]
        merged_squircle_group = []
        for group_i in all_squircle_group:
            require_merge = False
            for squircle in group_i:
                center = squircle.center
                if center[0] > select_region[0] and center[0] < select_region[1] and center[1] > select_region[2] and center[1] < select_region[3]:
                    require_merge = True
                    break
            if not require_merge:
                merged_squircle_group.append(group_i)
            else:
                merged_star = Star('obstacle', group_i)
                merged_squircle_group.append([merged_star])
        return merged_squircle_group

    def construct_semantic_world(self):
        all_squircle_group = self.class_squircle_in_group()
        # all_squircle_group = self.merge_group_to_star(all_squircle_group)
        for group_i in all_squircle_group:
            # print("111group_i", group_i)

            temp_group = group_i
            # print("111temp group", temp_group)
            ws = []
            obs = []
            is_ws = 0
            for j, squircle_j in enumerate(group_i):
                if self.check_squircle_ws_intersection(squircle_j):
                    ws = [self.ws_root, squircle_j]
                    del temp_group[j]
                    is_ws = 1
                    break
            if is_ws:
                iteration = 0
                while len(temp_group) > 0:
                    if iteration == 10:
                        ws = ws + temp_group
                        break
                    iteration += 1
                    if len(temp_group) == 1:
                        ws.append(temp_group[0])
                        del temp_group[0]
                    for k, temp_squircle_k in enumerate(temp_group):
                        if self.check_squircle_intersect(ws[-1], temp_squircle_k, threshold=0.1):
                            ws.append(temp_squircle_k)
                            del temp_group[k]
                self.semantic_world.workspace.append(ws)
            else:
                # temp_group = group_i
                # connect_num = [0 for _ in range(len(temp_group))]
                # for m, squircle_m in enumerate(temp_group[:-1]):
                #     for n, squircle_n in enumerate(temp_group[m + 1:]):
                #         n = n + m + 1
                #         if self.check_squircle_intersect(squircle_m, squircle_n, threshold=0.1):
                #             connect_num[m] += 1
                #             connect_num[n] += 1
                # root_index = 0
                # for k, num in enumerate(connect_num):
                #     if num == 1:
                #         root_index = k
                # obs.append(temp_group[root_index])
                # del temp_group[root_index]
                # iteration = 0
                # while len(temp_group) > 0:
                #     if iteration == 10:
                #         obs = obs + temp_group
                #         break
                #     iteration += 1
                #     if len(temp_group) == 1:
                #         obs.append(temp_group[0])
                #         del temp_group[0]
                #     for t, temp_squircle_t in enumerate(temp_group):
                #         if self.check_squircle_intersect(obs[-1], temp_squircle_t, threshold=0.1):
                #             obs.append(temp_squircle_t)
                #             del temp_group[t]
                # self.semantic_world.obstacles.append(obs)
    
                for squircle in group_i:
                    obs = [squircle]
                    self.semantic_world.obstacles.append(obs)


    def extend_ws_for_vis(self, polygon, threshold=0.1):
        ws_left = self.ws_root.center[0] - self.ws_root.width / 2
        ws_right = self.ws_root.center[0] + self.ws_root.width / 2
        ws_bottom = self.ws_root.center[1] - self.ws_root.height / 2
        ws_top = self.ws_root.center[1] + self.ws_root.height / 2
        # print(abs(line_start[0] - line_end[0]))
        extended_polygon = []
        extend_dis = 40.0
        for vertex in polygon:
            if vertex[1] < ws_bottom + threshold:
                new_vertex = (vertex[0], vertex[1] - extend_dis)
            elif vertex[1] > ws_top - threshold:
                new_vertex = (vertex[0], vertex[1] + extend_dis)
            elif vertex[0] < ws_left + threshold:
                new_vertex = (vertex[0] - extend_dis, vertex[1])
            elif vertex[0] > ws_right - threshold:
                new_vertex = (vertex[0] + extend_dis, vertex[1])
            else:
                new_vertex = vertex
            # print("new_vertex", new_vertex)
            extended_polygon.append(new_vertex)
        return extended_polygon

    def get_vis_rect_data(self, inflated_size=0.0):
        all_rects = []
        for ws_group in self.semantic_world.workspace:
            if len(ws_group) == 1:
                continue
            else:
                # if len(ws_group) == 3:
                #     print(ws_group[2].center, ws_group[2].width, ws_group[2].height)
                all_ws_group = ws_group[1:]

                center = all_ws_group[0].center
                width = all_ws_group[0].width
                height = all_ws_group[0].height
                theta = all_ws_group[0].theta
                ws_polygon_0 = [[(- width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                 (- width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]],
                                [(- width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                 (- width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                [(width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                 (width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                [(width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                 (width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]]
                                ]

                new_ws_polygon_0 = self.extend_ws_for_vis(ws_polygon_0, threshold=inflated_size)
                all_rects.append(new_ws_polygon_0)

                for ws_i in all_ws_group[1:]:
                    center = ws_i.center
                    width = ws_i.width
                    height = ws_i.height
                    theta = ws_i.theta
                    ws_polygon_i = [[(- width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                    (- width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]],
                                    [(- width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                    (- width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                    [(width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                    (width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                    [(width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                    (width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]]
                                    ]
                    all_rects.append(ws_polygon_i)

        for obs_group in self.semantic_world.obstacles:
            region = [3.5, 5.0, 1.8, 4.5]
            for obs_i in obs_group:
                center = obs_i.center
                width = obs_i.width
                height = obs_i.height
                theta = obs_i.theta
                if center[0] > region[0] and center[0] < region[1] and center[1] > region[2] and center[1] < region[3]:
                    width -= 0.0
                    height -= 0.0
                obs_polygon_i = [[(- width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                (- width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]],
                                [(- width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                (- width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                [(width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                (width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                [(width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                (width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]]
                                ]
                all_rects.append(obs_polygon_i)

        extend_rects = []
        for rect_i in all_rects:
            vertices_list = [list(vertex) for vertex in rect_i]
            # print("vertices", list(vertices_list))
            b = Building(np.array(vertices_list), inflated_size)
            extend_rects.append(b.expansion_anchors)

        if len(extend_rects) == 0:
            return []
        union_polygon = Polygon(extend_rects[0])
        for rect_i in extend_rects[1:]:
            rect_polygon_i = Polygon(rect_i)
            union_polygon = union_polygon.union(rect_polygon_i)

        polygon_union = []
        if type(union_polygon).__name__ == "Polygon":
            polygon_union.append(list(union_polygon.exterior.coords))
        else:
            for polygon_i in union_polygon.geoms:
                polygon_union.append(list(polygon_i.exterior.coords))

        all_polygon = []
        for polygon in polygon_union:
            new_polygon_union = self.remove_polygon_redundant_vertices(polygon)
            all_polygon.append(new_polygon_union)
        return all_polygon

    def get_vis_data(self):
        all_polygon_union = []
        for ws_group in self.semantic_world.workspace:
            if len(ws_group) == 1:
                continue
            else:
                # if len(ws_group) == 3:
                #     print(ws_group[2].center, ws_group[2].width, ws_group[2].height)
                all_ws_group = ws_group[1:]
                center = all_ws_group[0].center
                width = all_ws_group[0].width
                height = all_ws_group[0].height
                theta = all_ws_group[0].theta
                ws_polygon_0 = [[(- width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                 (- width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]],
                                [(- width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                 (- width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                [(width / 2) * np.cos(theta) - (height / 2) * np.sin(theta) + center[0],
                                 (width / 2) * np.sin(theta) + (height / 2) * np.cos(theta) + center[1]],
                                [(width / 2) * np.cos(theta) - (- height / 2) * np.sin(theta) + center[0],
                                 (width / 2) * np.sin(theta) + (- height / 2) * np.cos(theta) + center[1]]
                                ]

                if len(all_ws_group) == 1:

                    new_polygon_union = self.extend_ws_for_vis(ws_polygon_0)
                    # new_polygon_union = self.remove_polygon_redundant_vertices(new_polygon_union)
                    new_polygon_union = self.remove_polygon_redundant_vertices(new_polygon_union)
                    all_polygon_union.append(new_polygon_union)
                else:
                    union_polygon = Polygon(ws_polygon_0)
                    for ws_i in all_ws_group[1:]:
                        ws_polygon_i = [[ws_i.center[0] - ws_i.width / 2,
                                         ws_i.center[1] - ws_i.height / 2],
                                        [ws_i.center[0] - ws_i.width / 2,
                                         ws_i.center[1] + ws_i.height / 2],
                                        [ws_i.center[0] + ws_i.width / 2,
                                         ws_i.center[1] + ws_i.height / 2],
                                        [ws_i.center[0] + ws_i.width / 2,
                                         ws_i.center[1] - ws_i.height / 2]
                                        ]
                        ws_polygon_i = Polygon(ws_polygon_i)
                        union_polygon = union_polygon.union(ws_polygon_i)

                    polygon_union = []
                    if type(union_polygon).__name__ == "Polygon":
                        polygon_union.append(list(union_polygon.exterior.coords))
                    else:
                        for polygon_i in union_polygon.geoms:
                            polygon_union.append(list(polygon_i.exterior.coords))

                    # polygon_union = self.rectangles_union(all_ws_group)
                    # print("polygon_union", polygon_union)
                    for polygon in polygon_union:
                        new_polygon_union = self.extend_ws_for_vis(polygon)
                        new_polygon_union = self.remove_polygon_redundant_vertices(new_polygon_union)
                        all_polygon_union.append(new_polygon_union)

        for obs_group in self.semantic_world.obstacles:

            obs_polygon_0 = [[obs_group[0].center[0] - obs_group[0].width / 2,
                              obs_group[0].center[1] - obs_group[0].height / 2],
                             [obs_group[0].center[0] - obs_group[0].width / 2,
                              obs_group[0].center[1] + obs_group[0].height / 2],
                             [obs_group[0].center[0] + obs_group[0].width / 2,
                              obs_group[0].center[1] + obs_group[0].height / 2],
                             [obs_group[0].center[0] + obs_group[0].width / 2,
                              obs_group[0].center[1] - obs_group[0].height / 2]
                             ]

            if len(obs_polygon_0) == 1:
                # polygon_union = self.rectangles_union(obs_group)
                # print("polygon_union", polygon_union)
                obs_polygon_0 = self.remove_polygon_redundant_vertices(obs_polygon_0)
                all_polygon_union.append(obs_polygon_0)
            else:
                union_polygon = Polygon(obs_polygon_0)
                for obs_i in obs_group[1:]:
                    obs_polygon_i = [[obs_i.center[0] - obs_i.width / 2,
                                      obs_i.center[1] - obs_i.height / 2],
                                     [obs_i.center[0] - obs_i.width / 2,
                                      obs_i.center[1] + obs_i.height / 2],
                                     [obs_i.center[0] + obs_i.width / 2,
                                      obs_i.center[1] + obs_i.height / 2],
                                     [obs_i.center[0] + obs_i.width / 2,
                                      obs_i.center[1] - obs_i.height / 2]
                                     ]
                    obs_polygon_i = Polygon(obs_polygon_i)
                    union_polygon = union_polygon.union(obs_polygon_i)

                polygon_union = []
                if type(union_polygon).__name__ == "Polygon":
                    polygon_union.append(list(union_polygon.exterior.coords))
                else:
                    for polygon_i in union_polygon.geoms:
                        polygon_union.append(list(polygon_i.exterior.coords))

                # polygon_union = self.rectangles_union(obs_group)
                # print("polygon_union", polygon_union)
                for polygon in polygon_union:
                    polygon = self.remove_polygon_redundant_vertices(polygon)
                    all_polygon_union.append(polygon)

        return all_polygon_union

    @staticmethod
    def remove_polygon_redundant_vertices(polygon_vertices):
        new_vertices = []

        i = 0
        # print("polygon vertices", polygon_vertices)
        while i < len(polygon_vertices):
            new_vertices.append(polygon_vertices[i])
            if i == len(polygon_vertices) - 1:
                break
            for j in range(i + 1, len(polygon_vertices)):
                # print("j", j)
                if math.sqrt((polygon_vertices[i][0] - polygon_vertices[j][0]) ** 2 +
                             (polygon_vertices[i][1] - polygon_vertices[j][1]) ** 2) < 1e-2:
                    if j == len(polygon_vertices) - 1:
                        if math.sqrt((new_vertices[0][0] - new_vertices[-1][0]) ** 2 +
                                     (new_vertices[0][1] - new_vertices[-1][1]) ** 2) < 1e-2:
                            new_vertices = new_vertices[0:-1]
                        return new_vertices
                    continue
                else:
                    i = j
                    break
        if math.sqrt((new_vertices[0][0] - new_vertices[-1][0]) ** 2 +
                     (new_vertices[0][1] - new_vertices[-1][1]) ** 2) < 1e-2:
            new_vertices = new_vertices[0:-1]
        return new_vertices

    def get_delaunay_data(self):
        all_polygon_union = []
        all_holes = []
        for obs_group in self.semantic_world.obstacles:
            polygon_union = self.rectangles_union(obs_group)
            # print("polygon_union", polygon_union)
            all_polygon_union.append(polygon_union)
            all_holes.append([obs_group[-1].center[0], obs_group[-1].center[1]])

        all_ws_group = [self.semantic_world.workspace[0][0]]
        for ws_group in self.semantic_world.workspace:
            if len(ws_group) == 1:
                continue
            else:
                # if len(ws_group) == 3:
                #     print(ws_group[2].center, ws_group[2].width, ws_group[2].height)
                all_ws_group += ws_group[1:]
                for ws_obs_i in ws_group[1:]:
                    all_holes.append([ws_obs_i.center[0], ws_obs_i.center[1]])

        polygon_union = self.rectangles_complement(all_ws_group)
        # print("polygon_union", polygon_union)

        all_polygon_union.append(polygon_union)

        all_points = []
        all_lines = []

        point_index = 0
        for polygon_i in all_polygon_union:
            all_points += polygon_i
            for j in range(point_index, point_index + len(polygon_i) - 1):
                all_lines.append([j, j + 1])
            all_lines.append([j + 1, point_index])
            point_index += len(polygon_i)

        return all_points, all_lines, all_holes

    def construct_forest_world(self):
        self.forest_world.obstacles = self.semantic_world.obstacles
        if len(self.semantic_world.workspace) == 0:
            self.forest_world.workspace = [[self.ws_root]]
        else:
            self.forest_world.workspace = self.semantic_world.workspace

        # print("semantic_world.obstacles", self.semantic_world.obstacles)
        # print("semantic_world.workspace", self.semantic_world.workspace)
        # print("forest_world.obstacles", self.forest_world.obstacles)
        # print("forest_world.workspace", self.forest_world.workspace)

    @staticmethod
    def rectangles_union(obs_group):
        rectangles = []
        for squircle_i in obs_group:
            rectangles.append((squircle_i.center[0], squircle_i.center[1],
                               squircle_i.width, squircle_i.height))

        boxes = [box(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y, w, h in rectangles]

        union_polygon = unary_union(boxes)

        return list(union_polygon.exterior.coords)[:-1]

    @staticmethod
    def rectangles_complement(ws_group):
        ws = ws_group[0]
        ws_box = box(ws.center[0] - ws.width / 2, ws.center[1] - ws.height / 2,
                     ws.center[0] + ws.width / 2, ws.center[1] + ws.height / 2)

        small_rectangles = []
        for squircle_i in ws_group[1:]:
            small_rectangles.append((squircle_i.center[0], squircle_i.center[1], squircle_i.width, squircle_i.height))

        small_boxes = [box(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y, w, h in small_rectangles]

        small_union = unary_union(small_boxes)

        result_polygon = ws_box.difference(small_union)

        # print("result_polygon", result_polygon)

        if isinstance(result_polygon, Polygon):
            return list(result_polygon.exterior.coords)[:-1]
        else:
            return None

    def merge_group_to_polygon(self, group, if_ws):
        if not if_ws:
            obs = []
            # print("group", len(group))
            # for squircle in group:
            #     print("squircle", squircle.center, squircle.width, squircle.height)
            temp_group = group
            root_index = 0
            obs.append(temp_group[root_index])
            del temp_group[root_index]
            while len(temp_group) > 0:
                if len(temp_group) == 1:
                    obs.append(temp_group[0])
                    del temp_group[0]
                for t, temp_squircle_t in enumerate(temp_group):
                    if self.check_squircle_intersect(obs[-1], temp_squircle_t, threshold=0.1):
                        obs.append(temp_squircle_t)
                        del temp_group[t]
            group = obs
            # print("after group", group)
        else:
            # print("before group", group)
            # for squircle_i in group:
            #     print("squircle_i", squircle_i.center, squircle_i.width, squircle_i.height)
            obs = []
            temp_group = group
            connect_num = [0 for _ in range(len(temp_group))]
            for m, squircle_m in enumerate(temp_group[:-1]):
                # print("m", m)
                for n, squircle_n in enumerate(temp_group[m + 1:]):
                    n = n + m + 1
                    # print("n", n)
                    if self.check_squircle_intersect(squircle_m, squircle_n, threshold=0.1):
                        # print("mn", m, n)
                        # print("squircle_m", squircle_m.center, squircle_m.width, squircle_m.height)
                        # print("squircle_n", squircle_n.center, squircle_n.width, squircle_n.height)
                        # print("verctor", squircle_n.vector)
                        connect_num[m] += 1
                        connect_num[n] += 1
            # print("connect_num", connect_num)
            root_index = 0
            for k, num in enumerate(connect_num):
                if num == 1:
                    root_index = k
            obs.append(temp_group[root_index])
            del temp_group[root_index]
            while len(temp_group) > 0:
                if len(temp_group) == 1:
                    obs.append(temp_group[0])
                    del temp_group[0]
                for t, temp_squircle_t in enumerate(temp_group):
                    if self.check_squircle_intersect(obs[-1], temp_squircle_t, threshold=0.1):
                        obs.append(temp_squircle_t)
                        del temp_group[t]
            group = obs
        # print("group", group)
        # for squircle_i in group:
        #     print("squircle_i", squircle_i.center, squircle_i.width, squircle_i.height)
        all_outer_lines = []
        if if_ws:
            ws = self.ws_root
            for obs in group:
                # print("outer_line", obs.outer_line)
                all_outer_lines.append(obs.outer_line)

            threshold = 0.15
            ws_left = self.ws_root.center[0] - self.ws_root.width / 2 + threshold
            ws_right = self.ws_root.center[0] + self.ws_root.width / 2 - threshold
            ws_bottom = self.ws_root.center[1] - self.ws_root.height / 2 + threshold
            ws_top = self.ws_root.center[1] + self.ws_root.height / 2 - threshold

            all_outer_points = []
            # print("all_outer_lines", all_outer_lines)
            for line in all_outer_lines:
                for point in line:
                    if not ws.check_point_inside_limits(point, threshold=threshold):
                        all_outer_points.append(point)
            # print("all_outer_points", all_outer_points)

            if all_outer_points[0][0] < ws_left or all_outer_points[1][0] < ws_left:
                extreme_value = min(all_outer_points[0][0], all_outer_points[1][0])
                all_outer_points[0][0] = extreme_value - threshold
                all_outer_points[1][0] = extreme_value - threshold
            if all_outer_points[0][0] > ws_right or all_outer_points[1][0] > ws_right:
                extreme_value = max(all_outer_points[0][0], all_outer_points[1][0])
                all_outer_points[0][0] = extreme_value + threshold
                all_outer_points[1][0] = extreme_value + threshold
            if all_outer_points[0][1] < ws_bottom or all_outer_points[1][1] < ws_bottom:
                extreme_value = min(all_outer_points[0][1], all_outer_points[1][1])
                all_outer_points[0][1] = extreme_value - threshold
                all_outer_points[1][1] = extreme_value - threshold
            if all_outer_points[0][1] > ws_top or all_outer_points[1][1] > ws_top:
                extreme_value = max(all_outer_points[0][1], all_outer_points[1][1])
                all_outer_points[0][1] = extreme_value + threshold
                all_outer_points[1][1] = extreme_value + threshold

            # print("all_outer_points", all_outer_points)
            all_outer_lines.append(all_outer_points)
        else:
            for obs in group:
                all_outer_lines.append(obs.outer_line)

        polygon = []

        # new_all_outer_lines = []
        # for i in range(len(all_outer_lines)):
        #     line_i = all_outer_lines[i]
        #     if math.sqrt((line_i[0][0] - line_i[1][0])**2 + (line_i[0][1] - line_i[1][1])**2) > 0.1:
        #         new_all_outer_lines.append(line_i)
        # all_outer_lines = new_all_outer_lines

        # print("all_outer_lines", all_outer_lines)
        for i in range(len(all_outer_lines)):
            line_1 = all_outer_lines[i % (len(all_outer_lines))]
            line_2 = all_outer_lines[(i + 1) % (len(all_outer_lines))]
            # print("line_1", line_1)
            # print("line_2", line_2)

            intersection = self.line_intersection(line_1, line_2)

            ws_left = self.ws_root.center[0] - self.ws_root.width / 2
            ws_right = self.ws_root.center[0] + self.ws_root.width / 2
            ws_bottom = self.ws_root.center[1] - self.ws_root.height / 2
            ws_top = self.ws_root.center[1] + self.ws_root.height / 2

            threshold = 10
            if intersection[0] < ws_left - threshold or intersection[0] > ws_right + threshold or \
                    intersection[1] < ws_bottom - threshold or intersection[1] > ws_top + threshold:
                continue
            polygon.append(intersection)

        # polygon = [all_outer_lines[0][0], all_outer_lines[0][1]]
        # for line in all_outer_lines[1:-1]:
        #     start_point = line[0]
        #     end_point = line[1]
        #     if distance(start_point, polygon[-1]) < 0.05:
        #         polygon.append(end_point)
        #     else:
        #         polygon.append(start_point)
        # print("polygon", polygon)
        return polygon

    @staticmethod
    def polygon_partition(polygon):
        poly_part = PolygonPartition(polygon)
        squircle_list = poly_part.polygon_partition()
        return squircle_list

    def check_closed_loop_in_group(self, group):
        group_i = group
        is_ws = 0
        for squircle_j in group_i:
            if self.check_squircle_ws_intersection(squircle_j):
                is_ws = 1
                break
        if is_ws:
            num_of_intersection = 0
            for m, squircle_m in enumerate(group_i[:-1]):
                for n, squircle_n in enumerate(group_i[m + 1:]):
                    if self.check_squircle_intersect(squircle_m, squircle_n):
                        num_of_intersection += 1
            for squircle_m in group_i:
                if self.check_squircle_ws_intersection(squircle_m):
                    num_of_intersection += 1
            # print("num_of_intersection", num_of_intersection)
            if num_of_intersection == len(group_i) + 1:
                return True, is_ws
        else:
            num_of_intersection = 0
            for m, squircle_m in enumerate(group_i[:-1]):
                for n, squircle_n in enumerate(group_i[m + 1:]):
                    if self.check_squircle_intersect(squircle_m, squircle_n):
                        num_of_intersection += 1
            if num_of_intersection == len(group_i):
                return True, is_ws
        return False, is_ws

    def find_squircle_in_groups(self, squircle, all_groups):
        for group_i in all_groups:
            for squircle_j in group_i:
                if self.check_squircle_intersect(squircle, squircle_j):
                    return group_i
        return None

    def class_squircle_in_group(self):
        all_groups = []

        for squircle_i in self.squircles:
            # print("squircle_i data", squircle_i.center, squircle_i.width, squircle_i.height)
            # print("squircle_i", squircle_i)
            # print("all_groups", all_groups)
            connected_group = self.find_squircle_in_groups(squircle_i, all_groups)
            if connected_group is not None:
                connected_group.append(squircle_i)
            else:
                all_groups.append([squircle_i])

        # Merge groups that share common squircle
        merged_groups = []
        all_groups_pre = all_groups
        all_groups_suf = all_groups
        all_merge_index = []
        for i, group_i in enumerate(all_groups_pre):
            merged_groups_i = group_i
            if i in all_merge_index:
                continue
            for j, group_j in enumerate(all_groups_suf[i + 1:]):
                j = j + i + 1
                if j in all_merge_index:
                    continue
                if any(self.check_squircle_intersect(squircle_i, squircle_j)
                       for squircle_i in group_i for squircle_j in group_j):
                    merged_groups_i = merged_groups_i + group_j
                    all_merge_index.append(j)
            unique_merged_groups_i = []
            [unique_merged_groups_i.append(obj) for obj in merged_groups_i if obj not in unique_merged_groups_i]
            merged_groups.append(unique_merged_groups_i)
        return merged_groups

    def extend_line_to_ws(self, line_start, line_end, extension):
        ws_left = self.ws_root.center[0] - self.ws_root.width / 2
        ws_right = self.ws_root.center[0] + self.ws_root.width / 2
        ws_bottom = self.ws_root.center[1] - self.ws_root.height / 2
        ws_top = self.ws_root.center[1] + self.ws_root.height / 2
        # print(abs(line_start[0] - line_end[0]))
        dis = math.sqrt((line_start[0] - line_end[0])**2 + (line_start[1] - line_end[1])**2)
        if abs(line_start[0] - line_end[0]) < 1e-2:
            extended_line_start_y = line_start[1] + extension * (line_start[1] - line_end[1]) / dis
            extended_line_end_y = line_end[1] + extension * (line_end[1] - line_start[1]) / dis
            # print("extended_line_start_y", extended_line_start_y)
            # print("extended_line_end_y", extended_line_end_y)
            if extended_line_start_y < ws_bottom:
                line_start = [line_start[0], ws_bottom]
            if extended_line_start_y > ws_top:
                line_start = [line_start[0], ws_top]
            if extended_line_end_y < ws_bottom:
                line_end = [line_end[0], ws_bottom]
            if extended_line_end_y > ws_top:
                line_end = [line_end[0], ws_top]
        elif abs(line_start[1] - line_end[1]) < 1e-2:
            extended_line_start_x = line_start[0] + extension * (line_start[0] - line_end[0]) / dis
            extended_line_end_x = line_end[0] + extension * (line_end[0] - line_start[0]) / dis
            if extended_line_start_x < ws_left:
                line_start = [ws_left, line_start[1]]
            if extended_line_start_x > ws_right:
                line_start = [ws_right, line_start[1]]
            if extended_line_end_x < ws_left:
                line_end = [ws_left, line_end[1]]
            if extended_line_end_x > ws_right:
                line_end = [ws_right, line_end[1]]
        else:
            extended_line_start_y = line_start[1] + extension * (line_start[1] - line_end[1]) / dis
            extended_line_end_y = line_end[1] + extension * (line_end[1] - line_start[1]) / dis
            if extended_line_start_y < ws_bottom:
                line_start = [line_start[0], ws_bottom]
            if extended_line_start_y > ws_top:
                line_start = [line_start[0], ws_top]
            if extended_line_end_y < ws_bottom:
                line_end = [line_end[0], ws_bottom]
            if extended_line_end_y > ws_top:
                line_end = [line_end[0], ws_top]

            extended_line_start_x = line_start[0] + extension * (line_start[0] - line_end[0]) / dis
            extended_line_end_x = line_end[0] + extension * (line_end[0] - line_start[0]) / dis
            if extended_line_start_x < ws_left:
                line_start = [ws_left, line_start[1]]
            if extended_line_start_x > ws_right:
                line_start = [ws_right, line_start[1]]
            if extended_line_end_x < ws_left:
                line_end = [ws_left, line_end[1]]
            if extended_line_end_x > ws_right:
                line_end = [ws_right, line_end[1]]
        return line_start, line_end

    @staticmethod
    def check_squircle_intersect(squircle_1, squircle_2, threshold=0.1):
        center_x_1 = squircle_1.center[0]
        center_y_1 = squircle_1.center[1]
        theta_1 = (squircle_1.theta % (2 * math.pi) + 2 * math.pi) % (2 * math.pi)
        if abs(theta_1 - np.pi / 2) < np.pi / 4 or abs(theta_1 - 3 * np.pi / 2) < np.pi / 4:
            width_1 = squircle_1.height + 2 * threshold
            height_1 = squircle_1.width + 2 * threshold
        else:
            width_1 = squircle_1.width + 2 * threshold
            height_1 = squircle_1.height + 2 * threshold

        center_x_2 = squircle_2.center[0]
        center_y_2 = squircle_2.center[1]
        theta_2 = (squircle_2.theta % (2 * math.pi) + 2 * math.pi) % (2 * math.pi)
        if abs(theta_2 - np.pi / 2) < np.pi / 4 or abs(theta_2 - 3 * np.pi / 2) < np.pi / 4:
            width_2 = squircle_2.height + 2 * threshold
            height_2 = squircle_2.width + 2 * threshold
        else:
            width_2 = squircle_2.width + 2 * threshold
            height_2 = squircle_2.height + 2 * threshold

        left = max(center_x_1 - width_1 / 2, center_x_2 - width_2 / 2)
        right = min(center_x_1 + width_1 / 2, center_x_2 + width_2 / 2)
        top = max(center_y_1 - height_1 / 2, center_y_2 - height_2 / 2)
        bottom = min(center_y_1 + height_1 / 2, center_y_2 + height_2 / 2)
        if left < right and top < bottom:
            return True
        else:
            return False

    def check_squircle_ws_intersection(self, squircle, threshold=-0.1):
        ws = self.ws_root

        ws_center_x = ws.center[0]
        ws_center_y = ws.center[1]
        ws_width = ws.width + 2 * threshold
        ws_height = ws.height + 2 * threshold

        ws_left = ws_center_x - ws_width / 2
        ws_right = ws_center_x + ws_width / 2
        ws_bottom = ws_center_y - ws_height / 2
        ws_top = ws_center_y + ws_height / 2

        # print("ws_left", ws_left)
        # print("ws_right", ws_right)
        # print("ws_bottom", ws_bottom)
        # print("ws_top", ws_top)

        squircle_center_x = squircle.center[0]
        squircle_center_y = squircle.center[1]

        theta = (squircle.theta % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)
        if abs(theta - np.pi / 2) < np.pi / 4 or abs(theta - 3 * np.pi / 2) < np.pi / 4:
            squircle_width = squircle.height
            squircle_height = squircle.width
        else:
            squircle_width = squircle.width
            squircle_height = squircle.height

        squircle_left = squircle_center_x - squircle_width / 2
        squircle_right = squircle_center_x + squircle_width / 2
        squircle_bottom = squircle_center_y - squircle_height / 2
        squircle_top = squircle_center_y + squircle_height / 2

        # print("squircle_left", squircle_left)
        # print("squircle_right", squircle_right)
        # print("squircle_bottom", squircle_bottom)
        # print("squircle_top", squircle_top)

        if squircle_left < ws_left or \
                squircle_right > ws_right or \
                squircle_bottom < ws_bottom or \
                squircle_top > ws_top:
            return True
        else:
            return False

    def build_graph(self):
        if len(self.semantic_world.workspace) > 0:
            for ws in self.semantic_world.workspace:
                G = nx.Graph()
                n = len(ws[1:])
                if n == 1:
                    G.add_node(n - 1)
                else:
                    for i in range(n):
                        for j in range(i + 1, n):
                            G.add_edge(i, j)
                self.all_ws_graph.append(G)
        if len(self.semantic_world.obstacles) > 0:
            for obs in self.semantic_world.obstacles:
                G = nx.Graph()
                n = len(obs)
                if n == 1:
                    G.add_node(n - 1)
                else:
                    for i in range(n):
                        for j in range(i + 1, n):
                            G.add_edge(i, j)
                self.all_obs_graph.append(G)

    def find_root(self, G):
        max_depth = 0
        root = 0
        # print(len(G.nodes))
        for i in range(len(G.nodes)):
            T = self.build_tree(G, i)
            depth = self.tree_depth(T)
            if depth > max_depth:
                max_depth = depth
                root = i
        return root

    @staticmethod
    def build_tree(G, root):
        T = nx.dfs_tree(G, source=root)
        return T

    @staticmethod
    def tree_depth(T):
        depths = nx.single_source_shortest_path_length(T, source=list(T.nodes())[0])
        return max(depths.values())

    def calculate_subtree_width(self, node):
        if node is None:
            return 0
        num_children = len(node.children)
        if num_children == 0:
            return 1
        width = 0
        for child in node.children:
            width += self.calculate_subtree_width(child)
        return width

    def layout_tree(self, node, pos_x, pos_y, spacing, depth):
        if node is None:
            return
        subtree_widths = [self.calculate_subtree_width(child) for child in node.children]
        total_width = max(sum(subtree_widths), 1)
        x_start = pos_x - (total_width * spacing) / 2

        for i, child in enumerate(node.children):
            child_width = subtree_widths[i]
            x_child = x_start + (spacing * child_width) / 2
            y_child = pos_y - 200
            self.node_pos[child] = (x_child, y_child)
            self.layout_tree(child, x_child, y_child, spacing, depth + 1)
            x_start += spacing * child_width

    @staticmethod
    def bezier_curve(start, end, height_factor=0.3):
        x_start, y_start = start
        x_end, y_end = end
        x_mid = (x_start + 3 * x_end) / 5
        y_mid = (y_start + y_end) / 2
        x_ctrl1 = (x_mid + x_start) / 2
        y_ctrl1 = y_start + height_factor * (y_mid - y_start)
        x_ctrl2 = (x_mid + x_end) / 2
        y_ctrl2 = y_end - height_factor * (y_end - y_mid)
        return [(x_start, y_start), (x_ctrl1, y_ctrl1), (x_ctrl2, y_ctrl2), (x_end, y_end)]

    def create_forest(self):
        self.build_graph()
        root = TreeNode('Forest')
        ws = TreeNode(" W ", root)
        root.children.append(ws)
        for i, G in enumerate(self.all_obs_graph):
            best_root = self.find_root(G)
            T = self.build_tree(G, best_root)
            tree_edges = T.edges
            all_node = [root]
            if not tree_edges:
                tree = TreeNode("R", root)
                root.children.append(tree)
                all_node.append(tree)
            else:
                for j in range(len(tree_edges)):
                    if j == 0:
                        tree = TreeNode("R", root)
                        root.children.append(tree)
                        all_node.append(tree)
                    else:
                        tree = TreeNode("L", all_node[-1])
                        all_node[-1].children.append(tree)
                        all_node.append(tree)
                tree = TreeNode("L", all_node[-1])
                all_node[-1].children.append(tree)
                all_node.append(tree)
        for i, G in enumerate(self.all_ws_graph):
            best_root = self.find_root(G)
            T = self.build_tree(G, best_root)
            tree_edges = T.edges
            all_node = [ws]
            for j in range(len(tree_edges)):
                tree = TreeNode("L", all_node[-1])
                all_node[-1].children.append(tree)
                all_node.append(tree)
            tree = TreeNode("L", all_node[-1])
            all_node[-1].children.append(tree)
            all_node.append(tree)
        self.forest_root = root
        # print(self.forest_root)
        # print("self.forest_world", self.forest_world)

    def has_cycle(self, graph, node, visited, parent=None):
        visited[node] = True

        for neighbor in graph[node]:
            if not visited[neighbor]:
                if self.has_cycle(graph, neighbor, visited, node):
                    return True
            elif neighbor != parent:
                return True

        return False

    def has_cycle_in_tree(self, graph):
        visited = {node: False for node in graph}

        for node in graph:
            if not visited[node]:
                if self.has_cycle(graph, node, visited):
                    return True
        return False
