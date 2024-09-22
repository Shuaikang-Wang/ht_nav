import yaml
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString

from ENV.line_to_squircle import LineToSquircle
from NF.utils import compute_squicle_length_ray


class RealWorld(object):
    def __init__(self, config_file='./CONFIG/simple_world.yaml'):
        self.config_file = config_file
        self.workspace = None  # contains a lot of polygonobs
        self.obstacles = None  # contains a lot of polygonobs
        self.config = None

        self.x_limits = None
        self.y_limits = None
        self.width = None
        self.height = None

        self.load_world_config()
        self.construct_world()
        self.get_workspace_size()

    def load_world_config(self):
        # with open(self.config_file, "r") as stream:
        with open(self.config_file, "rb") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def construct_world(self):
        self.obstacles = []
        # add obstacles by param
        all_obs = self.config['obstacles']
        if all_obs is None:
            pass
        else:
            for obs in all_obs:
                obs_type = obs['type']
                obs_vertices = []
                for vertex in obs['vertices']:
                    obs_vertices.append(np.array(vertex))
                polygon = PolygonObs(obs_type, obs_vertices)
                self.obstacles.append(polygon)

        self.workspace = []
        # add workspace by param
        all_ws = self.config['workspace']
        for ws in all_ws:
            ws_type = ws['type']
            ws_vertices = []
            for vertex in ws['vertices']:
                ws_vertices.append(np.array(vertex))
            polygon = PolygonObs(ws_type, ws_vertices)
            self.workspace.append(polygon)

    def get_workspace_size(self):
        ws = self.workspace[0]
        left_bottom_corner = ws.vertices[0]
        right_top_corner = ws.vertices[2]
        self.x_limits = [left_bottom_corner[0], right_top_corner[0]]
        self.y_limits = [left_bottom_corner[1], right_top_corner[1]]
        self.width = self.x_limits[1] - self.x_limits[0]
        self.height = self.y_limits[1] - self.y_limits[0]

    def check_point_in_free_space(self, q):
        for ws in self.workspace:
            if not ws.check_point_inside(q):
                return False
        for obs in self.obstacles:
            if obs.check_point_inside(q):
                return False
        return True

    def check_point_in_obs_free_space(self, q):
        for obs in self.obstacles:
            if obs.check_point_inside(q):
                return False
        return True

    def check_point_distance_of_obstacle(self, q, radius):  # 可变相将机器人膨胀
        for ws in self.workspace:
            if not ws.check_point_inside_inflation(q, radius):
                return False
        for obs in self.obstacles:
            if obs.check_point_inside_inflation(q, radius):
                return False
        return True


class InitWorld(RealWorld):
    def __init__(self, config_file='./CONFIG/simple_world.yaml'):
        super().__init__(config_file)


class Line(object):
    def __init__(self, endpoint):
        self.endpoint = endpoint


class PolygonObs(object):
    def __init__(self, type_, vertices):
        self.type = type_
        self.vertices = vertices
        self.sides = None

        self.construct_sides()

    def construct_sides(self):
        self.sides = []
        for index in range(len(self.vertices) - 1):
            end_points = [self.vertices[index], self.vertices[index + 1]]
            side = Line(end_points)
            self.sides.append(side)
        end_points = [self.vertices[-1], self.vertices[0]]
        side = Line(end_points)
        self.sides.append(side)

    def check_point_inside(self, q):
        vertex_list = [tuple(vertex) for vertex in self.vertices]
        polygon = Polygon(vertex_list)
        point = Point(q[0], q[1])
        return polygon.contains(point)

    def check_point_inside_inflation(self, q, radius):
        vertex_list = [tuple(vertex) for vertex in self.vertices]
        polygon = Polygon(vertex_list)
        point = Point(q[0], q[1])
        circle = point.buffer(radius)
        return polygon.intersects(circle)


class ForestWorld(object):
    def __init__(self, workspace, obstacles):
        self.workspace = workspace
        self.obstacles = obstacles


class Squircle(object):
    def __init__(self, type_, center, width, height, theta=0.0, s=0.99):
        self.type = type_
        self.center = center
        self.width = width
        self.height = height
        self.theta = theta
        self.s = s
        if self.type == 'obstacle':
            if self.s > 0.1:
                self.radius = 0.1 * min(width, height)
            else:
                self.radius = 0.1 * width
        else:
            self.radius = 2.0 * max(width, height)

    def potential(self, q: np.ndarray) -> float:
        theta = self.theta
        s = self.s
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width / 2, self.height / 2
        rotated_x = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta) + x_0
        rotated_y = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta) + y_0
        x, y = rotated_x, rotated_y
        if self.type == 'obstacle':
            if self.s > 0.1:
                return ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                            (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (1 - s ** 2) * (
                                                                a ** 2 + b ** 2)) ** 0.5) - (a ** 2 + b ** 2)
            else:
                return (((x - x_0) ** 2 + (y - y_0) ** 2 + - a **2)**2)**0.01
        else:
            return ((a ** 2 + b ** 2) - ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                                        (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (
                                                                               1 - s ** 2) * (
                                                                               a ** 2 + b ** 2)) ** 0.5))**0.5

    def compute_v(self, q: np.ndarray, beta: float) -> float:
        maxVal = (self.width / 2) ** 2 + (self.height / 2) ** 2
        if self.type == 'obstacle':
            if np.linalg.norm(q - self.center) < 1.0e-5:
                return self.radius * (1.0 + beta / maxVal) * 1e5
            else:
                return self.radius * (1.0 + beta / maxVal) / np.linalg.norm(q - self.center)
        else:
            if np.linalg.norm(q - self.center) < 1.0e-5:
                return self.radius * (1 - beta / maxVal) * 1.0e5
            else:
                return self.radius * (1 - beta / maxVal) / np.linalg.norm(q - self.center)

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
    
    def check_point_inside_limits(self, q, threshold=0.0):
        x_min = self.center[0] - self.width / 2 + threshold
        x_max = self.center[0] + self.width / 2 - threshold
        y_min = self.center[1] - self.height / 2 + threshold
        y_max = self.center[1] + self.height / 2 - threshold
        if x_min <= q[0] <= x_max and y_min <= q[1] <= y_max:
            return True
        else:
            return False

    def workspace_meshgrid(self, resolution=0.05, threshold=0.0):
        # 0.05
        x_min, x_max = self.x_limits()
        y_min, y_max = self.y_limits()
        x = np.arange(x_min - threshold, x_max + threshold, resolution)
        y = np.arange(y_min - threshold, y_max + threshold, resolution)
        xx, yy = np.meshgrid(x, y)
        return xx, yy


    def check_point_inside(self, q: np.ndarray, threshold=0.0):
        theta = self.theta
        s = self.s
        x, y, x_0, y_0, a, b = q[0], q[1], self.center[0], self.center[1], self.width / 2 + threshold, self.height / 2 + threshold
        rotated_x = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta) + x_0
        rotated_y = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta) + y_0
        x, y = rotated_x, rotated_y
        if self.type == 'obstacle':
            if self.s > 0.1:
                potential_point = ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                            (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (1 - s ** 2) * (
                                                                a ** 2 + b ** 2)) ** 0.5) - (a ** 2 + b ** 2)
            else:
                potential_point = (x - x_0) ** 2 + (y - y_0) ** 2 + - a **2
        else:
            potential_point =  (a ** 2 + b ** 2) - ((x - x_0) ** 2 + (y - y_0) ** 2 + (((x - x_0) ** 2 -
                                                                        (y - y_0) ** 2 + b ** 2 - a ** 2) ** 2 + (
                                                                               1 - s ** 2) * (
                                                                               a ** 2 + b ** 2)) ** 0.5)
        if potential_point <= 0.0:
            return True
        return False


class Star(object):
    def __init__(self, type_, squircle_list):
        self.type = type_
        self.squircle_list = squircle_list 
        self.width = None
        self.height = None
        self.center = None
        self.theta = None
        self.radius = None
        self.s = None
        
        self.construct_root()

    def construct_root(self):
        centeral_squircle = self.squircle_list[0]
        for squircle_i in self.squircle_list:
            if squircle_i.s > 0.5:
                centeral_squircle = squircle_i
                break
        self.width = centeral_squircle.width
        self.height = centeral_squircle.height
        self.center = centeral_squircle.center
        self.theta = centeral_squircle.theta
        self.s = centeral_squircle.s
        self.radius = centeral_squircle.radius

    def potential(self, q, threshold=0.1) -> float:
        # total_potential = 0.0
        # for k in range(len(self.squircle_list)):
        #     alpha_potential_k = 0.0
        #     ratio = (-1)^k
        #     if k == 0:
        #         for i in range(0, len(self.squircle_list)):
        #             alpha_potential_k += self.squircle_list[i].potential(q)
        #     else:
        #         for i in range(0, len(self.squircle_list)-k):
        #             term_i = 0.0
        #             for j in range(i+k, len(self.squircle_list)):
        #                 sum_beta_square = 0.0
        #                 for m in range(i, j+1):
        #                     if m < i+k or m == j:
        #                         sum_beta_square += self.squircle_list[m].potential(q)**2
        #                 term_i += sum_beta_square**(0.5)
        #             alpha_potential_k += term_i
            
        #     alpha_potential_k = ratio * alpha_potential_k
        #     total_potential += alpha_potential_k
        # print("total_potential", total_potential / len(self.squircle_list))
        # return total_potential / len(self.squircle_list)
        potential=1.0
        for squircle_i in self.squircle_list:
            potential *= squircle_i.potential(q)
        return potential

    def compute_v(self, q: np.ndarray, beta: float) -> float:
        maxVal = (self.width / 2) ** 2 + (self.height / 2) ** 2
        if np.linalg.norm(q - self.center) < 1.0e-3:
            return self.radius * (1.0 + beta / maxVal) * 1e3
        else:
            return self.radius * (1.0 + beta / maxVal) / np.linalg.norm(q - self.center)

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
    
    def check_point_inside_limits(self, q, threshold=0.0):
        x_min = self.center[0] - self.width / 2 + threshold
        x_max = self.center[0] + self.width / 2 - threshold
        y_min = self.center[1] - self.height / 2 + threshold
        y_max = self.center[1] + self.height / 2 - threshold
        if x_min <= q[0] <= x_max and y_min <= q[1] <= y_max:
            return True
        else:
            return False

    def workspace_meshgrid(self, resolution=0.01, threshold=0.0):
        # 0.05
        x_min, x_max = self.x_limits()
        y_min, y_max = self.y_limits()
        x = np.arange(x_min - threshold, x_max + threshold, resolution)
        y = np.arange(y_min - threshold, y_max + threshold, resolution)
        xx, yy = np.meshgrid(x, y)
        return xx, yy


    def check_point_inside(self, q: np.ndarray, threshold=0.0):
        for squircle_i in self.squircle_list:
            if squircle_i.check_point_inside(q, threshold):
                return True
        return False

