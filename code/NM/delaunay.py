import math

import numpy as np
import triangle as tr
import networkx as nx
import itertools

import matplotlib.pyplot as plt


class Triangle(object):
    def __init__(self, vertices):
        self.vertices = vertices
        self.segments = []
        self.mid_points = []

        self.init_segments()

    def init_segments(self):
        for vertex_pair in list(itertools.combinations(self.vertices, 2)):
            self.segments.append(vertex_pair)
            mid_point = (vertex_pair[0] + vertex_pair[1]) / 2
            self.mid_points.append(mid_point)


class DelaunayTriangulation(object):
    def __init__(self, vertices_set, segments_set, holes_set=None):
        """
        :param vertices_set:
        :param segments_set:
        :param holes_set:
        """
        self.vertices_set = vertices_set
        self.segments_set = segments_set
        self.holes_set = holes_set

        self.data = {}

        self.triangles = []

        self.generate_triangles()

        self.path = None

    def generate_triangles(self):
        vertices_set_reshape = np.zeros((len(self.vertices_set), 2))
        segments_set_reshape = np.zeros((len(self.segments_set), 2))
        for i, vertex_i in enumerate(self.vertices_set):
            vertices_set_reshape[i] = vertex_i
        for i, segment_i in enumerate(self.segments_set):
            segments_set_reshape[i] = segment_i

        if (self.holes_set is None) or (len(self.holes_set) == 0):
            data = {'vertices': vertices_set_reshape,
                    'segments': segments_set_reshape
                    }
        else:
            data = {'vertices': vertices_set_reshape,
                    'segments': segments_set_reshape,
                    'holes': self.holes_set
                    }

        triangulation = tr.triangulate(data, 'pc')

        # print("self.vertices_set", self.vertices_set)
        # print("self.segments_set", self.segments_set)
        # print("self.holes_set", self.holes_set)

        for index in triangulation['triangles']:
            triangle_i = Triangle([triangulation['vertices'][index[0]],
                                   triangulation['vertices'][index[1]],
                                   triangulation['vertices'][index[2]]])
            self.triangles.append(triangle_i)
        # print(self.triangles)

    @staticmethod
    def distance(point_1, point_2):
        return ((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2) ** 0.5

    @staticmethod
    def is_point_in_triangle(A, B, C, P):
        def cross_product(u, v):
            return u[0] * v[1] - u[1] * v[0]

        AP = [P[0] - A[0], P[1] - A[1]]
        BP = [P[0] - B[0], P[1] - B[1]]
        CP = [P[0] - C[0], P[1] - C[1]]

        cross1 = cross_product([B[0] - A[0], B[1] - A[1]], AP)
        cross2 = cross_product([C[0] - B[0], C[1] - B[1]], BP)
        cross3 = cross_product([A[0] - C[0], A[1] - C[1]], CP)

        # print("cross1", cross1)
        # print("cross2", cross2)
        # print("cross3", cross3)

        if (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0):
            return True
        else:
            return False

    def change_mid_to_center(self, current_triangle, mid_path, shared_edges):
        start = list(mid_path[0])
        start.append(np.arctan2(mid_path[1][1] - start[1], mid_path[1][0] - start[0]))
        goal = list(mid_path[-1])
        goal.append(np.arctan2(goal[1] - mid_path[-2][1], goal[0] - mid_path[-2][0]))
        del mid_path[0]
        del mid_path[-1]
        changed_path = [start]

        center = [0.0, 0.0, 0.0]
        for vertex in current_triangle:
            center[0] = center[0] + vertex[0] / 3
            center[1] = center[1] + vertex[1] / 3
        changed_path.append(center)
        current_start = center

        reach_goal_flag = False
        for i, path_i in enumerate(mid_path[:-1]):
            if reach_goal_flag:
                break
            # print("path_i", path_i)
            near_triangles = shared_edges[path_i]
            # print("near_triangles", near_triangles)
            for triangle_j in near_triangles:
                # print("triangle_j", triangle_j)
                if reach_goal_flag:
                    break
                if self.is_point_in_triangle(triangle_j[0], triangle_j[1], triangle_j[2], goal):
                    reach_goal_flag = True

                center = [0.0, 0.0, 0.0]
                for vertex in triangle_j:
                    center[0] = center[0] + vertex[0] / 3
                    center[1] = center[1] + vertex[1] / 3
                center[2] = np.arctan2(mid_path[i + 1][1] - center[1], mid_path[i + 1][0] - center[0])

                # print("current_start", current_start)
                # print("center", center)

                if math.sqrt((center[0] - current_start[0])**2 + (center[1] - current_start[1])**2) < 1e-5:
                    continue
                else:
                    changed_path.append(center)
                    current_start = center
                    break
        changed_path.append(goal)
        return changed_path

    def recursive_traversal(self, G, shared_edges, start_triangle, start_mid_point, goal_point):
        stack = [(start_triangle, start_mid_point)]

        while stack:
            current_triangle, current_mid_point = stack.pop()

            keys_to_remove = [key for key, value in shared_edges.items() if not value]
            for key_j in keys_to_remove:
                del shared_edges[key_j]

            is_goal_in_triangle = self.is_point_in_triangle(current_triangle[0], current_triangle[1],
                                                            current_triangle[2],
                                                            goal_point)
            if is_goal_in_triangle:
                G.add_node(goal_point)
                dis_start_to_goal = self.distance(current_mid_point, goal_point)
                G.add_edge(current_mid_point, goal_point, weight=dis_start_to_goal)
                continue

            keys_in_current_triangle = [key for key, value in shared_edges.items() if current_triangle in value]

            for key_i in keys_in_current_triangle:
                G.add_node(key_i)
                dis_node = self.distance(current_mid_point, key_i)
                G.add_edge(current_mid_point, key_i, weight=dis_node)
                shared_edges[key_i].remove(current_triangle)

                if shared_edges[key_i]:
                    for triangle_i in shared_edges[key_i]:
                        stack.append((triangle_i, key_i))

        return G

    # def recursive_traversal(self, G, shared_edges, current_triangle, current_mid_point, goal_point):
    #     print("current_triangle", current_triangle)
    #     is_goal_in_triangle = self.is_point_in_triangle(current_triangle[0], current_triangle[1], current_triangle[2],
    #                                                     goal_point)
    #
    #     if is_goal_in_triangle:
    #         G.add_node(goal_point)
    #         dis_start_to_goal = self.distance(current_mid_point, goal_point)
    #         G.add_edge(current_mid_point, goal_point, weight=dis_start_to_goal)
    #         return G
    #
    #     keys_to_remove = []
    #     for key_j in shared_edges.keys():
    #         if len(shared_edges[key_j]) == 0:
    #             keys_to_remove.append(key_j)
    #     for key_j in keys_to_remove:
    #         del shared_edges[key_j]
    #
    #     keys_in_current_triangle = [key for key, value in shared_edges.items() if current_triangle in value]
    #
    #     for key_i in keys_in_current_triangle:
    #         G.add_node(key_i)
    #         dis_node = self.distance(current_mid_point, key_i)
    #         G.add_edge(current_mid_point, key_i, weight=dis_node)
    #         shared_edges[key_i].remove(current_triangle)
    #
    #     for key_i in keys_in_current_triangle:
    #         if key_i not in shared_edges:
    #             continue
    #         if len(shared_edges[key_i]) == 0:
    #             continue
    #         for triangle_i in shared_edges[key_i]:
    #             G = self.recursive_traversal(G, shared_edges, triangle_i, key_i, goal_point)
    #
    #     return G

    def generate_navigation_map(self, start_point, goal_point):
        shared_edges = {}
        for triangle in self.triangles:
            A, B, C = triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]
            A = list(A)
            B = list(B)
            C = list(C)
            edges = [(A, B), (B, C), (C, A)]
            for edge in edges:
                mid_point = ((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2)
                if mid_point not in shared_edges:
                    shared_edges[mid_point] = []
                shared_edges[mid_point].append([A, B, C])

        keys_to_remove = []
        for key_i in shared_edges.keys():
            if len(shared_edges[key_i]) < 2:
                keys_to_remove.append(key_i)

        for key_i in keys_to_remove:
            del shared_edges[key_i]

        start_point = tuple(start_point[0:2])
        goal_point = tuple(goal_point[0:2])

        G = nx.Graph()
        G.add_node(start_point)

        start_triangle = None
        for key_i in shared_edges.keys():
            if start_triangle is not None:
                break
            for triangle_i in shared_edges[key_i]:
                A, B, C = triangle_i
                is_start_in_triangle = self.is_point_in_triangle(A, B, C, start_point)
                if is_start_in_triangle:
                    start_triangle = triangle_i
                    break

        current_triangle = start_triangle
        current_mid_point = start_point
        # print("current_triangle", current_triangle)
        import copy
        copy_edges = copy.deepcopy(shared_edges)
        # print("shared_edges", len(copy_edges))
        G = self.recursive_traversal(G, shared_edges, current_triangle, current_mid_point, goal_point)
        # print(G)
        # print("shared_edges", len(copy_edges))
        mid_path = nx.shortest_path(G, source=start_point, target=goal_point, weight='weight')
        center_path = self.change_mid_to_center(current_triangle, mid_path, copy_edges)
        shortest_path = center_path
        return shortest_path

    def plot_triangles(self, ax):
        for triangle in self.triangles:
            A, B, C = triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]
            A = list(A)
            B = list(B)
            C = list(C)
            triangle = [A, B, C]
            points = np.array(triangle + [triangle[0]])
            ax.plot(points[:, 0], points[:, 1], c='cyan', alpha=0.5, linewidth=1.0, zorder=1)

    def plot_path(self, ax, robot_pose):
        path_x = []
        path_y = []
        path_theta = []
        for path_i in self.path[2:-1]:
            ax.plot(path_i[0], path_i[1], '.k')
            path_x.append(path_i[0])
            path_y.append(path_i[1])
            path_theta.append(path_i[2])
        path_x.append(self.path[-1][0])
        path_y.append(self.path[-1][1])
        path_theta.append(self.path[-1][2])
        near_path_x = [robot_pose[0], self.path[2][0]]
        near_path_y = [robot_pose[1], self.path[2][1]]
        for i in range(len(path_x)):
            ax.quiver(path_x[i], path_y[i], np.cos(path_theta[i]), np.sin(path_theta[i]), units='xy', width=0.028,
                      headwidth=3.3, scale=1 / 0.23, color='gold')
        ax.plot(near_path_x, near_path_y, ':', color='gold', linewidth=2.0, zorder=1)
        ax.plot(path_x, path_y, '-', color='gold', linewidth=2.0, zorder=1)
