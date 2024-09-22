import copy
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix


class SklearnCluster(object):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.unique_labels = None
        self.labels = None

    def cluster(self, points):
        points_xy = points[:, :2]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_xy)
        self.labels = db.labels_
        self.unique_labels = set(self.labels)
        cluster_points = []
        for k in range(len(self.unique_labels)):
            if k == -1:
                continue
            class_member_mask = (self.labels == k)
            point = points[class_member_mask]
            if len(point) < 10:
                continue
            cluster_points.append(point)
        return cluster_points

    @staticmethod
    def find_nearest_path_with_matrix(dist_matrix, start_index):
        n = len(dist_matrix)
        dist_matrix = copy.deepcopy(dist_matrix)
        path = [start_index]
        total_distance = 0.0

        current_index = start_index

        for _ in range(1, n - 1):
            distances = dist_matrix[current_index, :]
            next_index = np.argmin(distances)
            total_distance += distances[next_index]
            path.append(next_index)
            dist_matrix[current_index, :] = np.inf
            dist_matrix[:, current_index] = np.inf
            current_index = next_index

        return total_distance, path

    def find_index_with_min_total_distance(self, dist_matrix):
        min_total_distance = float('inf')
        best_start_index = -1
        best_path = []

        for i in range(len(dist_matrix)):
            total_distance, path = self.find_nearest_path_with_matrix(dist_matrix, i)
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_start_index = i
                best_path = path

        return best_start_index, best_path

    def sort_cluster_points(self, cluster_points):
        if len(cluster_points[0]) == 0:
            return cluster_points
        sorted_cluster_points = []
        for cluster_i in cluster_points:
            if len(cluster_i) == 0:
                continue
            else:
                dist_matrix = distance_matrix(cluster_i[:, :2], cluster_i[:, :2])
                for i in range(len(dist_matrix)):
                    dist_matrix[i, i] = np.inf
                _, best_path = self.find_index_with_min_total_distance(dist_matrix)
                sorted_cluster_i = cluster_i[best_path]
                sorted_cluster_points.append(sorted_cluster_i)
        return sorted_cluster_points

    @staticmethod
    def draw_results(ax, cluster_points, COLORS):
        for points, col in zip(cluster_points, COLORS):
            xy = points
            ax.plot(xy[0, 0], xy[0, 1], 'o', markerfacecolor='b', markeredgecolor='b', markersize=6,
                    zorder=35)
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=4, zorder=35)

