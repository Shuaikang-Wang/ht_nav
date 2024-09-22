import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev


class KBSpline(object):
    def __init__(self, s=0.001, k=3, num=100):
        self.s = s
        self.k = k
        self.num = num

    def traj_smoothing(self, points):
        self.num = int(len(points))
        
        x = points[:, 0]
        y = points[:, 1]
        points = np.vstack((x, y))
        tck, u = splprep(points, s=self.s, k=self.k)
        u_fine = np.linspace(0, 1, self.num)
        x_new, y_new = splev(u_fine, tck)
        return x_new, y_new


class ClusterSplit(object):
    def __init__(self, window_size=10, curvature_threshold=5):
        self.window_size = window_size
        self.curvature_threshold = curvature_threshold

    @staticmethod
    def smooth_points(points):
        points = np.array(points)
        b_spline = KBSpline()
        x_fine, y_fine = b_spline.traj_smoothing(points)
        return x_fine, y_fine

    @staticmethod
    def fit_circle(x, y):
        def residuals(c):
            Ri = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)
            return Ri - Ri.mean()

        x_m = np.mean(x)
        y_m = np.mean(y)
        R_m = np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2).mean()

        result = least_squares(residuals, [x_m, y_m, R_m])
        xc, yc = result.x[0], result.x[1]
        Ri = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        R = Ri.mean()
        return xc, yc, R

    def compute_curvature(self, x, y):
        _, _, R = self.fit_circle(x, y)
        curvature = 1 / R
        return curvature

    def detect_curvature_changes(self, points, window_size=5, curvature_threshold=0.01):
        num_points = int(len(points))

        x = points[:, 0]
        y = points[:, 1]

        half_window = window_size // 2

        curvatures = np.zeros(num_points)
        for i in range(num_points):
            start = max(0, i - half_window)
            end = min(num_points, i + half_window + 1)
            if end - start < 5:
                continue
            curvatures[i] = self.compute_curvature(x[start:end], y[start:end])
        curvature_diff = np.abs(curvatures)
        kappa = 0.99
        curvature_diff_no_zero = curvature_diff[curvature_diff > 0.0]
        count = np.sum(curvature_diff_no_zero > 5.0)
        if len(curvature_diff_no_zero) != 0:
            if count / len(curvature_diff_no_zero) > 0.6:
                kappa = 0.0
        split_indices = np.where(curvature_diff > curvature_threshold)[0] + 1
        return split_indices.tolist(), kappa

    def split_points_by_curvature(self, points):
        x_fine, y_fine = self.smooth_points(points)
        x_fine = np.array(x_fine).reshape(-1, 1)
        y_fine = np.array(y_fine).reshape(-1, 1)
        points = np.hstack((x_fine, y_fine))
        select_region = [1.6, 3.5, 0.6, 2.2]
        split_indices, kappa = self.detect_curvature_changes(points, self.window_size, self.curvature_threshold)
        if not (select_region[0] < points[0][0] < select_region[1] and select_region[2] < points[0][1] < select_region[3]):
            kappa = 0.99
        split_indices = [0] + split_indices + [len(points)]

        segments = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            if i == 0:
                segments.append(points[start_idx:end_idx])
            else:
                if abs(start_idx - end_idx) > 10:
                    segments.append(points[start_idx:end_idx])
                else:
                    segments[-1] = np.concatenate((segments[-1], points[start_idx:end_idx]))
        segments.append(kappa)
        return segments

    def split_all_cluster(self, cluster_points):
        all_cluster_segments = []
        for points in cluster_points:
            if len(points) < 8:
                continue
            x_fine, y_fine = self.smooth_points(points[:, :2])
            x_fine = np.array(x_fine).reshape(-1, 1)
            y_fine = np.array(y_fine).reshape(-1, 1)
            A = np.hstack((x_fine, y_fine))
            B = points[:, :2]
            distances = np.sqrt(((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2).sum(axis=2))
            closest_indices = np.argmin(distances, axis=1)
            smooth_points = np.hstack((x_fine, y_fine, points[closest_indices, 2:3], points[closest_indices, 3:4], points[closest_indices, 4:5]))
            if np.linalg.norm(np.array(smooth_points[0][:2]) - np.array(smooth_points[-1][:2])) < 0.1 and len(points) < 100:
                continue
            split_indices, kappa = self.detect_curvature_changes(smooth_points[:, :2], self.window_size, self.curvature_threshold)
            split_indices = [0] + split_indices + [len(smooth_points)]

            segments_with_kappa = []
            for i in range(len(split_indices) - 1):
                start_idx = split_indices[i]
                end_idx = split_indices[i + 1]
                if i == 0:
                    segments_with_kappa.append(smooth_points[start_idx:end_idx])
                else:
                    if abs(start_idx - end_idx) > 8:
                        segments_with_kappa.append(smooth_points[start_idx:end_idx])
                    else:
                        segments_with_kappa[-1] = np.concatenate((segments_with_kappa[-1], smooth_points[start_idx:end_idx]))
            if np.linalg.norm(np.array(segments_with_kappa[0][0][:2]) - np.array(segments_with_kappa[0][-1][:2])) < 0.1:
                del segments_with_kappa[0]
            if np.linalg.norm(np.array(segments_with_kappa[-1][0][:2]) - np.array(segments_with_kappa[-1][-1][:2])) < 0.1:
                del segments_with_kappa[-1]
            segments_with_kappa.append(kappa)
            all_cluster_segments.append(segments_with_kappa)
        return all_cluster_segments

    @staticmethod
    def remove_close_points(points, threshold):
        if len(points) == 0:
            return points

        dist_matrix = distance_matrix(points, points)
        close_points = np.where((dist_matrix < threshold) & (dist_matrix > 0))
        to_remove = set()
        for i, j in zip(*close_points):
            if i < j:
                to_remove.add(j)
        filtered_points = np.array([p for idx, p in enumerate(points) if idx not in to_remove])
        return filtered_points

