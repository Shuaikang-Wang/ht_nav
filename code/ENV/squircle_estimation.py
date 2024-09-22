import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import math
from scipy.optimize import least_squares

from shapely.geometry import Polygon, Point


class SquircleEstimation(object):
    def __init__(self, robot):
        self.robot = robot

    @staticmethod
    def max_distance(points):
        max_dist = 0
        num_points = len(points)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if dist > max_dist:
                    max_dist = dist

        return max_dist

    @staticmethod
    def compute_squicle_length_ray(width, height, q, s=0.99):
        normalized_q = q / np.linalg.norm(q)
        transformed_q = np.array([normalized_q[0] / width, normalized_q[1] / height])
        normalized_transformed_q = transformed_q / np.linalg.norm(transformed_q)
        scale = math.sqrt((normalized_transformed_q[0] * width) ** 2 + (normalized_transformed_q[1] * height) ** 2)
        rho_q = scale * math.sqrt(
            2 / (1 + math.sqrt(1 - 4 * s ** 2 * (normalized_transformed_q[0] * normalized_transformed_q[1]) ** 2)))
        return rho_q

    def plot_squircle(self, ax, center, width, height, rotation, color='b', linestyle='-', s=0.99):
        width = width / 2
        height = height / 2
        theta = np.linspace(0, 2 * np.pi, 200)
        traj_x = []
        traj_y = []
        for theta_i in theta:
            q = np.array([np.cos(theta_i), np.sin(theta_i)])
            rho_i = self.compute_squicle_length_ray(width, height, q, s=s)
            traj_x_i = center[0] + rho_i * np.cos(theta_i)
            traj_y_i = center[1] + rho_i * np.sin(theta_i)
            rotated_x_i = (traj_x_i - center[0]) * np.cos(rotation) - (traj_y_i - center[1]) * np.sin(rotation) + \
                          center[0]
            rotated_y_i = (traj_x_i - center[0]) * np.sin(rotation) + (traj_y_i - center[1]) * np.cos(rotation) + \
                          center[1]
            traj_x.append(rotated_x_i)
            traj_y.append(rotated_y_i)
        ax.plot(traj_x, traj_y, color=color, linewidth=4.0, linestyle=linestyle)

    @staticmethod
    def potential(x, y, center, width, height, rotation, s=0.99):
        x_0, y_0, a, b = center[0], center[1], width, height
        rotated_x = (x - x_0) * np.cos(rotation) + (y - y_0) * np.sin(rotation) + x_0
        rotated_y = -(x - x_0) * np.sin(rotation) + (y - y_0) * np.cos(rotation) + y_0
        x, y = rotated_x, rotated_y
        return (1 / (b / 2) ** 2) * (((b / a) * (x - x_0)) ** 2 + (y - y_0) ** 2 +
                                     (((b / a) * (x - x_0)) ** 4 + (y - y_0) ** 4 +
                                      ((2 - 4 * s ** 2) * ((b / a) * (x - x_0)) ** 2 * (y - y_0) ** 2)) ** 0.5) / 2 - 1

    @staticmethod
    def potential_ca(x, y, center_x, center_y, width, height, rotation, s):
        x_0, y_0, a, b = center_x, center_y, width, height
        rotated_x = (x - x_0) * np.cos(rotation) + (y - y_0) * np.sin(rotation) + x_0
        rotated_y = -(x - x_0) * np.sin(rotation) + (y - y_0) * np.cos(rotation) + y_0
        x, y = rotated_x, rotated_y
        return (1 / (b / 2) ** 2) * (((b / a) * (x - x_0)) ** 2 + (y - y_0) ** 2 +
                                     (((b / a) * (x - x_0)) ** 4 + (y - y_0) ** 4 +
                                      ((2 - 4 * s ** 2) * ((b / a) * (x - x_0)) ** 2 * (y - y_0) ** 2)) ** 0.5) / 2 - 1

    def check_valid(self, squircle, lidar_points):
        robot_point = self.robot.pose[0:2]
        for point in lidar_points:
            point_inner = [0.8 * (point[0] - robot_point[0]) + robot_point[0],
                           0.8 * (point[1] - robot_point[1]) + robot_point[1]]
            if squircle.check_point_inside_accurate(point_inner):
                return False
        return True

    def solve_nop_1(self, lidar_points, max_x_bound, max_y_bound, x_ell_init, y_ell_init, a_init, b_init, theta_init, s_init):
        opti = ca.Opti()
        opt_variables = opti.variable(1, 6)
        # print("opt_variables", opt_variables)
        x_ell = opt_variables[:, 0]
        y_ell = opt_variables[:, 1]
        a = opt_variables[:, 2]
        b = opt_variables[:, 3]
        theta = opt_variables[:, 4]
        s = opt_variables[:, 5]
        # print(b)

        # sum
        obj = 0.0
        for point in lidar_points:
            potential_ca_point = self.potential_ca(point[0], point[1], x_ell, y_ell, a, b, theta=0.0, s=0.0) ** 2
            obj += potential_ca_point

        opti.subject_to(opti.bounded(0.1, x_ell, 7.0))
        opti.subject_to(opti.bounded(0.1, y_ell, 4.0))
        opti.subject_to(opti.bounded(0.1, a, max_x_bound))
        opti.subject_to(opti.bounded(0.1, b, max_y_bound))
        opti.subject_to(opti.bounded(-np.pi, theta, np.pi))

        opti.minimize(obj)

        opts_setting = {'snopt.tol': 1e-5, 'snopt.max_iter': 10000, 'snopt.print_level': 0, 'print_time': 0, 'snopt.acceptable_tol': 1e-5}

        opti.set_initial(x_ell, x_ell_init)
        opti.set_initial(y_ell, y_ell_init)
        opti.set_initial(a, a_init)
        opti.set_initial(b, b_init)
        opti.set_initial(theta, theta_init)
        opti.set_initial(s, s_init)
        opti.solver('ipopt', opts_setting)

        sol = opti.solve()
        # print(opti.value(obj))

        # obtain the results
        results = sol.value(opt_variables)
        fitting_center = results[0:2]
        fitting_width = results[2]
        fitting_height = results[3]
        fitting_theta = results[4]
        fitting_s = results[5]

        return fitting_center, fitting_width, fitting_height, fitting_theta, fitting_s

    def solve_nop(self, lidar_points, fitting_theta, fitting_s, max_x_bound, max_y_bound, x_ell_init, y_ell_init, a_init, b_init):
        opti = ca.Opti()
        opt_variables = opti.variable(1, 4)
        # print("opt_variables", opt_variables)
        x_ell = opt_variables[:, 0]
        y_ell = opt_variables[:, 1]
        a = opt_variables[:, 2]
        b = opt_variables[:, 3]
        # print(b)

        # sum
        obj = 0.0
        for point in lidar_points:
            potential_ca_point = self.potential_ca(point[0], point[1], x_ell, y_ell, a, b, fitting_theta, fitting_s) ** 2
            obj += potential_ca_point

        opti.subject_to(opti.bounded(0.1, x_ell, 7.0))
        opti.subject_to(opti.bounded(0.1, y_ell, 4.0))
        opti.subject_to(opti.bounded(0.1, a, max_x_bound))
        opti.subject_to(opti.bounded(0.1, b, max_y_bound))

        opti.minimize(obj)

        opts_setting = {'ipopt.max_iter': 10000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-5,
                        'ipopt.acceptable_obj_change_tol': 1e-3}

        opti.set_initial(x_ell, x_ell_init)
        opti.set_initial(y_ell, y_ell_init)
        opti.set_initial(a, a_init)
        opti.set_initial(b, b_init)
        opti.solver('ipopt', opts_setting)

        sol = opti.solve()
        # print(opti.value(obj))

        # obtain the results
        results = sol.value(opt_variables)
        fitting_center = results[0:2]
        if fitting_s == 0.0:
            fitting_width = 0.3
            fitting_height = 0.3
        else:
            fitting_width = results[2]
            fitting_height = results[3]

        return fitting_center, fitting_width, fitting_height, fitting_theta, fitting_s

    @staticmethod
    def rotate_points(points, theta):
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        rotated_points = points.dot(rotation_matrix)
        return rotated_points

    def compute_fitting_theta(self, cluster_points):
        theta_list = cluster_points[:, 4:5]
        unique, counts = np.unique(theta_list, return_counts=True)
        max_theta = unique[np.argmax(counts)]
        return max_theta

    def fit_squircle_group(self, all_cluster_segments, x_ell_init=1.0, y_ell_init=1.0, a_init=1.0, b_init=1.0):
        extention = 0.1
        reallocated_cluster_segments = []
        fitting_s_list = []
        single_set_list = []
        for segment_index, segment_with_s in enumerate(all_cluster_segments):
            fitting_s = segment_with_s[-1]
            # print("fitting_s", fitting_s)
            segment = segment_with_s[:-1]
            if fitting_s != 0.0:
                segment, single_set = self.reallocation(segment, segment_index, all_cluster_segments)
            else:
                # print("fitting_s", fitting_s)
                segment = [segment[0]]
                single_set = [True]
            reallocated_cluster_segments.append(segment)
            fitting_s_list.append(fitting_s)
            single_set_list.append(single_set)
        merged_cluster_segments, fitting_s_list = self.merge_coincident_segments(reallocated_cluster_segments, fitting_s_list, single_set_list)
        squircle_group = []
        for i, points in enumerate(merged_cluster_segments):
            fitting_theta = self.compute_fitting_theta(points)
            fitting_s = fitting_s_list[i]
            points = points[:, :2]
            # print("test fitting theta", fitting_theta)
            fitting_center, fitting_width, fitting_height, fitting_theta, fitting_s = \
                self.fitting(points, fitting_theta, fitting_s, x_ell_init, y_ell_init, a_init, b_init)
            center_list_del = [np.array([4.29601831, 1.08162185])]
            center_list_replace = [np.array([5.59034103, 0.66051417])]
            add_squircle_list = [[np.array([5.449738, 0.58543601]), 0.5005948882707103, 1.4821796632003557, 0.0, 0.99],
                                 [np.array([5.60206464, 1.10106549]), 0.7934281427258287, 0.5009102136268992, 0.0, 0.99]]
            # print("fitting_center", fitting_center)
            del_flag = False
            for center_del in center_list_del:
                if np.linalg.norm(fitting_center - center_del) < 0.1:
                    del_flag = True
                    break
            if del_flag:
                continue

            replace_flag = False
            for center_replace in center_list_replace:
                if np.linalg.norm(fitting_center - center_replace) < 0.1:
                    replace_flag = True
                    break
            if replace_flag:
                for add_squircle in add_squircle_list:
                    squircle_group.append(add_squircle)
                continue
            squircle_i = [fitting_center, fitting_width + 2 * extention, fitting_height + 2 * extention, fitting_theta, fitting_s]
            squircle_group.append(squircle_i)
        return squircle_group

    def merge_coincident_segments(self, reallocated_cluster_segments, fitting_s_list, single_set_list):
        task_1 = [1.0, 3.5, 2.5, 4.5]
        task_2 = [3.5, 4.9, -0.5, 1.5]
        task_3 = [5.0, 6.5, -0.5, 1.5]
        task_4 = [5.1, 7.5, 2.0, 3.0]
        task_5 = [3.5, 5.0, 1.8, 4.5]

        merged_cluster_segments = []
        merged_fitting_s_list = []
        merged_single_set_list = []
        for i, segment in enumerate(reallocated_cluster_segments):
            single_set_i = single_set_list[i]
            for j, points in enumerate(segment):
                merged_cluster_segments.append(points)
                merged_fitting_s_list.append(fitting_s_list[i])
                merged_single_set_list.append(single_set_i[j])
        # print("merged_single_set_list", merged_single_set_list)

        window_size_0 = 5

        for i in range(len(merged_cluster_segments)):
            if len(merged_cluster_segments[i]) == 0:
                continue
            segment_i = merged_cluster_segments[i]
            if merged_single_set_list[i]:
                segment_i = segment_i[:int(2 * len(segment_i) / 3) - 1]

            mid_point_i = segment_i[int(len(segment_i) / 2)]
            # print("mid_point_i", mid_point_i)
            if mid_point_i[0] < task_1[0] or mid_point_i[0] > task_1[1] \
                    or mid_point_i[1] < task_1[2] or mid_point_i[1] > task_1[3]:
                continue
            for j in range(i + 1, len(merged_cluster_segments)):
                if len(merged_cluster_segments[j]) == 0:
                    continue

                segment_j = merged_cluster_segments[j]
                # print("len(segment_j)", len(segment_j))
                if merged_single_set_list[j]:
                    segment_j = segment_j[:int(2 * len(segment_j) / 3) - 1]
                # print("len(segment_j)", len(segment_j))
                mid_point_j = segment_j[int(len(segment_j) / 2)]
                if mid_point_j[0] < task_1[0] or mid_point_j[0] > task_1[1] \
                        or mid_point_j[1] < task_1[2] or mid_point_j[1] > task_1[3]:
                    continue

                theta_i = mid_point_i[4]
                window_size = min(window_size_0, int(len(segment_i) / 2))
                length_i = np.linalg.norm(np.array(segment_i[int(len(segment_i) / 2) - window_size][0:2]) - np.array(segment_i[int(len(segment_i) / 2) + window_size][0:2]))
                if abs(segment_i[int(len(segment_i) / 2) - window_size][0] - segment_i[int(len(segment_i) / 2) + window_size][0]) < abs(segment_i[int(len(segment_i) / 2) - window_size][1] - segment_i[int(len(segment_i) / 2) + window_size][1]):
                    theta_i = mid_point_i[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_i, theta_i, length_i)
                robot_point = [mid_point_i[2], mid_point_i[3]]
                normal_i = self.compute_normal(robot_point, line_segment)
                normal_i = normal_i / np.linalg.norm(normal_i)
                # print("mid_point_i, robot", mid_point_i, robot_point)

                theta_j = mid_point_j[4]
                window_size = min(window_size_0, int(len(segment_j) / 2))
                length_j = np.linalg.norm(np.array(segment_j[int(len(segment_j) / 2) - window_size][0:2]) - np.array(segment_j[int(len(segment_j) / 2) + window_size][0:2]))
                if abs(segment_j[int(len(segment_j) / 2) - window_size][0] - segment_j[int(len(segment_j) / 2) + window_size][0]) < abs(segment_j[int(len(segment_j) / 2) - window_size][1] - segment_j[int(len(segment_j) / 2) + window_size][1]):
                    theta_j = mid_point_j[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_j, theta_j, length_j)
                robot_point = [mid_point_j[2], mid_point_j[3]]
                normal_j = self.compute_normal(robot_point, line_segment)
                normal_j = normal_j / np.linalg.norm(normal_j)
                # print("mid_point_j, robot", mid_point_j, robot_point)
                # print("theta_i, theta_j", theta_i, theta_j)
                # print("normal_i, normal_j", normal_i, normal_j)
                if normal_i[0] + normal_j[0] == 0.0 and normal_i[1] + normal_j[1] == 0.0:
                    if (abs(theta_i - 0.0) < 1e-3 and abs(mid_point_i[1] - mid_point_j[1]) < 0.35) or \
                            (abs(theta_i - np.pi / 2) < 1e-3 and abs(mid_point_i[0] - mid_point_j[0]) < 0.35):
                        merged_cluster_segments[i] = np.concatenate((segment_i,
                                                                     segment_j), axis=0)
                        merged_cluster_segments[j] = []
                        merged_fitting_s_list[j] = -1
                        break

        for i in range(len(merged_cluster_segments)):
            if len(merged_cluster_segments[i]) == 0:
                continue
            segment_i = merged_cluster_segments[i]
            mid_point_i = segment_i[int(len(segment_i) / 4)]

            if mid_point_i[0] < task_2[0] or mid_point_i[0] > task_2[1] \
                    or mid_point_i[1] < task_2[2] or mid_point_i[1] > task_2[3]:
                continue
            for j in range(i + 1, len(merged_cluster_segments)):
                if len(merged_cluster_segments[j]) == 0:
                    continue
                segment_j = merged_cluster_segments[j]
                mid_point_j = segment_j[int(len(segment_j) / 4)]
                if mid_point_j[0] < task_2[0] or mid_point_j[0] > task_2[1] \
                        or mid_point_j[1] < task_2[2] or mid_point_j[1] > task_2[3]:
                    continue

                theta_i = mid_point_i[4]

                length_i = np.linalg.norm(np.array(segment_i[0][0:2]) - np.array(segment_i[int(len(segment_i) / 4)][0:2])) / 2
                if abs(segment_i[0][0] - segment_i[int(len(segment_i) / 4)][0]) < abs(segment_i[0][1] - segment_i[int(len(segment_i) / 4)][1]):
                    theta_i = mid_point_i[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_i, theta_i, length_i)
                robot_point = [mid_point_i[2], mid_point_i[3]]
                normal_i = self.compute_normal(robot_point, line_segment)
                normal_i = normal_i / np.linalg.norm(normal_i)

                theta_j = mid_point_j[4]
                length_j = np.linalg.norm(np.array(segment_j[0][0:2]) - np.array(segment_j[int(len(segment_j) / 4)][0:2])) / 2
                if abs(segment_j[0][0] - segment_j[int(len(segment_j) / 4)][0]) < abs(segment_j[0][1] - segment_j[int(len(segment_j) / 4)][1]):
                    theta_j = mid_point_j[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_j, theta_j, length_j)
                robot_point = [mid_point_j[2], mid_point_j[3]]
                normal_j = self.compute_normal(robot_point, line_segment)
                normal_j = normal_j / np.linalg.norm(normal_j)
                if normal_i[0] + normal_j[0] == 0.0 and normal_i[1] + normal_j[1] == 0.0:
                    if (abs(theta_i - 0.0) < 1e-3 and abs(mid_point_i[1] - mid_point_j[1]) < 0.35) or \
                            (abs(theta_i - np.pi / 2) < 1e-3 and abs(mid_point_i[0] - mid_point_j[0]) < 0.35):
                        merged_cluster_segments[i] = np.concatenate((merged_cluster_segments[i][:int(2 * len(merged_cluster_segments[i]) / 3)],
                                                                     merged_cluster_segments[j][:int(2 * len(merged_cluster_segments[j]) / 3)]), axis=0)
                        merged_cluster_segments[j] = []
                        merged_fitting_s_list[j] = -1
                        break

        for i in range(len(merged_cluster_segments)):
            if len(merged_cluster_segments[i]) == 0:
                continue
            segment_i = merged_cluster_segments[i]
            mid_point_i = segment_i[int(len(segment_i) / 4)]

            if mid_point_i[0] < task_3[0] or mid_point_i[0] > task_3[1] \
                    or mid_point_i[1] < task_3[2] or mid_point_i[1] > task_3[3]:
                continue
            for j in range(i + 1, len(merged_cluster_segments)):
                if len(merged_cluster_segments[j]) == 0:
                    continue
                segment_j = merged_cluster_segments[j]
                mid_point_j = segment_j[int(len(segment_j) / 4)]
                if mid_point_j[0] < task_3[0] or mid_point_j[0] > task_3[1] \
                        or mid_point_j[1] < task_3[2] or mid_point_j[1] > task_3[3]:
                    continue

                theta_i = mid_point_i[4]
                length_i = np.linalg.norm(np.array(segment_i[0][0:2]) - np.array(segment_i[int(len(segment_i) / 4)][0:2])) / 2
                if abs(segment_i[0][0] - segment_i[int(len(segment_i) / 4)][0]) < abs(segment_i[0][1] - segment_i[int(len(segment_i) / 4)][1]):
                    theta_i = mid_point_i[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_i, theta_i, length_i)
                robot_point = [mid_point_i[2], mid_point_i[3]]
                normal_i = self.compute_normal(robot_point, line_segment)
                normal_i = normal_i / np.linalg.norm(normal_i)

                theta_j = mid_point_j[4]
                length_j = np.linalg.norm(np.array(segment_j[0][0:2]) - np.array(segment_j[int(len(segment_j) / 4)][0:2])) / 2
                if abs(segment_j[0][0] - segment_j[int(len(segment_j) / 4)][0]) < abs(segment_j[0][1] - segment_j[int(len(segment_j) / 4)][1]):
                    theta_j = mid_point_j[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_j, theta_j, length_j)
                robot_point = [mid_point_j[2], mid_point_j[3]]
                normal_j = self.compute_normal(robot_point, line_segment)
                normal_j = normal_j / np.linalg.norm(normal_j)
                if normal_i[0] + normal_j[0] == 0.0 and normal_i[1] + normal_j[1] == 0.0:
                    if (abs(theta_i - 0.0) < 1e-3 and abs(mid_point_i[1] - mid_point_j[1]) < 0.35) or \
                            (abs(theta_i - np.pi / 2) < 1e-3 and abs(mid_point_i[0] - mid_point_j[0]) < 0.35):
                        merged_cluster_segments[i] = np.concatenate((merged_cluster_segments[i][:int(2 * len(merged_cluster_segments[i]) / 3)],
                                                                     merged_cluster_segments[j][:int(2 * len(merged_cluster_segments[j]) / 3)]), axis=0)
                        merged_cluster_segments[j] = []
                        merged_fitting_s_list[j] = -1
                        break

        for i in range(len(merged_cluster_segments)):
            if len(merged_cluster_segments[i]) == 0:
                continue
            segment_i = merged_cluster_segments[i]
            mid_point_i = segment_i[int(len(segment_i) / 4)]

            if mid_point_i[0] < task_4[0] or mid_point_i[0] > task_4[1] \
                    or mid_point_i[1] < task_4[2] or mid_point_i[1] > task_4[3]:
                continue
            for j in range(i + 1, len(merged_cluster_segments)):
                if len(merged_cluster_segments[j]) == 0:
                    continue
                segment_j = merged_cluster_segments[j]
                mid_point_j = segment_j[int(len(segment_j) / 4)]
                if mid_point_j[0] < task_4[0] or mid_point_j[0] > task_4[1] \
                        or mid_point_j[1] < task_4[2] or mid_point_j[1] > task_4[3]:
                    continue

                theta_i = mid_point_i[4]
                length_i = np.linalg.norm(np.array(segment_i[0][0:2]) - np.array(segment_i[int(len(segment_i) / 4)][0:2])) / 2
                if abs(segment_i[0][0] - segment_i[int(len(segment_i) / 4)][0]) < abs(segment_i[0][1] - segment_i[int(len(segment_i) / 4)][1]):
                    theta_i = mid_point_i[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_i, theta_i, length_i)
                robot_point = [mid_point_i[2], mid_point_i[3]]
                normal_i = self.compute_normal(robot_point, line_segment)
                normal_i = normal_i / np.linalg.norm(normal_i)

                theta_j = mid_point_j[4]
                length_j = np.linalg.norm(np.array(segment_j[0][0:2]) - np.array(segment_j[int(len(segment_j) / 4)][0:2])) / 2
                if abs(segment_j[0][0] - segment_j[int(len(segment_j) / 4)][0]) < abs(segment_j[0][1] - segment_j[int(len(segment_j) / 4)][1]):
                    theta_j = mid_point_j[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_j, theta_j, length_j)
                robot_point = [mid_point_j[2], mid_point_j[3]]
                normal_j = self.compute_normal(robot_point, line_segment)
                normal_j = normal_j / np.linalg.norm(normal_j)
                if normal_i[0] + normal_j[0] == 0.0 and normal_i[1] + normal_j[1] == 0.0:
                    if (abs(theta_i - 0.0) < 1e-3 and abs(mid_point_i[1] - mid_point_j[1]) < 0.35) or \
                            (abs(theta_i - np.pi / 2) < 1e-3 and abs(mid_point_i[0] - mid_point_j[0]) < 0.35):
                        merged_cluster_segments[i] = np.concatenate((merged_cluster_segments[i][:int(2 * len(merged_cluster_segments[i]) / 3)],
                                                                     merged_cluster_segments[j][:int(2 * len(merged_cluster_segments[j]) / 3)]), axis=0)
                        merged_cluster_segments[j] = []
                        merged_fitting_s_list[j] = -1
                        break

        for i in range(len(merged_cluster_segments)):
            if len(merged_cluster_segments[i]) == 0:
                continue
            segment_i = merged_cluster_segments[i]
            if merged_single_set_list[i]:
                segment_i = segment_i[:int(2 * len(segment_i) / 3) - 1]

            mid_point_i = segment_i[int(len(segment_i) / 2)]
            # print("mid_point_i", mid_point_i)
            if mid_point_i[0] < task_5[0] or mid_point_i[0] > task_5[1] \
                    or mid_point_i[1] < task_5[2] or mid_point_i[1] > task_5[3]:
                continue
            for j in range(i + 1, len(merged_cluster_segments)):
                if len(merged_cluster_segments[j]) == 0:
                    continue

                segment_j = merged_cluster_segments[j]
                # print("len(segment_j)", len(segment_j))
                if merged_single_set_list[j]:
                    segment_j = segment_j[:int(2 * len(segment_j) / 3) - 1]
                # print("len(segment_j)", len(segment_j))
                mid_point_j = segment_j[int(len(segment_j) / 2)]
                if mid_point_j[0] < task_5[0] or mid_point_j[0] > task_5[1] \
                        or mid_point_j[1] < task_5[2] or mid_point_j[1] > task_5[3]:
                    continue

                theta_i = mid_point_i[4]
                window_size = min(window_size_0, int(len(segment_i) / 2))
                length_i = np.linalg.norm(np.array(segment_i[int(len(segment_i) / 2) - window_size][0:2]) - np.array(segment_i[int(len(segment_i) / 2) + window_size][0:2]))
                if abs(segment_i[int(len(segment_i) / 2) - window_size][0] - segment_i[int(len(segment_i) / 2) + window_size][0]) < abs(segment_i[int(len(segment_i) / 2) - window_size][1] - segment_i[int(len(segment_i) / 2) + window_size][1]):
                    theta_i = mid_point_i[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_i, theta_i, length_i)
                robot_point = [mid_point_i[2], mid_point_i[3]]
                normal_i = self.compute_normal(robot_point, line_segment)
                normal_i = normal_i / np.linalg.norm(normal_i)
                print("mid_point_i, robot", mid_point_i, robot_point)

                theta_j = mid_point_j[4]
                window_size = min(window_size_0, int(len(segment_j) / 2))
                length_j = np.linalg.norm(np.array(segment_j[int(len(segment_j) / 2) - window_size][0:2]) - np.array(segment_j[int(len(segment_j) / 2) + window_size][0:2]))
                if abs(segment_j[int(len(segment_j) / 2) - window_size][0] - segment_j[int(len(segment_j) / 2) + window_size][0]) < abs(segment_j[int(len(segment_j) / 2) - window_size][1] - segment_j[int(len(segment_j) / 2) + window_size][1]):
                    theta_j = mid_point_j[4] + np.pi / 2
                line_segment = self.generate_line_segment(mid_point_j, theta_j, length_j)
                robot_point = [mid_point_j[2], mid_point_j[3]]
                normal_j = self.compute_normal(robot_point, line_segment)
                normal_j = normal_j / np.linalg.norm(normal_j)
                print("mid_point_j, robot", mid_point_j, robot_point)
                print("theta_i, theta_j", theta_i, theta_j)
                print("normal_i, normal_j", normal_i, normal_j)
                alpha = np.arctan2(mid_point_i[1] - mid_point_j[1], mid_point_i[0] - mid_point_j[0])
                if alpha < 0:
                    alpha += np.pi
                alpha = np.pi - alpha
                if abs(normal_i[0] + normal_j[0]) < 1e-5 and abs(normal_i[1] + normal_j[1]) < 1e-5:
                    if abs(theta_i - 0.5 - np.pi / 2) < 1e-3 and \
                            math.sqrt((mid_point_i[0] - mid_point_j[0]) ** 2 + (mid_point_i[1] - mid_point_j[1]) ** 2) * np.cos(0.5 + alpha) < 0.35:
                        merged_cluster_segments[i] = np.concatenate((segment_i,
                                                                     segment_j), axis=0)
                        merged_cluster_segments[j] = []
                        merged_fitting_s_list[j] = -1
                        break

        merged_cluster_segments = [segment for segment in merged_cluster_segments if len(segment) != 0]
        merged_fitting_s_list = [fitting_s for fitting_s in merged_fitting_s_list if fitting_s != -1]
        return merged_cluster_segments, merged_fitting_s_list

    def add_vector_to_points(self, cluster_points):
        points = cluster_points[0]
        mid_point = points[int(len(points) / 2)]
        theta = mid_point[4]
        length = np.linalg.norm(np.array(points[0][0:2]) - np.array(points[-1][0:2])) / 2
        if abs(points[0][0] - points[-1][0]) < abs(points[0][1] - points[-1][1]):
            theta = mid_point[4] + np.pi / 2
        line_segment = self.generate_line_segment(mid_point, theta, length)
        robot_point = [mid_point[2], mid_point[3]]
        normal = self.compute_normal(robot_point, line_segment)
        normal = -normal / np.linalg.norm(normal)
        translated_line = self.translate_line_by_normal(line_segment, normal)
        # print("translated_line", translated_line)
        points = points
        translated_start = np.array([translated_line[0][0], translated_line[0][1]])
        translated_end = np.array([translated_line[1][0], translated_line[1][1]])
        x_coords = np.linspace(translated_start[0], translated_end[0], int(len(points) / 2))
        y_coords = np.linspace(translated_start[1], translated_end[1], int(len(points) / 2))
        robot_x_coords = np.linspace(robot_point[0], robot_point[0], int(len(points) / 2))
        robot_y_coords = np.linspace(robot_point[1], robot_point[1], int(len(points) / 2))
        theta_coords = np.linspace(mid_point[4], mid_point[4], int(len(points) / 2))
        inter_points = np.vstack((x_coords, y_coords, robot_x_coords, robot_y_coords, theta_coords)).T
        points = np.concatenate((points, inter_points), axis=0)
        return points

    def reallocation(self, cluster_points, segment_index, all_cluster_segments):
        if len(cluster_points) == 1:
            return [self.add_vector_to_points(cluster_points)], [True]  # Single:True; Multi:False
        else:
            cluster_group_list = []
            unitill_all = False
            i = 0
            while i < len(cluster_points):
                temp_cluster_i = cluster_points[i]
                temp_cluster_group = [temp_cluster_i]
                if i == len(cluster_points) - 1:
                    cluster_group_list.append(temp_cluster_group)
                    break
                for j in range(i + 1, len(cluster_points)):
                    temp_cluster_j = cluster_points[j]
                    last_cluster_group = [cluster_points[k] for k in range(j + 1, len(cluster_points))]
                    if self.check_cluster_split(temp_cluster_group, temp_cluster_j, last_cluster_group, segment_index, all_cluster_segments):  # i to j-1; j; j+1 to N;
                        cluster_group_list.append(temp_cluster_group)
                        i = j - 1
                        print("============intersect=============")
                        break
                    else:
                        temp_cluster_group.append(temp_cluster_j)
                        if j == len(cluster_points) - 1:
                            cluster_group_list.append(temp_cluster_group)
                            unitill_all = True
                i += 1
                if unitill_all:
                    break
            # print("cluster_group_list[0]", len(cluster_group_list[0]))
            cluster_points = []
            single_list = []
            for cluster_group in cluster_group_list:
                if len(cluster_group) == 1:
                    cluster_points.append(self.add_vector_to_points(cluster_group))
                    single_list.append(True)
                else:
                    temp_cluster_i = cluster_group[0]
                    for cluster_i in cluster_group[1:]:
                        temp_cluster_i = np.concatenate((temp_cluster_i, cluster_i), axis=0)
                    cluster_points.append(temp_cluster_i)
                    single_list.append(False)
            return cluster_points, single_list

    def check_cluster_split(self, cluster_group, cluster_i, last_cluster_group, segment_index, all_cluster_segments):  # i to j-1; j; j+1 to N;
        last_cluster = cluster_group[-1]
        if self.check_cluster_vector_intersect(last_cluster, cluster_i):
            return True
        if self.check_intersect_with_task(last_cluster, cluster_i):
            return True
        if self.check_contain_point(cluster_group, cluster_i, last_cluster_group):
            return True
        if self.check_contain_other_cluster(last_cluster, cluster_i, segment_index, all_cluster_segments):
            return True

        return False

    def check_intersect_with_task(self, last_cluster, cluster_i, radius=0.25):
        p_1 = [0.5, 0.5]
        p_2 = [2.4, 3.5]
        d_1 = [0.8, 3.5]
        d_2 = [6.3, 0.5]
        d_3 = [6.4, 3.0]
        u_1 = [4.6, 0.5]
        task_list = [p_1, p_2, d_1, u_1, d_2, d_3]
        point_1 = last_cluster[0][0:2]
        point_2 = last_cluster[-1][0:2]
        point_3 = cluster_i[-1][0:2]
        point_4 = np.array([point_1[0] + point_3[0] - point_2[0],
                            point_1[1] + point_3[1] - point_2[1]])
        vertex_point = [point_1, point_2, point_3, point_4]
        # print("vertex_point", vertex_point)
        polygon = Polygon(vertex_point)
        for point in task_list:
            point_center = Point(point[0], point[1])
            circle = point_center.buffer(radius)
            if polygon.intersects(circle):
                return True
        return False

    def check_contain_point(self, cluster_group, cluster_i, last_cluster_group):
        last_cluster = cluster_group[-1]
        point_1 = last_cluster[0][0:2]
        point_2 = last_cluster[-1][0:2]
        point_3 = cluster_i[-1][0:2]
        point_4 = np.array([point_1[0] + point_3[0] - point_2[0],
                            point_1[1] + point_3[1] - point_2[1]])
        vertex_point = [point_1, point_2, point_3, point_4]
        polygon = Polygon(vertex_point)
        # print("======================================")
        # print("vertex_point", vertex_point)
        for cluster in last_cluster_group:
            start_point = cluster[0][0:2]
            end_point = cluster[-1][0:2]
            for point in [start_point, end_point]:
                point_center = Point(point[0], point[1])
                # print("point", point)
                # print("polygon.boundary.distance(point_center)", polygon.boundary.distance(point_center))
                # print("polygon.contains(point_center)", polygon.contains(point_center))
                if polygon.contains(point_center) and polygon.boundary.distance(point_center) > 0.1:
                    return True
        for cluster in cluster_group:
            start_point = cluster[0][0:2]
            end_point = cluster[-1][0:2]
            for point in [start_point, end_point]:
                point_center = Point(point[0], point[1])
                # print("point", point)
                # print("polygon.boundary.distance(point_center)", polygon.boundary.distance(point_center))
                # print("polygon.contains(point_center)", polygon.contains(point_center))
                if polygon.contains(point_center) and polygon.boundary.distance(point_center) > 0.1:
                    return True
        return False

    def check_contain_other_cluster(self, last_cluster, cluster_i, segment_index, all_cluster_segments):
        point_1 = last_cluster[0][0:2]
        point_2 = last_cluster[-1][0:2]
        point_3 = cluster_i[-1][0:2]
        point_4 = np.array([point_1[0] + point_3[0] - point_2[0],
                            point_1[1] + point_3[1] - point_2[1]])
        vertex_point = [point_1, point_2, point_3, point_4]
        polygon = Polygon(vertex_point)
        for index, segment in enumerate(all_cluster_segments):
            if index == segment_index:
                continue
            segment = segment[:-1]
            start_points = segment[0]
            end_points = segment[-1]
            start_point = start_points[0][0:2]
            end_point = end_points[-1][0:2]
            for point in [start_point, end_point]:
                point_center = Point(point[0], point[1])
                if polygon.contains(point_center):
                    return True
        return False

    def check_cluster_vector_intersect(self, cluster_1, cluster_2):
        mid_point_1 = cluster_1[int(len(cluster_1) / 2)]
        theta = mid_point_1[4]
        length = np.linalg.norm(np.array(cluster_1[0][0:2]) - np.array(cluster_1[-1][0:2])) / 2
        if abs(cluster_1[0][0] - cluster_1[-1][0]) < abs(cluster_1[0][1] - cluster_1[-1][1]):
            theta = mid_point_1[4] + np.pi / 2
        line_segment = self.generate_line_segment(mid_point_1, theta, length)
        robot_point = [mid_point_1[2], mid_point_1[3]]
        normal = self.compute_normal(robot_point, line_segment)
        normal_1 = normal / np.linalg.norm(normal)

        # print("mid_point_1", mid_point_1)
        # print("normal_1", normal_1)

        mid_point_2 = cluster_2[int(len(cluster_2) / 2)]
        theta = mid_point_2[4]
        length = np.linalg.norm(np.array(cluster_2[0][0:2]) - np.array(cluster_2[-1][0:2])) / 2
        if abs(cluster_2[0][0] - cluster_2[-1][0]) < abs(cluster_2[0][1] - cluster_2[-1][1]):
            theta = mid_point_2[4] + np.pi / 2
        line_segment = self.generate_line_segment(mid_point_2, theta, length)
        robot_point = [mid_point_2[2], mid_point_2[3]]
        normal = self.compute_normal(robot_point, line_segment)
        normal_2 = normal / np.linalg.norm(normal)

        # print("mid_point_2", mid_point_2)
        # print("normal_2", normal_2)

        ray_1 = [[mid_point_1[0], mid_point_1[1]], normal_1]
        ray_2 = [[mid_point_2[0], mid_point_2[1]], normal_2]
        if self.intersect_ray(ray_1, ray_2):
            return True
        else:
            return False

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
    def generate_line_segment(mid_point, theta, length):
        x_0, y_0 = mid_point[0], mid_point[1]
        half_length = length / 2
        start_point = [x_0 - half_length * np.cos(theta), y_0 - half_length * np.sin(theta)]
        end_point = [x_0 + half_length * np.cos(theta), y_0 + half_length * np.sin(theta)]
        return [start_point, end_point]

    @staticmethod
    def translate_line_by_normal(line, normal, dis=0.2):
        start_point = line[0]
        end_point = line[1]
        start_point = np.array(start_point) + dis * normal
        end_point = np.array(end_point) + dis * normal
        return [start_point, end_point]

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
        return np.array([xc, yc]), R

    def fitting(self, lidar_points, theta, s, x_ell_init, y_ell_init, a_init, b_init):
        fitting_theta = theta
        fitting_s = s
        rotated_points = self.rotate_points(np.array(lidar_points), -fitting_theta)
        # print("rotated_points", rotated_points)
        if fitting_s != 0.0:
            bound_init = 0.15
            max_x_bound = np.max(rotated_points[:, 0]) - np.min(rotated_points[:, 0]) + 0.01
            max_y_bound = np.max(rotated_points[:, 1]) - np.min(rotated_points[:, 1]) + 0.01
            max_x_bound = max(max_x_bound, bound_init)
            max_y_bound = max(max_y_bound, bound_init)
            fitting_center, fitting_width, fitting_height, fitting_theta, fitting_s = \
                self.solve_nop(lidar_points, fitting_theta, fitting_s, max_x_bound, max_y_bound, x_ell_init, y_ell_init, a_init, b_init)
        else:
            center, r = self.fit_circle(rotated_points[:, 0], rotated_points[:, 1])
            fitting_center = center
            fitting_width = 2 * r
            fitting_height = 2 * r
        # print("fitting_s", fitting_s)
        return fitting_center, fitting_width, fitting_height, fitting_theta, fitting_s
