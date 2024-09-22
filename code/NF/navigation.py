import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import math

# from NF.static_NF.controller import Controller
from NF.utils import distance
from NF.transformation import ForestToStar, StarToSphere, SphereToPoint, TransformationGradient


# from NF.static_NF.controller import Controller


class NavigationFunction(object):
    # navigation function in star world
    def __init__(self, world, goal_pose, nf_lambda, nf_mu):
        self.goal = goal_pose
        self.world = world
        self.nf_lambda = nf_lambda
        self.nf_mu = nf_mu

        self.mu = 1.0
        self.M = len(self.world.obstacles)

    def transformaton(self, q: np.ndarray) -> np.ndarray:
        # transforamtion from star world to unbounded point world
        forest2star = ForestToStar(self.world, self.goal[0:2], self.nf_mu)
        star2sphere = StarToSphere(self.world, self.goal[0:2], self.nf_lambda)
        sphere2point = SphereToPoint(self.world, self.goal[0:2])
        f = forest2star.compute_f(q)
        h_lambda = star2sphere.compute_h_lambda(f)
        T = sphere2point.compute_T_q(h_lambda)
        bounded_to_unbounded = sphere2point.bounded_pw_to_unbounded(T)
        """
        ===== some print function =====
        print("forest2star", forest2star.compute_f(self.goal[0:2]))
        print("star2sphere", star2sphere.compute_h_lambda(forest2star.compute_f(self.goal[0:2])))
        print("sphere2bounded", sphere2point.compute_T_q(star2sphere.compute_h_lambda(forest2star.compute_f(self.goal[0:2]))))
        print("sphere2unbounded", sphere2point.bounded_pw_to_unbounded
        (sphere2point.compute_T_q(star2sphere.compute_h_lambda(forest2star.compute_f(self.goal[0:2])))))
        """
        return bounded_to_unbounded

    def potential_point_world(self, h: np.ndarray) -> float:
        # potential function at point world
        k = self.M + 1
        obs_rep_potential = 1.0
        sphere2point = SphereToPoint(self.world, self.goal[0:2])
        for obs in self.world.obstacles:
            obs_center = obs[0].center
            # bounded_point_center = sphere2point.compute_T_q(obs_center)
            # point_center = sphere2point.bounded_pw_to_unbounded(bounded_point_center)
            point_center = self.transformaton(obs_center)
            obs_rep_potential *= distance(h, point_center) ** 2
        goal_atr_potential = distance(h, self.transformaton(self.goal[0:2])) ** 2
        return self.mu * goal_atr_potential / (goal_atr_potential + obs_rep_potential ** (1 / k))

    def compute_potential_at_point(self, q: np.ndarray, threshold = 0.0) -> float:
        # potential at star world
        if self.world.check_point_in_free_space(q, threshold = 0.0): #threshold = 0.101 0.02
            point_in_pw = self.transformaton(q)
            return self.potential_point_world(point_in_pw)
        else:
            return 1.0

    # def compute_gradient_point(self, q: np.ndarray, delta=1e-3) -> np.ndarray:
    #     if self.world.check_point_in_free_space(q):
    #         grad_x = (self.compute_potential_at_point(np.array([q[0] + delta, q[1]])) -
    #                   self.compute_potential_at_point(np.array([q[0] - delta, q[1]]))) / (2 * delta)
    #         grad_y = (self.compute_potential_at_point(np.array([q[0], q[1] + delta])) -
    #                   self.compute_potential_at_point(np.array([q[0], q[1] - delta]))) / (2 * delta)
    #         return np.array([[grad_x], [grad_y]])
    #     else:
    #         ws_all_obs = compute_all_obs(self.world)
    #         for ws_or_obs in ws_all_obs:
    #             for obs_i in ws_or_obs:
    #                 if obs_i.check_point_inside(q):
    #                     grad_point = obs_i.grad_potential(q)
    #                     print(grad_point)
    #                     grad_x, grad_y = -grad_point[0], -grad_point[1]
    #                     return np.array([[grad_x], [grad_y]])

    def compute_gradient_point(self, q: np.ndarray, delta=1e-3, threshold = 0.0) -> np.ndarray:
        grad_x = (self.compute_potential_at_point(np.array([q[0] + delta, q[1]]), threshold = threshold) -
                  self.compute_potential_at_point(np.array([q[0] - delta, q[1]]), threshold = threshold)) / (2 * delta)
        grad_y = (self.compute_potential_at_point(np.array([q[0], q[1] + delta]), threshold = threshold) -
                  self.compute_potential_at_point(np.array([q[0], q[1] - delta]), threshold = threshold)) / (2 * delta)
        return np.array([[grad_x], [grad_y]])

    def evaluate_potential(self, xx: np.ndarray, yy: np.ndarray, threshold=0.0, radius=None) -> np.ndarray:
        zz_nav = []
        for qx, qy in zip(xx.ravel(), yy.ravel()):
            q = np.array([qx, qy])
            zz_nav.append(self.compute_potential_at_point(q, threshold = threshold))
        zz_nav = np.asarray(zz_nav).reshape(xx.shape)
        return zz_nav

    def save_vector_follow_path(self, start_pose,
                                save_path='../complex_world/auto_results/tsp_trajectory_data_1.txt'):
        final_traj_x, final_traj_y, final_theta = self.compute_vector_follow_path(start_pose)
        with open(save_path, 'w', encoding='utf-8') as f:
            for i in range(len(final_traj_x)):
                f.write(str(final_traj_x[i]) + ',' + str(final_traj_y[i]) + ',' + str(final_theta[i]) + '\n')
        f.close()
        print("========== trajectory data has been saved ==========")

    def evaluate_gradient(self, xx: np.ndarray, yy: np.ndarray, threshold = 0.0) -> np.ndarray:
        grad_x, grad_y = [], []
        for qx, qy in zip(xx.ravel(), yy.ravel()):
            q = np.array([qx, qy])
            gradient = self.compute_gradient_point(q, threshold = threshold)
            if np.linalg.norm(gradient) == 0:
                normalized_grad = np.array([[0], [0]])
            else:
                normalized_grad = - gradient / np.linalg.norm(gradient)

            grad_x.append(normalized_grad[0][0])
            grad_y.append(normalized_grad[1][0])
        return grad_x, grad_y

    def save_evaluate_gradient(self, threshold = 0.0) -> np.ndarray:
        xx, yy = self.world.workspace[0][0].workspace_meshgrid(resolution=0.23, threshold=0.32)
        save_path = '../static_NF/evaluate_gradient.txt'
        with open(save_path, 'w', encoding='utf-8') as f:
            for qx, qy in zip(xx.ravel(), yy.ravel()):
                q = np.array([qx, qy])
                gradient = self.compute_gradient_point(q, threshold = threshold)
                if np.linalg.norm(gradient) == 0:
                    normalized_grad = np.array([[0], [0]])
                else:
                    normalized_grad = - gradient / np.linalg.norm(gradient)
                f.write(str(qx) + ',' + str(qy) + ',' + str(normalized_grad[0][0]) + ',' + str(normalized_grad[1][0]) + '\n')
        f.close()

    def calculate_path(self, start: np.ndarray, max_steps=2000, delta_t=0.05) -> np.ndarray:
        path = np.zeros((max_steps, 2))
        path[0] = start[0:2]
        step = 1
        current_vel = np.array([0, 0])
        current_pos = start[0:2]
        vel_list = []
        acc_list = []
        damp_factor = 4.0 * (self.mu ** 0.5)
        sphere2point = SphereToPoint(self.world, self.goal[0:2])
        for obs in self.world.obstacles:
            K = self.M + 1
            damp_factor *= distance(sphere2point.bounded_pw_to_unbounded(self.goal[0:2]),
                                    sphere2point.bounded_pw_to_unbounded(obs[0].center)) ** (- 1 / K)
        while distance(path[step - 1], self.goal[0:2]) > 0.01 and step < max_steps:

            gradient = self.compute_gradient_point(path[step - 1])
            # gradient = self.compute_mapped_gradient(path[step - 1])
            gradient = [gradient[0][0], gradient[1][0]]
            """
            first order controller.py
            """

            current_vel = - \
                              math.sqrt(
                                  2 * self.compute_potential_at_point(path[step - 1])) * (
                                  gradient / np.linalg.norm(gradient))
            current_pos = current_vel * delta_t + current_pos
            """
            second order controller.py
            """
            # current_input = - gradient - damp_factor * current_vel
            # current_acc = current_input
            # current_vel = current_acc * delta_t + current_vel
            # current_pos = current_vel * delta_t + 0.5 * current_acc * delta_t**2 + current_pos
            # print(current_acc, current_vel)
            # vel_list.append((current_vel[0] ** 2 + current_vel[1] ** 2) ** 0.5)
            # acc_list.append((current_acc[0] ** 2 + current_acc[1] ** 2) ** 0.5)

            """first order result"""
            # print(current_pos)
            path[step] = [current_pos[0], current_pos[1]]

            step += 1
        return path[:step], vel_list, acc_list

    def compute_mapped_gradient_1(self, q):
        return np.dot(self.discontinuous_map_1(q), self.compute_gradient_point(q))

    def compute_mapped_gradient(self, q):
        return np.dot(self.discontinuous_map(q), self.compute_mapped_gradient_1(q))

    def compute_partial_list(self, q: np.ndarray, delta=1e-3):
        partial_x_x = (self.compute_mapped_gradient(np.array([q[0] + delta, q[1]]))[0][0] -
                       self.compute_mapped_gradient(np.array([q[0] - delta, q[1]]))[0][0]) / (2 * delta)
        partial_x_y = (self.compute_mapped_gradient(np.array([q[0], q[1] + delta]))[0][0] -
                       self.compute_mapped_gradient(np.array([q[0], q[1] - delta]))[0][0]) / (2 * delta)
        partial_y_x = (self.compute_mapped_gradient(np.array([q[0] + delta, q[1]]))[1][0] -
                       self.compute_mapped_gradient(np.array([q[0] - delta, q[1]]))[1][0]) / (2 * delta)
        partial_y_y = (self.compute_mapped_gradient(np.array([q[0], q[1] + delta]))[1][0] -
                       self.compute_mapped_gradient(np.array([q[0], q[1] - delta]))[1][0]) / (2 * delta)
        return np.array([partial_x_x, partial_x_y, partial_y_x, partial_y_y])

    def compute_partial_gradient(self, q: np.ndarray, delta=1e-3):
        partial_x_x = (self.compute_gradient_point(np.array([q[0] + delta, q[1]]))[0][0] -
                       self.compute_gradient_point(np.array([q[0] - delta, q[1]]))[0][0]) / (2 * delta)
        partial_x_y = (self.compute_gradient_point(np.array([q[0], q[1] + delta]))[0][0] -
                       self.compute_gradient_point(np.array([q[0], q[1] - delta]))[0][0]) / (2 * delta)
        partial_y_x = (self.compute_gradient_point(np.array([q[0] + delta, q[1]]))[1][0] -
                       self.compute_gradient_point(np.array([q[0] - delta, q[1]]))[1][0]) / (2 * delta)
        partial_y_y = (self.compute_gradient_point(np.array([q[0], q[1] + delta]))[1][0] -
                       self.compute_gradient_point(np.array([q[0], q[1] - delta]))[1][0]) / (2 * delta)
        return np.array([partial_x_x, partial_x_y, partial_y_x, partial_y_y])

    def discontinuous_map_1(self, q, a=5.0):
        if self.compute_potential_at_point(q) == 1:
            s_d = 0.0
        else:
            s_d = math.exp(a - a / (1 - self.compute_potential_at_point(q)) ** 2)
        theta_1 = np.arctan2(q[1] - self.goal[1], q[0] - self.goal[0])
        gradient = - self.compute_gradient_point(q)
        theta_2 = np.arctan2(gradient[1][0], gradient[0][0])
        theta = (theta_2 - theta_1 + np.pi) % (2 * np.pi) - np.pi
        z = theta * s_d + np.pi * np.sign(theta) * (1 - s_d)
        # gamma = np.array([[np.cos(z), np.sin(z)], [- np.sin(z), np.cos(z)]])
        gamma = np.array([[- np.cos(z), - np.sin(z)], [np.sin(z), - np.cos(z)]])
        return gamma

    def discontinuous_map(self, q, a=5.0):
        theta = self.goal[2]
        if self.compute_potential_at_point(q) == 1:
            s_d = 0.0
        else:
            s_d = math.exp(a - a / (1 - self.compute_potential_at_point(q)) ** 2)
        x = (q[0] - self.goal[0]) * np.cos(theta) + (q[1] - self.goal[1]) * np.sin(theta)
        y = (q[0] - self.goal[0]) * (- np.sin(theta)) + (q[1] - self.goal[1]) * np.cos(theta)
        z = np.arctan2(y, x) * s_d + np.pi * np.sign(y) * (1 - s_d)
        gamma = np.array([[- np.cos(z), np.sin(z)], [- np.sin(z), -np.cos(z)]])
        return gamma

    def evaluate_mapped_gradient(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        grad_x, grad_y = [], []
        for qx, qy in zip(xx.ravel(), yy.ravel()):
            q = np.array([qx, qy])
            gradient = self.compute_mapped_gradient(q)
            if np.linalg.norm(gradient) == 0:
                normalized_grad = np.array([[0], [0]])
            else:
                normalized_grad = - gradient / np.linalg.norm(gradient)
            grad_x.append(normalized_grad[0][0])
            grad_y.append(normalized_grad[1][0])
        return grad_x, grad_y

    def vector_follow_controller(self, start, goal_pose, k_v=1.0, k_omega_1=0.8, k_omega_2=0.6): # 0.5 1.0 0.5
        gradient = self.compute_mapped_gradient(np.array(start[0: 2]))
        f_x, f_y = - gradient[0][0], - gradient[1][0]
        partial_x_x, partial_x_y, partial_y_x, partial_y_y = - self.compute_partial_list(start)
        # velocity = k_v * np.sqrt(2 * self.compute_potential_at_point(start[0:2]))
        # start_pose_1 = np.array([0.5, 0.45, 0])
        # goal_pose_1 = np.array([2.6, 0.4, 0])
        # velocity = k_v * min(5 * np.tanh(distance(start[0: 2], start_pose_1[0: 2])),
        #                   np.tanh(5 * distance(start[0:2], goal_pose_1[0:2])))
        # print("velocity_1:", velocity)
        velocity = k_v * np.tanh(1.2 * distance(start[0: 2], goal_pose[0: 2])) #1.0
        # print("velocity_2", velocity_1)
        theta_diff = (start[2] - np.arctan2(f_y, f_x) + np.pi) % (2 * np.pi) - np.pi
        yaw_velocity = - k_omega_1 * theta_diff + k_omega_2 * (velocity / np.linalg.norm(gradient) ** 2) * \
                       (f_x * (partial_y_y * np.sin(start[2]) + partial_y_x * np.cos(start[2])) - \
                        f_y * (partial_x_y * np.sin(start[2]) + partial_x_x * np.cos(start[2])))
        return velocity, yaw_velocity

    def gradient_follow_controller(self, start_pose, goal_pose, k_v=1.0, k_omega=1.0):
        gradient = self.compute_gradient_point(np.array(start_pose[0: 2]))
        f_x, f_y = - gradient[0][0], - gradient[1][0]
        partial_x_x, partial_x_y, partial_y_x, partial_y_y = - self.compute_partial_gradient(start_pose[0: 2])
        velocity = k_v * np.tanh(distance(start_pose[0: 2], goal_pose[0: 2]))
        theta_diff = (start_pose[2] - np.arctan2(f_y, f_x) + np.pi) % (2 * np.pi) - np.pi
        yaw_velocity = - k_omega * theta_diff + (velocity / np.linalg.norm(gradient) ** 2) * \
                       (f_x * (partial_y_y * np.sin(start_pose[2]) + partial_y_x * np.cos(start_pose[2])) - \
                        f_y * (partial_x_y * np.sin(start_pose[2]) + partial_x_x * np.cos(start_pose[2])))
        return velocity, yaw_velocity

    def calculate_nf_pose(self, start, max_steps=100, step_size=0.1):
        all_pose = np.zeros((max_steps, 3))
        start_gradient = self.compute_gradient_point(start)
        start_theta = np.arctan2(- start_gradient[1], - start_gradient[0])
        start_pose = [start[0], start[1], start_theta]
        all_pose[0] = start_pose
        step = 1
        goal_pose = self.goal
        while distance(all_pose[step - 1][0: 2], self.goal[0:2]) > 0.2 and step < max_steps:
            gradient = self.compute_gradient_point(all_pose[step - 1][0: 2])
            normalized_grad = - gradient / np.linalg.norm(gradient)
            direction = normalized_grad + 0.1 * np.tan((goal_pose[2] -
                                                        np.arctan2(normalized_grad[1], normalized_grad[1])))
            new_point = [all_pose[step - 1][0] + step_size * direction[0],
                         all_pose[step - 1][1] + step_size * direction[1]]
            new_gradient = self.compute_gradient_point(new_point)
            new_theta = math.atan2(- new_gradient[1], - new_gradient[0])
            all_pose[step] = [new_point[0], new_point[1], new_theta]
            step += 1
        all_pose[step] = goal_pose
        step += 1
        return all_pose[:step], step

    def compute_vector_follow_path(self, start_pose, delta_t=0.1, step_size=1000):
        goal_pose = self.goal
        x_traj = []
        y_traj = []
        theta_traj = []
        current_pose = start_pose
        r = ((current_pose[0] - goal_pose[0]) ** 2 +
             (current_pose[1] - goal_pose[1]) ** 2) ** 0.5
        step = 0
        while r > 0.02 and step < step_size:
            x_traj.append(current_pose[0])
            y_traj.append(current_pose[1])
            theta_traj.append(current_pose[2])
            # controller = Controller(self.world, current_pose, goal_pose, global_goal, self.nf_lambda, self.nf_mu)
            # velocity, yaw_velocity = self.gradient_follow_controller(current_pose, goal_pose)
            # velocity, yaw_velocity = self.vector_follow_controller(current_pose, goal_pose)
            velocity, yaw_velocity = self.ego_controller(current_pose, goal_pose)
            current_pose[0] = current_pose[0] + \
                              velocity * np.cos(current_pose[2]) * delta_t
            current_pose[1] = current_pose[1] + \
                              velocity * np.sin(current_pose[2]) * delta_t
            current_pose[2] = current_pose[2] + yaw_velocity * delta_t
            r = ((current_pose[0] - goal_pose[0]) ** 2 +
                 (current_pose[1] - goal_pose[1]) ** 2) ** 0.5
            step += 1
        return x_traj, y_traj, theta_traj

    """
    -----ego planner-----
    """

    # def compute_nf_ego_path(self, start, delta_t=0.001):
    #     all_pose, step_num = self.calculate_nf_pose(start)
    #     x_traj = []
    #     y_traj = []
    #     theta_traj = []
    #     for i in range(step_num - 1):
    #         start_pose = all_pose[i]
    #         goal_pose = all_pose[i + 1]
    #         current_pose = [start_pose[0], start_pose[1], start_pose[2]]
    #         r = ((current_pose[0] - goal_pose[0]) ** 2 +
    #              (current_pose[1] - goal_pose[1]) ** 2) ** 0.5
    #         while (r > 0.001):
    #             x_traj.append(current_pose[0])
    #             y_traj.append(current_pose[1])
    #             theta_traj.append(current_pose[2])
    #             velocity, yaw_velocity = self.ego_controller(current_pose, goal_pose)
    #             current_pose[0] = current_pose[0] + \
    #                               velocity * np.cos(current_pose[2]) * delta_t
    #             current_pose[1] = current_pose[1] + \
    #                               velocity * np.sin(current_pose[2]) * delta_t
    #             current_pose[2] = current_pose[2] + yaw_velocity * delta_t
    #             r = ((current_pose[0] - goal_pose[0]) ** 2 +
    #                  (current_pose[1] - goal_pose[1]) ** 2) ** 0.5
    #         x_traj.append(current_pose[0])
    #         y_traj.append(current_pose[1])
    #         theta_traj.append(current_pose[2])
    #     return x_traj, y_traj

    def cartesian_to_egocentric(self, start_pose, goal_pose):
        rho = ((start_pose[0] - goal_pose[0]) ** 2 + (start_pose[1] - goal_pose[1]) ** 2) ** 0.5
        if rho == 0:
            alpha = start_pose[2]
        else:
            alpha = np.arctan2(goal_pose[1] - start_pose[1], goal_pose[0] - start_pose[0])
        phi = (goal_pose[2] - alpha + np.pi) % (2 * np.pi) - np.pi
        delta = (start_pose[2] - alpha + np.pi) % (2 * np.pi) - np.pi
        return rho, phi, delta

    def ego_controller(self, start_pose, goal_pose, k_1=1.0, k_2=0.6):
        rho, theta, delta = self.cartesian_to_egocentric(start_pose, goal_pose)
        velocity = 0.1
        yaw_velocity = (- velocity / rho) * (k_2 * (delta - np.arctan(- k_2 * theta)
                                                    ) + (1 + k_1 / (1 + (k_1 * theta) ** 2)) * np.sin(delta))
        return velocity, yaw_velocity

    """
    -----old gradient functions-----
    """

    def grad_transformation(self, q: np.ndarray) -> np.ndarray:
        # the gradient of the transformation from star world to unbounded point world
        transformation_gradient = TransformationGradient(self.world, self.goal, self.nf_lambda)
        return transformation_gradient.grad_tf_world_to_pw(q)

    def grad_potential_point_world(self, h: np.ndarray) -> np.ndarray:
        k = self.M + 1
        product = 1.0
        sphere2point = SphereToPoint(self.world)
        for obs in self.world.obstacles:
            for obs_i in obs:
                product *= distance(h, sphere2point.bounded_pw_to_unbounded(obs_i.center)) ** 2
        norms = []
        grad_norms = []
        for obs in self.world.obstacles:
            for obs_i in obs:
                norms.append(distance(h, sphere2point.bounded_pw_to_unbounded(obs_i.center)) ** 2)
                grad_norms.append(2.0 * (h - sphere2point.bounded_pw_to_unbounded(obs_i.center)))
        grad_norm_product = np.zeros(2)
        for i in range(len(norms)):
            norm_product = grad_norms[i]
            for j in range(len(norms)):
                if j != i:
                    norm_product *= norms[j]
            grad_norm_product += norm_product
        return self.mu * (2 * (h - sphere2point.bounded_pw_to_unbounded(self.goal)) * (
                distance(h, sphere2point.bounded_pw_to_unbounded(self.goal)) ** 2 +
                product ** (1 / k)) - distance(h, sphere2point.bounded_pw_to_unbounded(self.goal)) ** 2 * (
                                  2 * (h - sphere2point.bounded_pw_to_unbounded(self.goal)) +
                                  (1 / k) * product ** (1 / k - 1) * grad_norm_product)) / (
                       distance(h, sphere2point.bounded_pw_to_unbounded(self.goal)) ** 2 + product ** (1 / k)) ** 2

    def grad_potential_at_point(self, q: np.ndarray) -> np.ndarray:
        if self.world.check_point_in_free_space(q):
            return np.dot(np.array([[self.grad_potential_point_world(self.transformaton(q))[0],
                                     self.grad_potential_point_world(self.transformaton(q))[1]]]),
                          self.grad_transformation(q))
        else:
            for obs in self.world.obstacles:
                for obs_i in obs:
                    if obs_i.check_point_inside(q) == True:
                        return np.array([[obs_i.center[0] - q[0], obs_i.center[1] - q[1]]])
