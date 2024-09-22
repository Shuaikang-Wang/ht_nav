import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import math

from NF.utils import distance
from NF.navigation import NavigationFunction


class NFController(object):
    def __init__(self, world, start_pose, goal_pose, nf_lambda, nf_mu):
        self.current_pose = start_pose
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        self.world = world
        self.nf_lambda = nf_lambda
        self.nf_mu = nf_mu

    def nf(self):
        return NavigationFunction(self.world, self.goal_pose, self.nf_lambda, self.nf_mu)

    def vector_follow_controller(self, tau =1.0, k_v=1.0, k_omega_1=0.8):
        velocity = k_v * np.tanh(tau * distance(self.current_pose[0:2], self.goal_pose[0:2]))

        gradient = self.compute_mapped_gradient(np.array(self.current_pose[0: 2]))
        f_x, f_y = - gradient[0][0], - gradient[1][0]
        partial_x_x, partial_x_y, partial_y_x, partial_y_y = - self.compute_partial_list(self.current_pose[0:2])

        theta_diff = (self.current_pose[2] - np.arctan2(f_y, f_x) + np.pi) % (2 * np.pi) - np.pi
        theta_deri = (velocity / (np.linalg.norm(gradient) ** 2)) * \
                    (f_x * (partial_y_y * np.sin(self.current_pose[2]) + partial_y_x * np.cos(self.current_pose[2])) -
                    f_y * (partial_x_y * np.sin(self.current_pose[2]) + partial_x_x * np.cos(self.current_pose[2])))
        yaw_velocity = - k_omega_1 * theta_diff + theta_deri
        return velocity, yaw_velocity

    def gradient_controller(self):
        gradient = self.nf().compute_gradient_point(np.array(self.current_pose[0: 2]))
        normalized_grad = - gradient / np.linalg.norm(gradient)
        direction = normalized_grad + 0.1 * np.tan((self.goal_pose[2] -
                                                    np.arctan2(normalized_grad[1], normalized_grad[1])))
        gradient = [direction[0][0], direction[1][0]]

        return gradient

    def gradient_follow_controller(self, k_v=1.0, k_omega_1=2.0, k_omega_2=0.45):
        gradient = self.nf().compute_gradient_point(np.array(self.current_pose[0: 2]))
        f_x, f_y = - gradient[0][0], - gradient[1][0]
        partial_x_x, partial_x_y, partial_y_x, partial_y_y = - self.compute_partial_gradient(self.current_pose[0: 2])
        velocity = k_v * np.sqrt(2 * self.nf().compute_potential_at_point(self.current_pose[0:2]))
        theta_diff = (self.current_pose[2] - np.arctan2(f_y, f_x) + np.pi) % (2 * np.pi) - np.pi
        yaw_velocity = - k_omega_1 * theta_diff + k_omega_2 * (velocity / np.linalg.norm(gradient) ** 2) * \
                       (f_x * (partial_y_y * np.sin(self.current_pose[2]) + partial_y_x * np.cos(self.current_pose[2])) - \
                        f_y * (partial_x_y * np.sin(self.current_pose[2]) + partial_x_x * np.cos(self.current_pose[2])))
        return velocity, yaw_velocity

    def compute_mapped_gradient_1(self, q):
        return np.dot(self.discontinuous_map_1(q), self.nf().compute_gradient_point(q))

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
        partial_x_x = (self.nf().compute_gradient_point(np.array([q[0] + delta, q[1]]))[0][0] -
                       self.nf().compute_gradient_point(np.array([q[0] - delta, q[1]]))[0][0]) / (2 * delta)
        partial_x_y = (self.nf().compute_gradient_point(np.array([q[0], q[1] + delta]))[0][0] -
                       self.nf().compute_gradient_point(np.array([q[0], q[1] - delta]))[0][0]) / (2 * delta)
        partial_y_x = (self.nf().compute_gradient_point(np.array([q[0] + delta, q[1]]))[1][0] -
                       self.nf().compute_gradient_point(np.array([q[0] - delta, q[1]]))[1][0]) / (2 * delta)
        partial_y_y = (self.nf().compute_gradient_point(np.array([q[0], q[1] + delta]))[1][0] -
                       self.nf().compute_gradient_point(np.array([q[0], q[1] - delta]))[1][0]) / (2 * delta)
        return np.array([partial_x_x, partial_x_y, partial_y_x, partial_y_y])

    def discontinuous_map_1(self, q, a=0.5):
        if self.nf().compute_potential_at_point(q) == 1:
            s_d = 0.0
        else:
            s_d = math.exp(a - a / (1 - self.nf().compute_potential_at_point(q)) ** 2)
        theta_1 = np.arctan2(q[1] - self.goal_pose[1], q[0] - self.goal_pose[0])
        gradient = - self.nf().compute_gradient_point(q)
        theta_2 = np.arctan2(gradient[1][0], gradient[0][0])
        theta = (theta_2 - theta_1 + np.pi) % (2 * np.pi) - np.pi
        z = theta * s_d + np.pi * np.sign(theta) * (1 - s_d)
        # gamma = np.array([[np.cos(z), np.sin(z)], [- np.sin(z), np.cos(z)]])
        gamma = np.array([[- np.cos(z), - np.sin(z)], [np.sin(z), - np.cos(z)]])
        return gamma

    def discontinuous_map(self, q, a=0.5):
        theta = self.goal_pose[2]
        if self.nf().compute_potential_at_point(q) == 1:
            s_d = 0.0
        else:
            s_d = math.exp(a - a / (1 - self.nf().compute_potential_at_point(q)) ** 2)
        x = (q[0] - self.goal_pose[0]) * np.cos(theta) + (q[1] - self.goal_pose[1]) * np.sin(theta)
        y = (q[0] - self.goal_pose[0]) * (- np.sin(theta)) + (q[1] - self.goal_pose[1]) * np.cos(theta)
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