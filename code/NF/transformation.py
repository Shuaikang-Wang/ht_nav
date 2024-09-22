import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from math import exp, sqrt

from NF.utils import distance, compute_gamma, grad_gamma, compute_all_obs, compute_squicle_length_ray
from NF.geometry import Rectangular


class ForestToStar(object):
    # transformation from forest world to star world
    def __init__(self, world, goal, nf_mu):
        self.world = world
        self.goal = goal
        self.nf_mu = nf_mu
        """
        nf_mu is a list [mu_1, mu_2, ,mu_d]
        all_depth: depth for each tree before purging, 0, 1, 2,...
        all_purging_depth: depth for each tree as purging, 0, 1, 2,...
        """
        self.all_depth = []
        self.compute_all_depth()
    
    def compute_all_depth(self) -> list:
        ws_all_obs = compute_all_obs(self.world)
        for obs in ws_all_obs:
            self.all_depth.append(len(obs) - 1)

    def compute_all_purging_depth(self, d: int) -> list:
        all_purging_depth = []
        for depth in self.all_depth:
            if depth <= d:
                all_purging_depth.append(0)
            else:
                all_purging_depth.append(depth - d)
        return all_purging_depth
    
    def compute_tree_num(self, d: int) -> int:
        tree_num = 0
        all_purging_depth = self.compute_all_purging_depth(d)
        for depth in all_purging_depth:
            if depth > 0:
                tree_num += 1
        return tree_num
    
    def check_parent_center_inside(self, obs_parent, obs_son, threshold=0.0):
        return obs_son.check_point_inside(obs_parent.center, threshold=0.0)
    
    def compute_virtual_center(self, obs_parent, obs_son):
        x_p_min, x_p_max, y_p_min, y_p_max = obs_parent.center[0] - obs_parent.width / 2, obs_parent.center[0] + obs_parent.width / 2, \
                                             obs_parent.center[1] - obs_parent.height / 2, obs_parent.center[1] + obs_parent.height / 2
        x_s_min, x_s_max, y_s_min, y_s_max = obs_son.center[0] - obs_son.width / 2, obs_son.center[0] + obs_son.width / 2, \
                                             obs_son.center[1] - obs_son.height / 2, obs_son.center[1] + obs_son.height / 2
        x_v_min, x_v_max, y_v_min, y_v_max = max(x_p_min, x_s_min), min(x_p_max, x_s_max), max(y_p_min, y_s_min), min(y_p_max, y_s_max)
        x_v, y_v = 0.5 * (x_v_min + x_v_max), 0.5 * (y_v_min + y_v_max)
        virtual_center = np.array([x_v, y_v])
        return virtual_center
    
    def compute_virtual_length_ray(self, obs_parent, obs_son, q):
        virtual_center = self.compute_virtual_center(obs_parent, obs_son)
        ray = np.array(q) - np.array(virtual_center)
        if ray[0] > 0.0 and ray[1] >= 0.0:
            virtual_width = obs_parent.center[0] + obs_parent.width / 2 - virtual_center[0]
            virtual_height = obs_parent.center[1] + obs_parent.height / 2 - virtual_center[1]
            virtual_length_ray = compute_squicle_length_ray(virtual_width, virtual_height, ray, 0.0, 0.99)
        elif ray[0] <= 0.0 and ray[1] > 0.0:
            virtual_width = virtual_center[0] - (obs_parent.center[0] - obs_parent.width / 2)
            virtual_height = obs_parent.center[1] + obs_parent.height / 2 - virtual_center[1]
            virtual_length_ray = compute_squicle_length_ray(virtual_width, virtual_height, ray, 0.0, 0.99)
        elif ray[0] < 0.0 and ray[1] <= 0.0:
            virtual_width = virtual_center[0] - (obs_parent.center[0] - obs_parent.width / 2)
            virtual_height = virtual_center[1] - (obs_parent.center[1] - obs_parent.height / 2) 
            virtual_length_ray = compute_squicle_length_ray(virtual_width, virtual_height, ray, 0.0, 0.99)
        else:
            virtual_width = obs_parent.center[0] + obs_parent.width / 2 - virtual_center[0]
            virtual_height = virtual_center[1] - (obs_parent.center[1] - obs_parent.height / 2) 
            virtual_length_ray = compute_squicle_length_ray(virtual_width, virtual_height, ray, 0.0, 0.99)
        return virtual_length_ray
    
    def compute_virtual_ws(self, ws, ws_son):
        center = [0.0, 0.0]
        height = 0.0
        width = 0.0
        if ws_son.center[0] > ws.center[0] and ws_son.width > ws_son.height:
            center = [ws_son.center[0] + ws_son.width / 2, ws_son.center[1]]
            width = 2 * (ws_son.center[0] + ws_son.width / 2 - (ws.center[0] + ws.width / 2))
            height = 2 * width
        elif ws_son.center[1] > ws.center[1] and ws_son.width < ws_son.height:
            center = [ws_son.center[0], ws_son.center[1] + ws_son.height / 2]
            height = 2 * (ws_son.center[1] + ws_son.height / 2 - (ws.center[1] + ws.height / 2))
            width = 2 * height
        elif ws_son.center[0] < ws.center[0] and ws_son.width > ws_son.height:
            center = [ws_son.center[0] - ws_son.width / 2, ws_son.center[1]]
            width = 2 * (ws.center[0] - ws.width / 2 - (ws_son.center[0] - ws_son.width / 2))
            height = 2 * width
        elif ws_son.center[1] < ws.center[1] and ws_son.width < ws_son.height:
            center = [ws_son.center[0], ws_son.center[1] - ws_son.height / 2]
            height = 2 * (ws.center[1] - ws.height / 2 - (ws_son.center[1] - ws_son.height / 2))
            width = 2 * height
        _type = 'Rectangle'
        virtual_parent = Rectangular(_type, center, width - 0.16, height + 0.16)
        return virtual_parent
    
    def compute_all_leaves_beta(self, q: np.ndarray, d: int) -> list:
        all_leaves_beta = []
        ws_all_obs = compute_all_obs(self.world)
        all_depth = self.compute_all_purging_depth(d)
        for k, obs in enumerate(ws_all_obs):
            depth = all_depth[k]
            if depth > 0:
                leaf_beta_i = obs[depth].potential(q)
                all_leaves_beta.append(leaf_beta_i)
        return all_leaves_beta

    def compute_all_beta_bar(self, q: np.ndarray, d: int) -> list:
        all_beta_bar = []
        ws_all_obs = compute_all_obs(self.world)
        tree_num = self.compute_tree_num(d)
        all_depth = self.compute_all_purging_depth(d)
        for i in range(tree_num):
            beta_bar_i = compute_gamma(q, self.goal)
            product1 = 1.0
            product2 = 1.0
            num = 0
            num_ws = 0
            for k, obs in enumerate(ws_all_obs):
                depth = all_depth[k]
                if obs[0].type == 'Workspace':
                    num_ws += 1
                    if depth == 0:
                        product1 *= obs[depth].potential(q)
                    else:
                        if num == i:
                            while depth > 1:
                                product1 *= obs[depth - 1].potential(q)
                                depth -= 1
                        else:
                            product2 *= obs[depth].potential(q)
                            while depth >= 0:
                                product1 *= obs[depth].potential(q)
                                depth -= 1
                        num += 1
                else:
                    if depth == 0:
                        product1 *= obs[depth].potential(q)
                    else:
                        if num == i:
                            while depth > 1:
                                product1 *= obs[depth - 2].potential(q)
                                depth -= 1
                        else:
                            product2 *= obs[depth].potential(q)
                            while depth >= 0:
                                product1 *= obs[depth].potential(q)
                                depth -= 1
                        num += 1
            # product1 = product1 / ((num_ws - 1) * ws_all_obs[0][0].potential(q))
            beta_bar_i = beta_bar_i * product1 * product2
            all_beta_bar.append(beta_bar_i)
        return all_beta_bar

    # def compute_all_beta_bar(self, q: np.ndarray, d: int) -> list:
    #     all_beta_bar = []
    #     ws_all_obs = compute_all_obs(self.world)
    #     tree_num = self.compute_tree_num(d)
    #     all_depth = self.compute_all_purging_depth(d)
    #     for i in range(tree_num):
    #         beta_bar_i = compute_gamma(q, self.goal)
    #         all_beta_bar.append(beta_bar_i)
    #     return all_beta_bar

    # def compute_all_beta_bar(self, q: np.ndarray, d: int) -> list:
    #     all_beta_bar = []
    #     ws_all_obs = compute_all_obs(self.world)
    #     tree_num = self.compute_tree_num(d)
    #     all_depth = self.compute_all_purging_depth(d)
    #     for i in range(tree_num):
    #         beta_bar_i = compute_gamma(q, self.goal)
    #         product1 = 1.0
    #         product2 = 1.0
    #         num = 0
    #         for k, obs in enumerate(ws_all_obs):
    #             depth = all_depth[k]
    #             if k == 0:
    #                 if depth == 0:
    #                     product1 *= obs[depth].potential(q)
    #                 else:
    #                     if num == i:
    #                         while depth > 1:
    #                             product1 *= obs[depth - 1].potential(q)
    #                             depth -= 1
    #                     else:
    #                         product2 *= obs[depth].potential(q)
    #                         while depth >= 0:
    #                             product1 *= obs[depth].potential(q)
    #                             depth -= 1
    #                     num += 1
    #             else:
    #                 if depth == 0:
    #                     product1 *= obs[depth].potential(q)
    #                 else:
    #                     if num == i:
    #                         while depth > 1:
    #                             product1 *= obs[depth - 2].potential(q)
    #                             depth -= 1
    #                     else:
    #                         product2 *= obs[depth].potential(q)
    #                         while depth >= 0:
    #                             product1 *= obs[depth].potential(q)
    #                             depth -= 1
    #                     num += 1
    #         beta_bar_i = beta_bar_i * product1 * product2
    #         all_beta_bar.append(beta_bar_i)
    #     return all_beta_bar
    
    def compute_all_E(self, d: int) -> list:
        all_E = []
        tree_num = self.compute_tree_num(d)
        for i in range(tree_num):
            E_i = 1.0
            all_E.append(E_i)
        return all_E
    
    def compute_all_beta_tilde(self, q: np.ndarray, d: int) -> list:
        all_beta_tilde = []
        all_E = self.compute_all_E(d)
        all_leaves_beta = self.compute_all_leaves_beta(q, d)
        ws_all_obs = compute_all_obs(self.world)
        all_depth = self.compute_all_purging_depth(d)
        i = 0
        for k, obs in enumerate(ws_all_obs):
            depth = all_depth[k]
            if depth > 0:
                beta_p_i = obs[depth - 1].potential(q)
                E_beta_i = 2 * all_E[i] - all_leaves_beta[i]
                beta_tilde_i = beta_p_i + E_beta_i + sqrt(beta_p_i**2 + E_beta_i**2)
                i += 1
                all_beta_tilde.append(beta_tilde_i)
        return all_beta_tilde
    
    def compute_all_kappa_tilde(self, q: np.ndarray, d: int) -> list:
        all_kappa_tilde = []
        all_E = self.compute_all_E(d)
        all_leaves_beta = self.compute_all_leaves_beta(q, d)
        ws_all_obs = compute_all_obs(self.world)
        all_depth = self.compute_all_purging_depth(d)
        i = 0
        for k, obs in enumerate(ws_all_obs):
            depth = all_depth[k]
            if depth > 0:
                beta_p_i = obs[depth - 1].potential(q)
                E_beta_i = all_leaves_beta[i] - 2 * all_E[i]
                kappa_tilde_i = beta_p_i + E_beta_i + sqrt(beta_p_i**2 + E_beta_i**2)
                i += 1
                all_kappa_tilde.append(kappa_tilde_i)
        return all_kappa_tilde
    
    def compute_all_sigma(self, q: np.ndarray, d: int) -> list:
        all_sigma = []
        all_beta_bar = self.compute_all_beta_bar(q, d)
        all_bata_tilde = self.compute_all_beta_tilde(q, d)
        all_leaves_bata = self.compute_all_leaves_beta(q, d)
        tree_num = self.compute_tree_num(d)
        for i in range(tree_num):
            sigma_i = (all_beta_bar[i] * all_bata_tilde[i]) / (all_beta_bar[i] * all_bata_tilde[i] +
                                                               self.nf_mu[d] * all_leaves_bata[i])
            all_sigma.append(sigma_i)
        # print("all_sigma", all_sigma)
        return all_sigma

    # def compute_all_sigma(self, q: np.ndarray, d: int) -> list:
    #     all_sigma = []
    #     tree_num = self.compute_tree_num(d)
    #     for i in range(tree_num):
    #         sigma_i = 1.0
    #         all_sigma.append(sigma_i)
    #     # print("all_sigma", all_sigma)
    #     return all_sigma
    
    def compute_all_v(self, q: np.ndarray, d: int) -> list:
        all_v = []
        kappa_tilde = self.compute_all_kappa_tilde(q, d)
        # print("kappa_tilde", kappa_tilde)
        all_leaves_beta = self.compute_all_leaves_beta(q, d)
        # print("all_leaves_beta", all_leaves_beta)
        ws_all_obs = compute_all_obs(self.world)
        all_depth = self.compute_all_purging_depth(d)
        i = 0
        for k, obs in enumerate(ws_all_obs):
            depth = all_depth[k]
            if depth > 0:
                if obs[0].type == 'Workspace':
                    if depth == 1:
                        virtual_ws = self.compute_virtual_ws(obs[depth - 1], obs[depth])
                        rho_i = compute_squicle_length_ray(virtual_ws.width / 2, virtual_ws.height / 2, q - virtual_ws.center, virtual_ws.theta, virtual_ws.s)
                        maxVal = (obs[depth].width / 2)**2 + (obs[depth].height / 2)**2
                        v_i = rho_i * (1 + all_leaves_beta[i] / maxVal) / distance(q, virtual_ws.center)
                        all_v.append(v_i)
                    else:
                        if self.check_parent_center_inside(obs[depth - 1], obs[depth]):
                            rho_i = compute_squicle_length_ray(obs[depth - 1].width / 2, obs[depth - 1].height / 2,
                                                               q - obs[depth - 1].center, virtual_ws.theta, virtual_ws.s)
                            center = obs[depth - 1].center
                        else:
                            rho_i = self.compute_virtual_length_ray(obs[depth - 1], obs[depth], q)
                            center = self.compute_virtual_center(obs[depth - 1], obs[depth])
                        maxVal = (obs[depth].width / 2) ** 2 + (obs[depth].height / 2) ** 2
                        if distance(q, center) < 1.0e-3:
                            distance_to_center = 1.0e3
                        else:
                            distance_to_center = distance(q, center)
                        v_i = rho_i * (1 + all_leaves_beta[i] / maxVal) / distance_to_center
                        all_v.append(v_i)
                else:
                    if self.check_parent_center_inside(obs[depth - 1], obs[depth]):
                        rho_i = compute_squicle_length_ray(obs[depth - 1].width / 2, obs[depth - 1].height / 2, q - obs[depth - 1].center, obs[depth - 1].theta, obs[depth - 1].s)
                        center = obs[depth - 1].center

                    else:
                        rho_i = self.compute_virtual_length_ray(obs[depth - 1], obs[depth], q)
                        center = self.compute_virtual_center(obs[depth - 1], obs[depth])
                    maxVal = (obs[depth].width / 2)**2 + (obs[depth].height / 2)**2
                    if distance(q, center) < 1e-3:
                        distance_to_center = 1e3
                    else:
                        distance_to_center = distance(q, center)
                    v_i = rho_i * (1 + all_leaves_beta[i] / maxVal) / distance_to_center
                    all_v.append(v_i)
                i += 1
        # print(all_v)
        return all_v
    
    def compute_all_f(self, q: np.ndarray, d: int) -> list:
        all_f = []
        all_v = self.compute_all_v(q, d)
        # print("all_v:", all_v)
        ws_all_obs = compute_all_obs(self.world)
        all_depth = self.compute_all_purging_depth(d)
        i = 0
        for k, obs in enumerate(ws_all_obs):
            depth = all_depth[k]
            if depth > 0:
                if obs[0].type == 'Workspace':
                    if depth == 1:
                        virtual_ws = self.compute_virtual_ws(obs[depth - 1], obs[depth])
                        center = virtual_ws.center
                    else:
                        if self.check_parent_center_inside(obs[depth - 1], obs[depth]):
                            center = obs[depth - 1].center
                        else:
                            center = self.compute_virtual_center(obs[depth - 1], obs[depth])
                else:
                    if self.check_parent_center_inside(obs[depth - 1], obs[depth]):
                        center = obs[depth - 1].center
                    else:
                        center = self.compute_virtual_center(obs[depth - 1], obs[depth])
                f_i = all_v[i] * (np.array(q) - np.array(center)) + np.array(center)
                all_f.append(f_i)
                i += 1
        return all_f
    
    def compute_f_mu(self, q: np.ndarray, d: int) -> np.ndarray:
        q = np.array(q)
        all_sigma = self.compute_all_sigma(q, d)
        # print(all_sigma)
        all_f = self.compute_all_f(q, d)
        # print("all_sigma", all_sigma)
        # print("q", q)
        f_mu = (1 - sum(all_sigma)) * q
        tree_num = self.compute_tree_num(d)
        for i in range(tree_num):
            f_mu += all_sigma[i] * all_f[i]
        return f_mu
    
    def compute_f(self, q: np.ndarray) -> np.ndarray:
        max_depth = max(self.all_depth)
        f = q
        for i in range(0, max_depth):
            f = self.compute_f_mu(f, i)
            # print(f)
        return f
        
class StarToSphere(object):
    # transformation from star world to sphere world
    def __init__(self, world, goal, nf_lambda):
        self.world = world
        self.goal = goal
        self.nf_lambda = nf_lambda
        
    def compute_h_lambda(self, q: np.ndarray) -> np.ndarray:
        all_s = self.compute_all_s(q)
        h_lambda = (1 - sum(all_s)) * q
        all_beta = self.compute_all_beta(q)
        ws_all_obs = self.compute_all_root_obs()
        for i, ws_or_obs in enumerate(ws_all_obs):
            h_lambda += all_s[i] * ws_or_obs.compute_T(q, all_beta[i])
        return h_lambda
    
    def compute_all_s(self, q: np.array) -> list:
        all_s = []
        all_beta = self.compute_all_beta(q)
        all_beta_bar = self.compute_all_beta_bar(q)
        for i in range(0, len(all_beta)):
            s_i = compute_gamma(q, self.goal) * all_beta_bar[i] / (compute_gamma(q, self.goal) * all_beta_bar[i] +
                                                                   self.nf_lambda * all_beta[i])
            all_s.append(s_i)
        return all_s

    # def compute_all_s(self, q: np.array) -> list:
    #     all_s = []
    #     all_beta = self.compute_all_beta(q)
    #     for i in range(0, len(all_beta)):
    #         s_i = 1.0
    #         all_s.append(s_i)
    #     return all_s
    
    def compute_all_beta_bar(self, q: np.ndarray) -> list:
        all_beta_bar = []
        all_beta = self.compute_all_beta(q)
        for i in range(0, len(all_beta)):
            beta_bar_i = 1.0
            for j in range(0, len(all_beta)):
                if j != i:
                    beta_bar_i *= all_beta[j]
            all_beta_bar.append(beta_bar_i)
        return all_beta_bar

    # def compute_all_beta_bar(self, q: np.ndarray) -> list:
    #     all_beta_bar = []
    #     all_beta = self.compute_all_beta(q)
    #     for i in range(0, len(all_beta)):
    #         beta_bar_i = 1.0
    #         all_beta_bar.append(beta_bar_i)
    #     return all_beta_bar
    
    def compute_all_beta(self, q: np.ndarray) -> list:
        all_beta = []
        ws_all_obs = self.compute_all_root_obs()
        for ws_or_obs in ws_all_obs:
            all_beta.append(ws_or_obs.potential(q))
        return all_beta

    def compute_all_root_obs(self) -> list:
        all_root_obs = []
        ws_all_obs = compute_all_obs(self.world)
        all_root_obs.append(ws_all_obs[0][0])
        for ws_or_obs in ws_all_obs[1:]:
            all_root_obs.append(ws_or_obs[0])
        return all_root_obs


class SphereToPoint(object):
    # transformation from sphere world to point world   
    def __init__(self, world, goal):
        self.world = world
        self.goal = goal
        self.M = len(self.world.obstacles)
    
    def bounded_pw_to_unbounded(self, q: np.ndarray, margin=1e-5, bound=1e5) -> np.ndarray:
        center = self.world.workspace[0][0].center
        ws_radius = self.world.workspace[0][0].radius
        dist_to_ws_center = distance(q, center)
        if dist_to_ws_center > ws_radius - margin:
            weight = bound
        else:
            weight = ws_radius / (ws_radius - dist_to_ws_center)
        return (q - center) * weight + center
    
    def compute_T_q(self, q: np.ndarray) -> np.ndarray:
        T = q
        for obs in self.world.obstacles:
            T += self.compute_T_q_i(q, obs[0])
        return T
    
    def compute_T_q_i(self, q: np.ndarray, obs) -> np.ndarray:
        center, radius = obs.center, obs.radius
        return (1 - self.compute_s_delta(self.compute_b(q, center, radius), self.mu)) * (center - q)
    
    def compute_s_delta(self, x: float, delta: float) -> float:
        return (x / delta) * (1 - self.compute_eta(x, delta)) + self.compute_eta(x, delta)
    
    def compute_eta(self, x: float, delta: float) -> float:
        return self.compute_sigma(x) / (self.compute_sigma(x) + self.compute_sigma(delta - x))

    def compute_sigma(self, x: float) -> float:
        if x <= 0:
            return 0.0
        else:
            return exp(-1.0 / x)

    def compute_b(self, q: np.ndarray, center: np.ndarray, radius: float) -> float:
        return distance(q, center) - radius
    
    @property
    def mu_a(self) -> float:
        if len(self.world.obstacles) == 1:
            return self.mu_0
        min_mu_a = distance(self.world.obstacles[0][0].center, self.world.obstacles[1][0].center) - (
            self.world.obstacles[0][0].radius + self.world.obstacles[1][0].radius)
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    if distance(self.world.obstacles[i][0].center, self.world.obstacles[j][0].center) - (self.world.obstacles[i][0].radius + self.world.obstacles[j][0].radius) < min_mu_a:
                        min_mu_a = distance(self.world.obstacles[i][0].center, self.world.obstacles[j][0].center) - (
                            self.world.obstacles[i][0].radius + self.world.obstacles[j][0].radius)
        if min_mu_a < 0.01:
            min_mu_a = 0.01
        return min_mu_a

    @property
    def mu_0(self) -> float:
        min_mu_0 = self.world.workspace[0][0].radius - \
            self.world.obstacles[0][0].radius - \
            distance(self.world.obstacles[0][0].center, self.world.workspace[0][0].center)
        for i in range(0, len(self.world.obstacles)):
            if self.world.workspace[0][0].radius - self.world.obstacles[i][0].radius - distance(self.world.obstacles[i][0].center, self.world.workspace[0][0].center) < min_mu_0:
                min_mu_0 = self.world.workspace[0][0].radius - \
                    self.world.obstacles[i][0].radius - \
                    distance(
                        self.world.obstacles[i][0].center, self.world.workspace[0][0].center)
        return min_mu_0

    @property
    def mu_d(self) -> float:
        min_mu_d = distance(self.world.obstacles[0][0].center, self.goal) - self.world.obstacles[0][0].radius
        for i in range(self.M):
            if distance(self.world.obstacles[i][0].center, self.goal) - self.world.obstacles[i][0].radius < min_mu_d:
                min_mu_d = distance(self.world.obstacles[i][0].center, self.goal) - self.world.obstacles[i][0].radius
        return min_mu_d

    @property
    def mu(self) -> float:
        # print("self.mu_a", self.mu_a)
        # min_mu = min(self.mu_a, 2 * self.mu_0, 2*self.mu_d)
        min_mu = 0.02
        return min_mu / 2
        




"""
-----transformation gradient functions-----
"""
class TransformationGradient(StarToSphere, SphereToPoint):
    # calculate the gradient for the transformation
    def __init__(self, world, goal, nf_lambda):
        super().__init__(world, goal, nf_lambda)
        self.M = len(self.world.obstacles)
        
    def grad_tf_world_to_pw(self, q: np.ndarray) -> np.ndarray:
        return np.dot(self.grad_bounded_pw_to_unbounded(self.compute_T_q(self.compute_h_lambda(q))), self.grad_bounded_pw(q))
    
    def grad_bounded_pw_to_unbounded(self, q: np.ndarray, margin=1e-10, bound=1e20) -> np.ndarray:
        center =  self.world.workspace[0].center
        world_radius = self.world.workspace[0].radius
        dist_to_world_center = distance(q, center)
        if dist_to_world_center > world_radius - margin:
            return bound * np.array([[1, 0], [0, 1]])
        else:
            return world_radius * ((world_radius - distance(q, center)) * np.array([[1, 0], [0, 1]]) +
                    np.dot(np.array([[(q - center)[0]], [(q - center)[1]]]),
                           np.array([[(q - center)[0], (q - center)[1]]])) /
                    distance(q, center)) / (world_radius -distance(q, center))**2
    
    def grad_bounded_pw(self, q: np.ndarray) -> np.ndarray:
        return np.dot(self.grad_T_q(self.compute_h_lambda(q)), self.grad_h_lambda(q))
                 
    def grad_h_lambda(self, q: np.ndarray) -> np.ndarray:
        all_s = self.compute_all_s(q)
        all_grad_s = self.grad_all_s(q)
        all_grad_T = self.grad_all_T(q)
        grad_h = np.dot(np.array([[q[0]], [q[1]]]), np.array([[-sum(all_grad_s)[0],
                                                               -sum(all_grad_s)[1]]])) + (1-sum(all_s)) * np.array([[1, 0], [0, 1]])
        all_beta = self.compute_all_beta(q)
        ws_all_obs = compute_all_obs(self.world)
        for i, ws_or_obs in enumerate(ws_all_obs):
            grad_h += np.dot(np.array([[ws_or_obs[0].compute_T(q, all_beta[i])[0]], [ws_or_obs[0].compute_T(q, all_beta[i])[1]]]),
                             np.array([[all_grad_s[i][0], all_grad_s[i][1]]])) + all_s[i] * all_grad_T[i]
        return grad_h
    
    def grad_T_q(self, q: np.ndarray) -> np.ndarray:
        grad_T = np.array([[1.0, 0.0], [0.0, 1.0]])
        for obs in self.world.obstacles:
            grad_T += self.grad_T_q_i(q, obs[0])
        return grad_T
    
    def grad_T_q_i(self, q: np.ndarray, obs) -> np.ndarray:
        return np.dot(np.array([[(obs.center - q)[0]], [(obs.center - q)[1]]]), 
                      -self.grad_s_delta(self.compute_b(q, obs.center, obs.radius), self.mu) * np.array([[self.grad_b(q, obs.center)[0],
                                         self.grad_b(q, obs.center)[1]]])) + (1 - self.compute_s_delta(self.compute_b(q, obs.center, obs.radius), self.mu)) * np.array([[-1, 0], [0, -1]])

    def grad_all_beta(self, q: np.ndarray) -> list:
        all_grad_beta = []
        ws_all_obs = compute_all_obs(self.world)
        for ws_or_obs in ws_all_obs:
            all_grad_beta.append(ws_or_obs[0].grad_potential(q))
        return all_grad_beta

    def grad_all_beta_bar(self, q: np.ndarray) -> list:
        all_grad_beta_bar = []
        for k in range(self.M + 1):
            all_beta = self.compute_all_beta(q)
            all_grad_beta = self.grad_all_beta(q)
            all_beta.pop(k)
            all_grad_beta.pop(k)
            grad_beta_bar = np.zeros(2)
            for i in range(len(all_beta)):
                product = all_grad_beta[i]
                for j in range(len(all_beta)):
                    if j != i:
                        product *= all_beta[j]
                grad_beta_bar += product
            all_grad_beta_bar.append(grad_beta_bar)
        return all_grad_beta_bar
    
    def grad_all_s(self, q: np.ndarray) -> list:
        all_grad_s = []
        all_beta = self.compute_all_beta(q)
        all_beta_bar = self.compute_all_beta_bar(q)
        all_grad_beta = self.grad_all_beta(q)
        all_grad_beta_bar = self.grad_all_beta_bar(q)
        for i in range(len(all_beta)):
            grad_s_i = ((grad_gamma(q, self.goal) * all_beta_bar[i] + compute_gamma(q, self.goal) * all_grad_beta_bar[i]) *
                        (compute_gamma(q, self.goal) * all_beta_bar[i] + self.nf_lambda * all_beta[i]) - (compute_gamma(q, self.goal) * all_beta_bar[i]) *
                        (grad_gamma(q, self.goal) * all_beta_bar[i] + compute_gamma(q, self.goal) * all_grad_beta_bar[i] +
                        self.nf_lambda * all_grad_beta[i]))/(compute_gamma(q, self.goal) * all_beta_bar[i] + self.nf_lambda * all_beta[i])**2
            all_grad_s.append(grad_s_i)
        return all_grad_s

    def grad_s_d(self, q: np.ndarray) -> np.ndarray:
        grad_s = 0
        for i in range(0, self.M + 1):
            grad_s += self.grad_s(q, i)
        return -grad_s

    def grad_all_v(self, q: np.ndarray) -> list:
        all_grad_v = []
        all_beta = self.compute_all_beta(q)
        all_grad_beta = self.grad_all_beta(q)
        ws_all_obs = compute_all_obs(self.world)

        for i, ws_or_obs in enumerate(ws_all_obs):
            ws_or_obs = ws_or_obs[0]
            if i == 0:
                grad_v_i = ws_or_obs.radius * (-all_grad_beta[i] * distance(q, ws_or_obs.center) -
                                          (1 - all_beta[i]) * (q - ws_or_obs.center) / distance(q, ws_or_obs.center)) / distance(q, ws_or_obs.center)**2
            else:
                grad_v_i = ws_or_obs.radius * (all_grad_beta[i] * distance(q, ws_or_obs.center) -
                                          (1 + all_beta[i]) * (q - ws_or_obs.center) / distance(q, ws_or_obs.center)) / distance(q, ws_or_obs.center)**2
            all_grad_v.append(grad_v_i)
        return all_grad_v

    def grad_all_T(self, q: np.ndarray) -> list:
        all_grad_T = []
        all_grad_v = self.grad_all_v(q)
        all_beta = self.compute_all_beta(q)
        ws_all_obs = compute_all_obs(self.world)
        for i, ws_or_obs in enumerate(ws_all_obs):
            grad_T_i = np.dot(np.array([[(q - ws_or_obs[0].center)[0]], [(q - ws_or_obs[0].center)[1]]]),
                              np.array([[all_grad_v[i][0], all_grad_v[i][1]]])) + ws_or_obs[0].compute_v(q, all_beta[i]) * np.array([[1, 0], [0, 1]])
            all_grad_T.append(grad_T_i)
        return all_grad_T

    def grad_b(self, q: np.ndarray, center: np.ndarray) -> np.ndarray:
        return (q - center) / distance(q, center)
    
    def grad_sigma(self, x: float) -> float:
        if x <= 0:
            return 0.0
        else:
            return exp(- 1 / x) / x**2

    def grad_eta(self, x: float, delta: float) -> float:
        return (self.grad_sigma(x) * (self.compute_sigma(x) + self.compute_sigma(delta - x)) - self.compute_sigma(x) * (self.grad_sigma(x) -
                self.grad_sigma(delta - x))) / (self.compute_sigma(x) + self.compute_sigma(delta - x))**2

    def grad_s_delta(self, x: float, delta: float) -> float:
        return (1 / delta) * (1 - self.compute_eta(x, delta)) + (x / delta) * (-self.grad_eta(x, delta)) + self.grad_eta(x, delta)
    
    