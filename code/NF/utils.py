import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import math

# some general functions

def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compute_gamma(q: np.ndarray, goal: np.ndarray) -> float:
    return distance(q, goal)**2

def grad_gamma(q: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return 2 * (q - goal)

def compute_all_obs(world) -> list:
    all_ws = world.workspace.copy()
    all_obs = world.obstacles.copy()
    ws_all_obs = all_ws + all_obs
    return ws_all_obs

def compute_squicle_length_ray(width, height, q, theta, s):
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    q = rotation @ q
    normalized_q = q / np.linalg.norm(q)
    transformed_q = np.array([normalized_q[0] / width, normalized_q[1] / height])
    normalized_transformed_q = transformed_q / np.linalg.norm(transformed_q)
    scale = math.sqrt((normalized_transformed_q[0] * width) ** 2 + (normalized_transformed_q[1] * height) ** 2)
    rho_q = scale * math.sqrt(2 / (1 + math.sqrt(1 - 4 * s**2 * (normalized_transformed_q[0] * normalized_transformed_q[1])**2)))
    return rho_q

def compute_squicle_length_ray_plt(q, width, height, theta, s):
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    q = rotation @ q
    normalized_q = q / np.linalg.norm(q)
    transformed_q = np.array([normalized_q[0] / width, normalized_q[1] / height])
    normalized_transformed_q = transformed_q / np.linalg.norm(transformed_q)
    scale = math.sqrt((normalized_transformed_q[0] * width) ** 2 + (normalized_transformed_q[1] * height) ** 2)
    rho_q = scale * math.sqrt(2 / (1 + math.sqrt(1 - 4 * s**2 * (normalized_transformed_q[0] * normalized_transformed_q[1])**2)))
    return rho_q