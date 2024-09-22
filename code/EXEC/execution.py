import math
import os
import sys

sys.path.append(os.getcwd())

import numpy as np

from NF.controller import NFController
from ENV.construct_forest import ConstructForest
from NM.delaunay import DelaunayTriangulation
from NM.navigation_map import NavigationMap
from ENV.point_to_line import PointToLine
from ENV.cluster import SklearnCluster
from ENV.cluster_split import ClusterSplit
from ENV.squircle_estimation import SquircleEstimation
from ROBOT.robot import Robot


class Execution(object):
    def __init__(self, robot, forest_world, dt=0.2):
        self.robot = robot
        self.forest_world = forest_world
        self.dt = dt

        self.sk_clu = SklearnCluster(eps=0.12, min_samples=6)
        self.cluster_split = ClusterSplit(window_size=5, curvature_threshold=10)
        self.sq_esti = SquircleEstimation(robot)
        self.navigation_map = NavigationMap()

        self.current_step = 0
        self.delaunay = None
        self.all_line_list = None
        self.all_squircles = None
        self.construct_forest = None
        self.current_goal = None
        self.trajectory = [[self.robot.pose[0]], [self.robot.pose[1]]]
        self.all_polygon_list = None
        self.goal_index = 0
        self.grid_map = None
        self.path = None

    def one_step_forward(self, current_step):
        lambda_ = 1e3
        mu_ = [1e10, 1e8, 1e6, 1e4, 1e2, 1e1]

        self.robot.get_measurements_update_world()

        self.current_step = current_step

        self.robot.step = self.current_step

        cluster_points = self.sk_clu.cluster(np.array(self.forest_world.global_lidar_points))
        if len(cluster_points) == 0:
            self.construct_forest = ConstructForest([])
            self.robot_move_one_step(lambda_, mu_)
            return
        sorted_cluster_points = self.sk_clu.sort_cluster_points(cluster_points)
        self.forest_world.cluster_points = sorted_cluster_points
        all_cluster_segments = self.cluster_split.split_all_cluster(self.forest_world.cluster_points)
        self.forest_world.all_cluster_segments = all_cluster_segments
        # print("num segments", [len(segment)-1 for segment in all_cluster_segments])

        squircle_data = self.sq_esti.fit_squircle_group(self.forest_world.all_cluster_segments)
        self.forest_world.squircle_data = squircle_data
        self.construct_forest = ConstructForest(self.forest_world.squircle_data)

        self.robot_move_one_step(lambda_, mu_)

    def robot_move_one_step(self, lambda_, mu_):

        self.set_robot_goal(threshold=0.5)

        # print("path", self.delaunay.path)
        print("********Robot pose: " + str(self.robot.pose) + " ********")
        print("********Move to goal: " + str(self.current_goal) + " ********")

        nf_lambda = lambda_
        nf_mu = mu_

        nf_controller = NFController(self.construct_forest.forest_world,
            np.array(self.robot.pose), np.array(self.current_goal),
            nf_lambda, nf_mu)

        v, omega = nf_controller.vector_follow_controller()
        print("********Control Input: " + str(v) + " " + str(omega) + " ********")
        new_pose = [self.robot.pose[0] + self.dt * v * np.cos(self.robot.pose[2]),
                    self.robot.pose[1] + self.dt * v * np.sin(self.robot.pose[2]),
                    self.robot.pose[2] + self.dt * omega]
        new_pose[2] = (new_pose[2] + np.pi) % (2 * np.pi) - np.pi

        self.robot.pose = new_pose
        self.trajectory[0].append(self.robot.pose[0])
        self.trajectory[1].append(self.robot.pose[1])

    def set_robot_goal(self, threshold=1.0):
        all_polygon_list = self.construct_forest.get_vis_rect_data(inflated_size=0.16)
        self.all_polygon_list = all_polygon_list

        robot_pose = self.robot.pose

        self.navigation_map = NavigationMap()
        self.navigation_map.construct_planner_rect_multi_goal(robot_pose, self.robot.goal_list[self.goal_index:], self.all_polygon_list)
        self.path = self.navigation_map.path
        print("navigation path", self.path)
        next_path_index = 1
        current_index = 1
        next_path_goal = self.path[next_path_index]

        if math.sqrt((self.robot.goal_list[self.goal_index][0] - self.robot.pose[0]) ** 2 + (self.robot.goal_list[self.goal_index][1] - self.robot.pose[1]) ** 2) < 0.5:
            self.current_goal = self.robot.goal_list[self.goal_index]
            return

        for path_goal in self.path[next_path_index:]:
            dis_to_goal = math.sqrt(
                (path_goal[0] - self.robot.pose[0]) ** 2 + (path_goal[1] - self.robot.pose[1]) ** 2)
            if dis_to_goal > 0.3:
                next_path_goal = path_goal
                break

        dis_to_goal = math.sqrt(
            (next_path_goal[0] - self.robot.pose[0]) ** 2 + (next_path_goal[1] - self.robot.pose[1]) ** 2)
        if dis_to_goal < threshold:
            self.current_goal = np.array([next_path_goal[0], next_path_goal[1], next_path_goal[2]])
        else:
            # new_theta = next_path_goal[2]
            new_theta = np.arctan2(next_path_goal[1] - self.robot.pose[1], next_path_goal[0] - self.robot.pose[0])
            if current_index == len(self.path) - 1:
                new_theta = np.arctan2(next_path_goal[1] - self.robot.pose[1], next_path_goal[0] - self.robot.pose[0])
            if abs(next_path_goal[0] - self.robot.pose[0]) < 1e-5:
                next_path_goal = [self.robot.pose[0],
                                  self.robot.pose[1] + threshold * (next_path_goal[1] - self.robot.pose[1]) / dis_to_goal,
                                  new_theta]
            elif abs(next_path_goal[1] - self.robot.pose[1]) < 1e-5:
                next_path_goal = [self.robot.pose[0] + threshold * (next_path_goal[0] - self.robot.pose[0]) / dis_to_goal,
                                  self.robot.pose[1],
                                  new_theta]
            else:
                next_path_goal = [self.robot.pose[0] + threshold * (next_path_goal[0] - self.robot.pose[0]) / dis_to_goal,
                                  self.robot.pose[1] + threshold * (next_path_goal[1] - self.robot.pose[1]) / dis_to_goal,
                                  new_theta]
            self.current_goal = np.array([next_path_goal[0], next_path_goal[1], next_path_goal[2]])

    def test_one_step(self, current_step):
        self.current_step = current_step

        self.robot.step = self.current_step
        self.point_to_line.update_line()

        all_line_list = self.robot_world.all_line_list
        self.all_line_list = all_line_list

        self.construct_forest = ConstructForest(all_line_list)

        all_polygon_list = self.construct_forest.get_vis_rect_data(inflated_size=0.16)
        self.all_polygon_list = all_polygon_list
        print("all_polygon_list", all_polygon_list)

        # round_num = 5
        # for i, polygon_i in enumerate(all_polygon_list):
        #     for j, vertex_j in enumerate(polygon_i):
        #         all_polygon_list[i][j] = np.round(list(all_polygon_list[i][j]), round_num)
        #
        # robot_pose = [0.0, 0.0, 0.0]
        # goal_pose = [0.0, 0.0, 0.0]
        # for i, coord_i in enumerate(self.robot.pose):
        #     robot_pose[i] = np.round(self.robot.pose[i], round_num)
        #
        # for i, coord_i in enumerate(self.robot.goal):
        #     goal_pose[i] = np.round(self.robot.goal[i], round_num)

        robot_pose = self.robot.pose
        goal_pose = self.robot.goal

        self.navigation_map = NavigationMap()
        self.navigation_map.construct_planner_rect(robot_pose, goal_pose, all_polygon_list)
        path = self.navigation_map.path
        print("navigation path", path)

        all_squircles = []
        for ws_group in self.construct_forest.forest_world.workspace:
            if len(ws_group) == 1:
                continue
            else:
                all_squircles += ws_group[1:]
        for obs_group in self.construct_forest.forest_world.obstacles:
            all_squircles += obs_group
        self.all_squircles = all_squircles

        print("forest world workspace structure:", len(self.construct_forest.forest_world.workspace))
        print("forest world obstacles structure", len(self.construct_forest.forest_world.obstacles))

        self.robot.move_one_step()

        self.trajectory[0].append(self.robot.pose[0])
        self.trajectory[1].append(self.robot.pose[1])

        self.robot.get_measurements_update_world()
