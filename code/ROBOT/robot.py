import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import yaml
import math
from scipy.spatial import distance_matrix

from ROBOT.lidar import Lidar
from ROBOT.utils import distance

from ENV.geometry import RealWorld
from NF.controller import NFController



class Robot(object):
    def __init__(self, pose, real_world, init_world, robot_config='robot_config.yaml'):
        self.start = None
        self.goal_list = None
        self.goal = None
        self._config = None
        self._name = None
        self._name = None
        self._type = None
        self._size = None
        self._lidar = None
        self._real_world = real_world
        self._init_world = init_world
        self.load_config(robot_config)
        self._lidar_points = []
        self._lidar_obs_points = []
        self._log = []
        self.pose = pose
        self.get_measurements_update_world()
        self.step = 0

        self.waypoints = [np.array([1.35, 1.8, 1.57]), np.array([1.4, 2.4, 0.8]), np.array([2.2, 2.5, 0.0]),
                          np.array([3.0, 2.5, 0.0]),
                          np.array([4.5, 2.4, 0.0])]
        self.current_goal_index = 0
        self.current_goal = self.waypoints[self.current_goal_index]
        self.pre_path = None

    def load_config(self, robot_config):
        with open(robot_config, 'r') as stream:
            try:
                self._config = yaml.safe_load(stream)
                self.construct_robot()
            except yaml.YAMLError as exc:
                print(exc)

    @property
    def config(self) -> dict:
        return self._config

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return self._size

    @property
    def real_world(self):
        return self._real_world

    @property
    def init_world(self):
        return self._init_world

    @property
    def lidar_points(self):
        return self._lidar_points

    @property
    def lidar_obs_points(self):
        return self._lidar_obs_points

    @property
    def lidar(self):
        return self._lidar

    @property
    def log(self):
        return self._log

    def set_pre_path(self, path):
        self.pre_path = path

    def set_goal(self, goal):
        self.goal = goal

    def set_start(self, start):
        self.start = start

    def set_goal_list(self, goal_list):
        self.goal_list = goal_list

    def move_one_step(self, current_frame):
        self.step = current_frame
        current_step = self.step
        next_point = self.pre_path[current_step]
        direction = [(next_point[0] - self.pose[0]), (next_point[1] - self.pose[1])]
        angle = math.atan2(direction[1], direction[0])
        self.pose = np.array([next_point[0], next_point[1], angle])

    def construct_robot(self):
        self._name = self.config['name']
        self._type = self.config['type']
        self._size = self.config['size']
        self._lidar = Lidar(**self.config['lidar'])

    def get_measurements_update_world(self):
        absolute_points = self.lidar.get_measurements(
            self.pose, self.real_world)
        obs_points = self.lidar.get_measurements(
            self.pose, self.real_world, obs=True)
        # self._lidar_obs_points = obs_points
        self._lidar_points = absolute_points

        global_lidar_points = []
        for ws in self.real_world.workspace:
            if len(ws) == 1:
                continue
            ws_1 = ws[1]
            global_lidar_points += ws_1.accumulated_local_points
        for obs in self.real_world.obstacles:
            obs_0 = obs[0]
            global_lidar_points += obs_0.accumulated_local_points
        points_2d = np.array(global_lidar_points)
        global_lidar_points = self.remove_close_points(points_2d, threshold=0.004)
        # _, indices = np.unique(points_2d, axis=0, return_index=True)
        # unique_global_lidar_points = np.array(global_lidar_points)[indices]
        # global_lidar_points = unique_global_lidar_points.tolist()
        # print("global_lidar_points", global_lidar_points)
        self.real_world.global_lidar_points = global_lidar_points

    @staticmethod
    def remove_close_points(points, threshold):
        points_array = np.array(points)
        dist_matrix = distance_matrix(points_array[:, :2], points_array[:, :2])

        keep_indices = []
        processed = set()

        for i in range(len(points)):
            if i not in processed: 
                keep_indices.append(i)
            close_points = np.where(dist_matrix[i] < threshold)[0]
            processed.update(close_points)
        return [points[i] for i in keep_indices]

    def control_input(self, control_input):
        x, y, theta = self.pose[:]
        linear_v = 1.0
        angular_v = 3.0 * (np.arctan2(control_input[1], control_input[0]) - theta)
        return linear_v, angular_v

    def check_goal_reached(self, goal, threshold=0.2):
        if distance(self.pose[0:2], goal[0:2]) < threshold:
            return True
        return False

    def find_path(self, start_pose, goal_pose, real_world):
        # # goal_pose = np.array([4.2, 3.2, 0.0])
        # # start_pose = np.array([1.8, 1.8, 1.57])
        #
        # robot_radius = self.size  # [m]
        # # print("radius",robot_radius)
        #
        # # set obstacle positions
        # # for robot, the whole map is an unknown area
        #
        # a_star = AStarPlanner( 1, robot_radius)
        # path = a_star.planning(start_pose, goal_pose, real_world)
        # return path

        a = AStar()
        a.init_grid(7 * 10, 4 * 10, real_world)
        a.caculate_one_way((start_pose[0] * 10, start_pose[1] * 10), (goal_pose[0] * 10, goal_pose[1] * 10))
        path = a.solve()
        path = list(path)
        # print("path",path)
        for i in range(0, len(path)):
            path[i][0] = path[i][0] / 10
            path[i][1] = path[i][1] / 10
        # print(path)
        return path

    def multi_goal_find_path(self, start_pose, goal_pose_list, real_world):
        path = list()
        for i in range(0, len(goal_pose_list)):
            if i == 0:
                path_ = self.find_path(start_pose, goal_pose_list[i], real_world)
            else:
                path_ = self.find_path(goal_pose_list[i - 1], goal_pose_list[i], real_world)
                path_ = path_[1:]
            path.extend(path_)
        return path

    def move_to_goal(self, robot_world):
        if self.check_goal_reached(self.current_goal):
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                self.current_goal_index -= 1
            self.current_goal = self.waypoints[self.current_goal_index]
        print("current_goal", self.current_goal)
        nf_controller = NFController(robot_world.forest_world, self.pose, self.current_goal, nf_lambda=100,
                                     nf_mu=[1e5, 1e10])
        vel, yaw = nf_controller.gradient_follow_controller()
        if self.check_goal_reached(self.waypoints[-1]):
            vel = 0.0
            yaw = 0.0
            print("GOAL REACHED!!!")
        # else: control = (self.current_goal[0:2] - self.pose[0:2]) / np.linalg.norm(self.current_goal[0:2] -
        # self.pose[0:2]) * 0.05 self.pose[0:2] = self.pose[0:2] + control
        self.pose[0] = self.pose[0] + vel * np.cos(self.pose[2]) * 0.1
        self.pose[1] = self.pose[1] + vel * np.sin(self.pose[2]) * 0.1
        self.pose[2] = self.pose[2] + yaw * 0.1
        self.get_measurements_update_world(self.pose)

    def move_to_point(self, point):
        direction = [(point[0] - self.pose[0]), (point[1] - self.pose[1])]
        angle = math.atan2(direction[1], direction[0])
        self.pose = np.array([point[0], point[1], angle])
        self.get_measurements_update_world(self.pose)

    #
    # def move_and_sense_until_goal(self, time_step=0.01, max_steps=200, testing=True):
    #     print('NF navigation starts!')
    #     k = 0
    #     while (not self.check_goal_reached()) and (k <= max_steps):
    #         if self.get_measurements_update_world():
    #             self.compute_nf()
    #             print('Workspace updated at %.2f' % float(k * time_step))
    #         grad = - self.nf.grad_potential_at_point(self.pose[0:2])[0]
    #         linear_v, angular_v = self.compute_control(grad)
    #         self.change_pose(linear_v, angular_v, time_step)
    #         # print('--------', k, '------------')
    #         # print('pose', self.pose)
    #         # print('control', linear_v, angular_v)
    #         # print('grad', grad[0], grad[1], np.arctan2(grad[1], grad[0]))
    #         # for testing:
    #         if testing:
    #             self._pose[2] = np.arctan2(grad[1], grad[0])
    #         self._log.append([k * time_step, np.copy(self.pose), np.copy(self.lidar_points),
    #                           np.array([linear_v, angular_v]), np.array(grad),
    #                           len(self.world_model.obstacles)])
    #         k += 1
    #     if self.check_goal_reached():
    #         print('Goal reached after %.2f s' % float(k * time_step))
    #     else:
    #         print('Max step %d reached' % max_steps)
    #
    # def change_pose(self, linear_v, angular_v, time_step):
    #     theta = self.pose[2]
    #     self._pose += np.array([linear_v * np.cos(theta),
    #                                linear_v * np.sin(theta),
    #                                angular_v]) * time_step
    #
    #
    #
    #
    #
    # def rrt_navigation(self):
    #     rrt = RRT(self.world_model)
    #     start_point = list(self.pose)
    #     goal_point = list(self.goal)
    #     explored, final_path , final_pose_collection = rrt.rrt_star(start_point, goal_point)
    #     total_traj = rrt.get_branches_traj()
    #     return explored, final_path , rrt.nodes , total_traj , final_pose_collection
    #
    #
    # def rrt_move_and_sense_until_goal(self, time_step=0.01, max_steps=200, testing=True):
    #     print('RRT navigation starts!')
    #     k = 0
    #     q = 0
    #     _ , _ , now_all_nodes, now_total_traj , final_pose_collection = self.rrt_navigation()
    #     while (not self.check_goal_reached()) and (k <= max_steps):
    #         if self.get_measurements_update_world():
    #             _ , _ , now_all_nodes, now_total_traj , final_pose_collection = self.rrt_navigation()
    #             q = 0
    #             print('Workspace updated at %.2f' % float(k * time_step))
    #         [linear_v, angular_v] = np.array(final_pose_collection[q][3])
    #         self._pose = np.array(final_pose_collection[q][0:3])
    #         # for testing:
    #         # if testing:
    #             # self._pose[2] = np.arctan2(grad[1], grad[0])
    #         self._log.append([k * time_step, np.copy(self.pose), np.copy(self.lidar_points),[linear_v, angular_v],
    #                           len(self.world_model.obstacles) , now_all_nodes , now_total_traj ])
    #         k += 1
    #         q += 1
    #     if self.check_goal_reached():
    #         print('Goal reached after %.2f s' % float(k * time_step))
    #     else:
    #         print('Max step %d reached' % max_steps)
    #
    #
    # def rrt_mix_nf_navigation(self):
    #     rrt = RRT_mix_NF(self.world_model)
    #     start_point = list(self.pose)
    #     goal_point = list(self.goal)
    #     # q =rrt.rrt_nf_star(start_point, goal_point)
    #     # print(q)
    #     explored, final_path , final_pose_collection = rrt.rrt_nf_star(start_point, goal_point)
    #     print("static_RRT")
    #     total_traj = rrt.get_branches_traj()
    #     return explored, final_path , rrt.nodes , total_traj , final_pose_collection
    #
    #
    #
    #
    # def rrt_mix_nf_move_and_sense_until_goal(self, time_step=0.01, max_steps=200, testing=True):
    #     print('RRT navigation starts!')
    #     k = 0
    #     q = 0
    #     _ , _ , now_all_nodes, now_total_traj , final_pose_collection = self.rrt_mix_nf_navigation()
    #     while (not self.check_goal_reached()) and (k <= max_steps):
    #         if self.get_measurements_update_world():
    #             _ , _ , now_all_nodes, now_total_traj , final_pose_collection = self.rrt_mix_nf_navigation()
    #             q = 0
    #             print('Workspace updated at %.2f' % float(k * time_step))
    #         [linear_v, angular_v] = np.array(final_pose_collection[q][3])
    #         self._pose = np.array(final_pose_collection[q][0:3])
    #         # for testing:
    #         # if testing:
    #             # self._pose[2] = np.arctan2(grad[1], grad[0])
    #         self._log.append([k * time_step, np.copy(self.pose), np.copy(self.lidar_points),[linear_v, angular_v],
    #                           len(self.world_model.obstacles) , now_all_nodes , now_total_traj ])
    #         k += 1
    #         q += 1
    #     if self.check_goal_reached():
    #         print('Goal reached after %.2f s' % float(k * time_step))
    #     else:
    #         print('Max step %d reached' % max_steps)
