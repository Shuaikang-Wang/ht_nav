import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# from skimage.draw import line
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import random
import math
import copy

from NF.vis import plot_world, plot_task_area, plot_squircle, plot_init_world, plot_inflate_world

from ENV.construct_forest import ConstructForest, ForestWorld
from ENV.geometry import Squircle
from ENV.line_to_squircle import LineToSquircle
from ENV.b_spline import KBSpline
from NF.navigation import NavigationFunction
from ENV.construct_forest import ForestWorld

from ROBOT.vis import plot_robot_in_world, plot_point_in_inflated_world


def plot_data(ax, robot, world):
    plot_world(ax, world)
    plot_robot_in_world(ax, robot)
    plot_task_area(ax)
    plot_point(ax, world)


def plot_init_data(ax, robot, world):
    plot_init_world(ax, world)
    plot_robot_in_world(ax, robot)
    plot_task_area(ax)


def plot_cluster(ax, world, COLORS):
    for points, col in zip(world.cluster_points, COLORS):
            xy = points
            # ax.plot(xy[0, 0], xy[0, 1], 'o', markerfacecolor='b', markeredgecolor='b', markersize=6,
            #         zorder=35)
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), markersize=3, zorder=35)


def plot_polygon_list(ax, polygon_list):
    for polygon in polygon_list:
        # print("polygon", polygon)
        poly = Polygon(polygon, edgecolor='k', linestyle='-', linewidth=2.5, fill=False)
        ax.add_patch(poly)


def plot_fitting_squircle(ax, world):
    squircle_group = world.squircle_data
    for squircle in squircle_group:
        plot_squircle(ax, squircle[0], squircle[1] / 2, squircle[2] / 2, squircle[3], squircle[4],
                        line_color = 'b', line_style = '--', line_width = 2.5, plot_z_order = 40)


def plot_cluster_segment(ax, world):
    all_cluster_segments = world.all_cluster_segments
    for cluster_segments_with_s in all_cluster_segments:
        cluster_segments = cluster_segments_with_s[:-1]
        for segment in cluster_segments:
            segment = np.array(segment)
            ax.plot(segment[-1, 0], segment[-1, 1], 'o', color='r', markersize=6, zorder=36)

def plot_fitting_world(ax, real_world, world):
    extention = 0.1
    workspace = []
    obstacles = []
    for ws_group in world.workspace:
        ori_ws_group = [real_world.workspace[0][0]]
        for ws_i in ws_group[1:]:
            ori_squircle = Squircle('Rectangular', ws_i.center, ws_i.width - 2 * extention,
                                    ws_i.height - 2 * extention, ws_i.theta, ws_i.s)
            ori_ws_group.append(ori_squircle)
        workspace.append(ori_ws_group)
    for obs_group in world.obstacles:
        ori_obs_group = []
        for obs_i in obs_group:
            ori_squircle = Squircle('Rectangular', obs_i.center, obs_i.width - 2 * extention,
                                    obs_i.height - 2 * extention, obs_i.theta, obs_i.s)
            ori_obs_group.append(ori_squircle)
        obstacles.append(ori_obs_group)
    ori_forest_world = ForestWorld(workspace, obstacles)
    plot_world(ax, ori_forest_world)
    plot_inflate_world(ax, world)

def plot_nf(ax, execution):
    extention = 0.04
    squircle_data = execution.forest_world.squircle_data
    # print("squircle_data", squircle_data)
    inflated_squircle_data = []
    for squircle_i in squircle_data:
        inflated_squircle_i = [squircle_i[0], squircle_i[1] + 2 * extention, squircle_i[2] + 2 * extention, squircle_i[3], squircle_i[4]]
        inflated_squircle_data.append(inflated_squircle_i)

    inflated_squircle_data.append(inflated_squircle_i)
    construct_forest = ConstructForest(inflated_squircle_data)
    world = construct_forest.forest_world

    for ws_group in world.workspace:
        print("ws_group num", len(ws_group))
        for ws_i in ws_group:
            print("ws_i", ws_i, ws_i.center, ws_i.width, ws_i.height, ws_i.theta, ws_i.s)
    for obs_group in world.obstacles:
        print("obs_group num", len(obs_group))
        for obs_i in obs_group:
            print("obs_i", obs_i, obs_i.center, obs_i.width, obs_i.height, obs_i.theta, obs_i.s)
    
    """
    5
    nf_lambda= 1e5
    nf_mu = [1e10, 1e10, 1e3, 1e2, 1e2, 1e2]
    29
    nf_lambda= 1e5
    nf_mu = [1e50, 1e50, 1e20, 1e20, 1e20, 1e20]
    144
    nf_lambda= 1e9
    nf_mu = [1e200, 1e100, 1e100, 1e50, 1e50, 1e50]
    262
    nf_lambda= 6e9
    nf_mu = [1e200, 1e100, 1e100, 1e50, 1e50, 1e50]
    345
    nf_lambda= 1e12
    nf_mu = [1e200, 1e100, 1e100, 1e100, 1e100, 1e100]
    410
    nf_lambda= 5e12
    nf_mu = [1e200, 1e100, 1e100, 1e100, 1e100, 1e100]
    """
    
    nf_lambda= 1e10
    nf_mu = [1e200, 1e100, 1e100, 1e50, 1e50, 1e50]
    goal = execution.path[1]
    global_nf = NavigationFunction(world,
                                   np.array(goal), 
                                   nf_lambda=nf_lambda, nf_mu=nf_mu)
    x_min, x_max = [-0.2, 7.2]
    y_min, y_max = [-0.2, 4.2]
    resolution = 0.01
    x = np.arange(x_min, x_max, resolution)
    y = np.arange(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)
    zz_nav = global_nf.evaluate_potential(xx, yy, threshold=0.0, radius=None)

    contour_levels = [0, 0.0005, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9, 1.0]
    ctour = ax.contour(xx, yy, zz_nav, contour_levels, linewidths = 1.0, zorder=1)

def plot_nm_path(ax, path):
    path_x = []
    path_y = []
    path_theta = []
    for path_i in path[1:-1]:
        ax.plot(path_i[0], path_i[1], marker='o', color='b', markersize=5, zorder=41)
        path_x.append(path_i[0])
        path_y.append(path_i[1])
        path_theta.append(path_i[2])
    ax.plot(path[-1][0], path[-1][1], marker='o', color='b', markersize=5, zorder=41)
    path_x.append(path[-1][0])
    path_y.append(path[-1][1])
    path_theta.append(path[-1][2])
    # print("final waypoint", self.path[-1])
    # for i in range(len(path_x)):
        # ax.quiver(path_x[i], path_y[i], np.cos(path_theta[i]), np.sin(path_theta[i]), units='xy', width=0.05,
                #   headwidth=3.3, scale=1 / 0.5, color='red', zorder=2)
    near_path_x = [path[0][0], path[1][0]]
    near_path_y = [path[0][1], path[1][1]]
    ax.plot(near_path_x, near_path_y, ':', color='gold', linewidth=2.0, zorder=40)
    ax.plot(path_x, path_y, '-', color='gold', linewidth=2.0, zorder=40)
    ax.scatter(path[1][0], path[1][1], marker='^', color='blue', s=80, zorder=40)


def plot_estimated_squircles(ax, world):
    all_squircles_list = []
    for ws in world.workspace:
        if len(ws) == 1:
            continue
        ws_1 = ws[1]
        if len(ws_1.estimated_squircle_list) != 0:
            squircle_list = ws_1.estimated_squircle_list
            all_squircles_list += squircle_list
    for obs in world.obstacles:
        is_sphere = False
        for obs_i in obs:
            if obs_i.s < 0.1:
                is_sphere = True
                break
        obs_0 = obs[0]
        if is_sphere:
            squircle_list = obs_0.estimated_squircle_list
            all_squircles_list += squircle_list
        else:
            obs_0 = obs[0]
            if len(obs_0.estimated_squircle_list) != 0:
                squircle_list = obs_0.estimated_squircle_list
                all_squircles_list += squircle_list
    for squircle in all_squircles_list:
        plot_squircle(ax, squircle[0], squircle[1] / 2, squircle[2] / 2, squircle[3], squircle[4],
                      line_color = 'b', line_style = '--', line_width = 2.5, plot_z_order = 30)


def plot_trajectory(ax, trajectory):
    # print("trajectory", trajectory)
    path_x = trajectory[0][1:]
    path_y = trajectory[1][1:]
    if len(path_x) == 1:
        return
    k = 100
    if len(path_x) < k**3 - 1:
        k = int((len(path_x) + 1)**(1/3))
    b_spline = KBSpline(k=k)
    path = np.array([[path_x[i], path_y[i]] for i in range(len(path_x))])
    smooth_traj_x, smooth_traj_y = b_spline.traj_smoothing(path)

    ax.plot(smooth_traj_x[1:], smooth_traj_y[1:], '-', color='red', linewidth=2.5, zorder=40)


def plot_contour(ax, main_execute):
    # global_forest = main_execute.construct_forest.forest_world
    radius = 1.0
    all_line_list = main_execute.robot_world.all_line_list
    local_line_list = []
    goal_point = main_execute.navigation_map.path[1]
    from shapely.geometry import LineString, Point
    for line_i in all_line_list:
        points = line_i[1]
        line_start_point = points[0]
        line_end_point = points[1]
        line_sp = LineString([line_start_point, line_end_point])
        point = Point(goal_point[0], goal_point[1])
        dis = line_sp.distance(point)
        if dis < radius:
            local_line_list.append(line_i)

    construct_forest = ConstructForest(local_line_list)
    local_forest = construct_forest.forest_world
    nf_lambda = 5e1
    nf_mu = [9e3, 1e3, 1e3, 1e3, 1e3, 1e3]
    goal = main_execute.navigation_map.path[1]
    global_nf = NavigationFunction(local_forest,
                                   np.array(goal), 
                                   nf_lambda=nf_lambda, nf_mu=nf_mu)
    print("nf goal", main_execute.navigation_map.path[1])
    x_min, x_max = main_execute.robot.real_world.x_limits
    y_min, y_max = main_execute.robot.real_world.y_limits
    y_min = 1.5
    x_max = 2.5
    threshold = 0.1
    resolution = 0.01
    x = np.arange(x_min - threshold, x_max + threshold, resolution)
    y = np.arange(y_min - threshold, y_max + threshold, resolution)
    xx, yy = np.meshgrid(x, y)
    zz_nav = global_nf.evaluate_potential(xx, yy, threshold=0.05, radius=None)

    # print("zz_nav", zz_nav)
    # contour_levels = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.63, 0.7, 0.9, 1.0]
    contour_levels = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.85, 1.0]
    ax.contour(xx, yy, zz_nav, contour_levels, cmap='GnBu_r', linewidths = 0.5, zorder=0)


def plot_world_env(ax, world):
    xmin, xmax = world.x_limits
    ymin, ymax = world.y_limits
    ax.set_xlim([xmin - 0.5, xmax + 0.5])
    ax.set_ylim([ymin - 0.5, ymax + 0.5])
    ax.set_aspect('equal')
    ax.set_axis_off()
    plot_workspace_boundary(ax, world.workspace)
    plot_obstacle_boundary(ax, world.obstacles)


def plot_point(ax, world):
    global_lidar_points = world.global_lidar_points
    point_x = []
    point_y = []
    for point in global_lidar_points:
        point_x.append(point[0])
        point_y.append(point[1])
    ax.scatter(point_x, point_y, marker='s', color='lime', alpha=0.8, s=11, zorder=20)

def plot_noise_point(ax, main_execute, noise_gain=0.0):
    obstacle_points = main_execute.robot_world.obstacles_points
    # print(obstacle_points)
    point_x = []
    point_y = []
    for i, point in enumerate(obstacle_points):
        if i % 10 == 0:
            point_x.append(point[0] + np.random.normal(0.0, noise_gain))
            point_y.append(point[1] + np.random.normal(0.0, noise_gain))
    ax.scatter(point_x, point_y, marker='s', color='m', s=2)
    ax.axis('off')

def plot_occupancy_world(ax, main_execute, grid_map, grid_wall, grid_resolution):
    # grid_size_x = abs(main_execute.robot.real_world.x_limits[0] - main_execute.robot.real_world.x_limits[1])
    # grid_size_y = abs(main_execute.robot.real_world.y_limits[0] - main_execute.robot.real_world.y_limits[1])
    # grid_size_x = int(grid_size_x / grid_resolution)
    # grid_size_y = int(grid_size_y / grid_resolution)

    # grid_map = np.zeros((grid_size_y, grid_size_x))
    # grid_wall = np.zeros((grid_size_y, grid_size_x))
    
    obstacle_points = main_execute.robot.lidar_points
    # print("len(obstacle_points)", len(obstacle_points))
    robot_x = int(main_execute.robot.pose[0] / grid_resolution)
    robot_y = int(main_execute.robot.pose[1] / grid_resolution)
    for point in obstacle_points:
        grid_x = int(point[0] / grid_resolution)
        grid_y = int(point[1] / grid_resolution)
        if grid_x <= int(main_execute.forest_world.workspace[0][0].x_limits()[0] / grid_resolution):
            grid_x = int(main_execute.forest_world.workspace[0][0].x_limits()[0] / grid_resolution) + 1
        if grid_x >= int(main_execute.forest_world.workspace[0][0].x_limits()[1] / grid_resolution):
            grid_x = int(main_execute.forest_world.workspace[0][0].x_limits()[1] / grid_resolution) - 1
        if grid_y <= int(main_execute.forest_world.workspace[0][0].y_limits()[0] / grid_resolution):
            grid_y = int(main_execute.forest_world.workspace[0][0].y_limits()[0] / grid_resolution) + 1
        if grid_y >= int(main_execute.forest_world.workspace[0][0].y_limits()[1] / grid_resolution):
            grid_y = int(main_execute.forest_world.workspace[0][0].y_limits()[1] / grid_resolution) - 1
        # print("int(main_execute.robot.real_world.x_limits[1] / grid_resolution)", int(main_execute.robot.real_world.x_limits[1] / grid_resolution))
        # print("grad_x", grid_x)
        rr, cc = line(robot_y, robot_x, grid_y, grid_x)
        grid_map[rr, cc] = 1
    
    newcolors = (['lightgrey', 'None'])
    newcmap = ListedColormap(newcolors[::1]) # 重构为新的colormap
    ax.imshow(grid_map, cmap=newcmap, origin='lower',alpha=1.0, extent=[main_execute.robot.real_world.x_limits[0], 
                                                           main_execute.robot.real_world.x_limits[1], 
                                                           main_execute.robot.real_world.y_limits[0],
                                                           main_execute.robot.real_world.y_limits[1]])
    
    current_points = []
    for points in obstacle_points:
            for j, obs_polygon in enumerate(main_execute.robot.real_world.obstacles):
                if main_execute.point_to_line.points_is_in_polygon(obs_polygon, points):
                    # point transform
                    point1 = main_execute.point_to_line.trans_point_on_line(points, obs_polygon)
                    point1 = [point1.x, point1.y]
                    point1 = main_execute.point_to_line.point_approximate(point1, obs_polygon)
                    current_points.append(point1)
    # print("len(current_points)", len(current_points))
    noise_gain = 0.02
    for i, point in enumerate(current_points):
        if True:
            point_x = point[0] + np.random.normal(0.0, noise_gain)
            point_y = point[1] + np.random.normal(0.0, noise_gain)
            if not main_execute.robot.real_world.check_point_distance_of_obstacle([point_x, point_y], 0.045):
                continue
        
            grid_x = int(point_x / grid_resolution)
            grid_y = int(point_y / grid_resolution)
            if grid_x <= int(main_execute.forest_world.workspace[0][0].x_limits()[0] / grid_resolution):
                grid_x = int(main_execute.forest_world.workspace[0][0].x_limits()[0] / grid_resolution) + 1
            if grid_x >= int(main_execute.forest_world.workspace[0][0].x_limits()[1] / grid_resolution):
                grid_x = int(main_execute.forest_world.workspace[0][0].x_limits()[1] / grid_resolution) - 1
            if grid_y <= int(main_execute.forest_world.workspace[0][0].y_limits()[0] / grid_resolution):
                grid_y = int(main_execute.forest_world.workspace[0][0].y_limits()[0] / grid_resolution) + 1
            if grid_y >= int(main_execute.forest_world.workspace[0][0].y_limits()[1] / grid_resolution):
                grid_y = int(main_execute.forest_world.workspace[0][0].y_limits()[1] / grid_resolution) - 1
            grid_wall[grid_y, grid_x] = 2
    noise_gain = 0.01
    for i, point in enumerate(current_points):
        point_x = point[0] + np.random.normal(0.0, noise_gain)
        point_y = point[1] + np.random.normal(0.0, noise_gain)
        if not main_execute.robot.real_world.check_point_distance_of_obstacle([point_x, point_y], 0.002):
                continue
        
        grid_x = int(point_x / grid_resolution)
        grid_y = int(point_y / grid_resolution)
        if grid_x <= int(main_execute.forest_world.workspace[0][0].x_limits()[0] / grid_resolution):
            grid_x = int(main_execute.forest_world.workspace[0][0].x_limits()[0] / grid_resolution) + 1
        if grid_x >= int(main_execute.forest_world.workspace[0][0].x_limits()[1] / grid_resolution):
            grid_x = int(main_execute.forest_world.workspace[0][0].x_limits()[1] / grid_resolution) - 1
        if grid_y <= int(main_execute.forest_world.workspace[0][0].y_limits()[0] / grid_resolution):
            grid_y = int(main_execute.forest_world.workspace[0][0].y_limits()[0] / grid_resolution) + 1
        if grid_y >= int(main_execute.forest_world.workspace[0][0].y_limits()[1] / grid_resolution):
            grid_y = int(main_execute.forest_world.workspace[0][0].y_limits()[1] / grid_resolution) - 1
        grid_wall[grid_y, grid_x] = 1
    newcolors = (['None', '#ffb7c0', '#ffb7c0'])
    newcmap = ListedColormap(newcolors[::1]) # 重构为新的colormap
    ax.imshow(grid_wall, cmap=newcmap, origin='lower',alpha=1.0, extent=[main_execute.robot.real_world.x_limits[0], 
                                                           main_execute.robot.real_world.x_limits[1], 
                                                           main_execute.robot.real_world.y_limits[0],
                                                           main_execute.robot.real_world.y_limits[1]], zorder=1)

    # ax.scatter(select_points[0], select_points[1], marker='s', color='orchid', s=0.6, alpha=0.8)
    # ax.scatter(acc_points[0], acc_points[1], marker='s', color='orchid', s=1.0)

def plot_start_and_goal(ax, start_pose, goal_pose):
    ax.scatter(start_pose[0], start_pose[1], c='red', marker='*',
               s=50, alpha=1.0, zorder=8)

    ax.scatter(goal_pose[0], goal_pose[1], c='red', marker='^',
               s=30, alpha=1.0, zorder=8)

    # for goal_pose_i in goal_pose_list:
    #     ax.scatter(goal_pose_i[0], goal_pose_i[1], c='orange', marker='^',
    #                s=40, alpha=1.0, zorder=8)

def plot_local_and_global_squircles(ax, main_execute):
    from shapely.geometry import Point
    from shapely.geometry import box

    select_range = 1.0
    robot_point = main_execute.current_goal[0:2]
    circle_center = Point(robot_point[0], robot_point[1])
    circle_radius = select_range
    circle = circle_center.buffer(circle_radius)

    ws_select_index_list = []
    obs_select_index_list = []
    ws = main_execute.construct_forest.forest_world.workspace
    obs = main_execute.construct_forest.forest_world.obstacles
    index = 0
    for ws_tree in ws:
        for squ in ws_tree[1:]:
            rect_center = Point(squ.center[0], squ.center[1])
            rect_width = squ.width
            rect_height = squ.height
            rect = box(rect_center.x - rect_width/2, rect_center.y - rect_height/2,
            rect_center.x + rect_width/2, rect_center.y + rect_height/2)
            if rect.intersects(circle):
                ws_select_index_list.append(index)
                break
        index += 1
    index = 0
    for obs_tree in obs:
        for squ in obs_tree:
            rect_center = Point(squ.center[0], squ.center[1])
            rect_width = squ.width
            rect_height = squ.height
            rect = box(rect_center.x - rect_width/2, rect_center.y - rect_height/2,
            rect_center.x + rect_width/2, rect_center.y + rect_height/2)
            if rect.intersects(circle):
                obs_select_index_list.append(index)
                break
        index += 1
    # print("ws_select_index_list", ws_select_index_list)
    # print("obs_select_index_list", obs_select_index_list)
    for index in range(len(ws)):
        if index in ws_select_index_list:
            for squircle in ws[index][1:]:
                attached_squircle = plt.Rectangle((squircle.center[0] - squircle.width / 2,
                                               squircle.center[1] - squircle.height / 2), squircle.width,
                                              squircle.height,
                                              edgecolor='dimgrey', facecolor='#ffc6ff', linewidth=1.5, fill=True,
                                              alpha=0.8, zorder=5)
                ax.add_patch(attached_squircle)
        else:
            for squircle in ws[index][1:]:
                attached_squircle = plt.Rectangle((squircle.center[0] - squircle.width / 2,
                                               squircle.center[1] - squircle.height / 2), squircle.width,
                                              squircle.height,
                                              edgecolor='dimgrey', facecolor='#6ae9ff', linewidth=1.5, fill=True,
                                              alpha=0.8, zorder=5)
                ax.add_patch(attached_squircle)
    for index in range(len(obs)):
        if index in obs_select_index_list:
            for squircle in obs[index]:
                attached_squircle = plt.Rectangle((squircle.center[0] - squircle.width / 2,
                                               squircle.center[1] - squircle.height / 2), squircle.width,
                                              squircle.height,
                                              edgecolor='dimgrey', facecolor='#ffc6ff', linewidth=1.5, fill=True,
                                              alpha=0.8, zorder=5)
                ax.add_patch(attached_squircle)
        else:
            for squircle in obs[index]:
                attached_squircle = plt.Rectangle((squircle.center[0] - squircle.width / 2,
                                               squircle.center[1] - squircle.height / 2), squircle.width,
                                              squircle.height,
                                              edgecolor='dimgrey', facecolor='#6ae9ff', linewidth=1.5, fill=True,
                                              alpha=0.8, zorder=5)
                ax.add_patch(attached_squircle)

    all_squircles = main_execute.all_squircles

    for squircle in all_squircles:
        if squircle.ori_line is not None:
            ax.plot((squircle.ori_line[0][0], squircle.ori_line[1][0]),
                    (squircle.ori_line[0][1], squircle.ori_line[1][1]), color='blue',
                    linewidth=2.0, alpha=0.8, zorder=6)
            ax.scatter(squircle.ori_line[0][0], squircle.ori_line[0][1], c='blue', marker='o',
                       s=10, alpha=0.8, zorder=6)
            ax.scatter(squircle.ori_line[1][0], squircle.ori_line[1][1], c='blue', marker='o',
                       s=10, alpha=0.8, zorder=6)

    nm = main_execute.navigation_map
    nm.plot_path(ax)


def plot_all_lines(ax, world):
    all_line_list = []
    for ws in world.workspace:
        if len(ws) == 1:
            continue
        ws_1 = ws[1]
        if len(ws_1.accumulated_line_list) != 0:
            all_line_list += ws_1.accumulated_line_list
    for obs in world.obstacles:
        is_sphere = False
        for obs_i in obs:
            if obs_i.s < 0.1:
                is_sphere = True
                break
        if not is_sphere:
            obs_0 = obs[0]
            if len(obs_0.accumulated_line_list) != 0:
                all_line_list += obs_0.accumulated_line_list
    # print("all_line_list", all_line_list)
    for line_i in all_line_list:
        end_points_i = line_i[1]
        ax.plot((end_points_i[0][0], end_points_i[1][0]),
                (end_points_i[0][1], end_points_i[1][1]), color='blue',
                linewidth=3.0, alpha=0.8, zorder=25)
        ax.scatter(end_points_i[0][0], end_points_i[0][1], c='blue', marker='o',
                   s=16, alpha=0.8, zorder=25)
        ax.scatter(end_points_i[1][0], end_points_i[1][1], c='blue', marker='o',
                   s=16, alpha=0.8, zorder=25)



def plot_squircles(ax, main_execute):
    # for line in main_execute.all_line_list:
    #     points = line[1]
    #     ax.plot(points[:, 0], points[:, 1])
    #     mid_point = (points[0] + points[1]) / 2
    #     ax.quiver(mid_point[0], mid_point[1], line[2][0], line[2][1], color='blue', angles='xy', scale_units='xy',
    #               scale=3.0, alpha=0.5, zorder=6)
    #     ax.axis('off')
    
    #     line_start_point = points[0]
    #     line_end_point = points[1]
    #     extension = 0.18
    #     vector = [line[2][0], line[2][1]]
    #     extended_squircle = LineToSquircle(line_start_point, line_end_point, vector, extension)
    
    #     ax.scatter(extended_squircle.start_point[0], extended_squircle.start_point[1], c='blue', marker='o',
    #                s=10, alpha=0.8, zorder=6)
    #     ax.scatter(extended_squircle.end_point[0], extended_squircle.end_point[1], c='blue', marker='o',
    #                s=10, alpha=0.8, zorder=6)
    
    #     ax.plot((extended_squircle.start_point[0], extended_squircle.end_point[0]),
    #             (extended_squircle.start_point[1], extended_squircle.end_point[1]), color='blue',
    #             linewidth=2.0, alpha=0.8, zorder=6)

    all_squircles = main_execute.all_squircles

    for squircle in all_squircles:
        if squircle.ori_line is not None:
            attached_squircle = plt.Rectangle((squircle.center[0] - squircle.width / 2,
                                               squircle.center[1] - squircle.height / 2), squircle.width,
                                              squircle.height,
                                              edgecolor='dimgrey', facecolor='#6ae9ff', linewidth=1.5, fill=True,
                                              alpha=0.8, zorder=5)
            ax.add_patch(attached_squircle)
        else:
            attached_squircle = plt.Rectangle((squircle.center[0] - squircle.width / 2,
                                               squircle.center[1] - squircle.height / 2), squircle.width,
                                              squircle.height,
                                              edgecolor='dimgrey', facecolor='#6ae9ff', linewidth=1.5, fill=True,  # ffff8f
                                              alpha=0.8, zorder=6)
            ax.add_patch(attached_squircle)
        if squircle.ori_line is not None:
            ax.plot((squircle.ori_line[0][0], squircle.ori_line[1][0]),
                    (squircle.ori_line[0][1], squircle.ori_line[1][1]), color='blue',
                    linewidth=2.0, alpha=0.8, zorder=6)
            ax.scatter(squircle.ori_line[0][0], squircle.ori_line[0][1], c='blue', marker='o',
                       s=10, alpha=0.8, zorder=6)
            ax.scatter(squircle.ori_line[1][0], squircle.ori_line[1][1], c='blue', marker='o',
                       s=10, alpha=0.8, zorder=6)
    # test_squircle_list = [[[5.5, 2.45], 0.5, 1.0000000000000002]]
    # for squircle_i in test_squircle_list:
    #     center = squircle_i[0]
    #     width = squircle_i[1]
    #     height = squircle_i[2]
    #     attached_squircle = plt.Rectangle((center[0] - width / 2,
    #                                        center[1] - height / 2), width, height,
    #                                       edgecolor='k', facecolor='#89ffc6', linewidth=1.5, fill=True,  # ffff8f
    #                                       alpha=1.0, zorder=5)
    #     ax.add_patch(attached_squircle)

    nm = main_execute.navigation_map
    nm.plot_path(ax)

    # current_goal = main_execute.current_goal
    # ax.scatter(main_execute.current_goal[0], main_execute.current_goal[1], c='red', marker='^',
    #            s=30, alpha=0.5, zorder=8)
    # ax.quiver(main_execute.current_goal[0], main_execute.current_goal[1],
    #           np.cos(main_execute.current_goal[2]), np.sin(main_execute.current_goal[2]),
    #           units='xy', width=0.05,
    #           headwidth=3.3, scale=1 / 0.5, color='red', zorder=2)


def plot_workspace_boundary(ax, workspace):
    ws = workspace[0]
    vertices = ws.vertices
    extend_dis = 0.06
    extended_vertices = [np.array([vertices[0][0] - extend_dis, vertices[0][1] - extend_dis]),
                         np.array([vertices[1][0] + extend_dis, vertices[1][1] - extend_dis]),
                         np.array([vertices[2][0] + extend_dis, vertices[2][1] + extend_dis]),
                         np.array([vertices[3][0] - extend_dis, vertices[3][1] + extend_dis])]
    ws_patch = patches.Polygon(extended_vertices, linewidth=3.0, edgecolor='k', fill=False)
    ax.add_patch(ws_patch)


def plot_obstacle_boundary(ax, obstacles):
    for obs in obstacles:
        obs_patch = patches.Polygon(obs.vertices, linewidth=2.0, facecolor='mediumslateblue', alpha=1.0, edgecolor='k', fill=False)
        ax.add_patch(obs_patch)


def plot_tree(ax, node, forest_world):
    x, y = forest_world.node_pos[node]
    ax.text(x, y, str(node.name), ha='center', va='center',
            bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle="circle,pad=0.3"))

    for child in node.children:
        x_child, y_child = forest_world.node_pos[child]
        curve = forest_world.bezier_curve((x, y), (x_child, y_child), height_factor=0.1)
        path = np.array([list(path_i) for path_i in curve])
        b_spline = KBSpline(k=3)
        smooth_points_x, smooth_points_y = b_spline.traj_smoothing(path)
        ax.plot(smooth_points_x, smooth_points_y, color='blue', linewidth=2)
        plot_tree(ax, child, forest_world)


def plot_forest_world(ax, main_execute):
    forest_root = main_execute.construct_forest.forest_root

    max_depth = 0
    node_depth = {}
    node_queue = [(forest_root, 0)]
    while node_queue:
        node, depth = node_queue.pop(0)
        node_depth[node] = depth
        max_depth = max(max_depth, depth)
        node_queue.extend((child, depth + 1) for child in node.children)

    spacing = 180
    main_execute.construct_forest.layout_tree(forest_root, 0, 0, spacing, 0)

    # print(forest_root)

    plot_tree(ax, forest_root, main_execute.construct_forest)
    ax.axis('equal')
    ax.axis('off')


def plot_four_ax(ax1, ax2, ax3, ax4, main_execute):
    # plot on real world
    plot_world(ax1, main_execute.robot.real_world)
    plot_robot_in_world(ax1, main_execute.robot)
    plot_trajectory(ax1, main_execute)
    plot_start_and_goal(ax1, main_execute.robot.start, main_execute.robot.goal)

    # plot on point world
    plot_world(ax2, main_execute.robot.real_world)
    plot_robot_in_world(ax2, main_execute.robot)
    # plot_point(ax2, main_execute.robot_world)
    # plot_noise_point(ax2, main_execute, noise_gain=0.003)
    plot_trajectory(ax2, main_execute)
    plot_start_and_goal(ax2, main_execute.robot.start, main_execute.robot.goal)

    # plot on squircle world
    plot_world(ax3, main_execute.robot.init_world)
    plot_local_and_global_squircles(ax3, main_execute)
    plot_point_in_inflated_world(ax3, main_execute.robot)
    # plot_noise_point(ax3, main_execute, noise_gain=0.02)
    # plot_contour(ax3, main_execute)
    plot_trajectory(ax3, main_execute)
    plot_start_and_goal(ax3, main_execute.robot.start, main_execute.robot.goal)

    # plot forest
    plot_forest_world(ax4, main_execute)


def test_data(ax1, ax2, ax3, ax4, main_execute):
    # plot on real world
    plot_world(ax1, main_execute.robot.real_world)
    plot_robot_in_world(ax1, main_execute.robot)
    plot_trajectory(ax1, main_execute)
    plot_start_and_goal(ax1, main_execute.robot.start, main_execute.robot.goal, main_execute.robot.goal_list)

    # plot on point world
    plot_world(ax2, main_execute.robot.init_world)
    plot_robot_in_world(ax2, main_execute.robot)
    plot_point(ax2, main_execute.robot_world)
    plot_trajectory(ax2, main_execute)
    plot_start_and_goal(ax2, main_execute.robot.start, main_execute.robot.goal, main_execute.robot.goal_list)

    # plot on squircle world
    plot_world(ax3, main_execute.robot.init_world)
    plot_local_and_global_squircles(ax3, main_execute)
    plot_point_in_inflated_world(ax3, main_execute.robot)
    plot_trajectory(ax3, main_execute)
    plot_start_and_goal(ax3, main_execute.robot.start, main_execute.robot.goal, main_execute.robot.goal_list)

    # plot on polygon world
    plot_world(ax4, main_execute.robot.init_world)
    plot_squircles(ax4, main_execute)
    polygon_list = main_execute.all_polygon_list    
    for polygon in polygon_list:
        print("polygon", polygon)
        poly = Polygon(polygon, edgecolor='k', linestyle='-', fill=False)
        ax4.add_patch(poly)
