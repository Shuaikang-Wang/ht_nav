import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon, Circle
import numpy as np
from os.path import abspath, join


def plot_robot_in_world(ax, robot, lidar_on=True):
    if lidar_on:
        plot_lidar(ax, robot.pose, robot.lidar_points)
    plot_robot(ax, robot.pose, robot.size, robot.type)


def plot_point_in_inflated_world(ax, robot, plot_z_order=100):
    pose = robot.pose
    size = 0.08
    car_circle_inner = plt.Circle(pose[0:2], size, color='red', alpha=0.5, zorder=plot_z_order, fill=True)
    ax.add_patch(car_circle_inner)


def plot_robot(ax, pose, size, type='unicycle', facecolor='b', edgecolor='k', plot_z_order=100):
    if type == 'unicycle':
        x, y, theta = pose[:]
        car = [(x - size, y - size), (x - size, y + size), (x, y + 2 * size), (x + size, y + size),
               (x + size, y - size)]
        car_polygon = Polygon(car, fill=True, facecolor=facecolor, edgecolor=edgecolor, lw=2.5,
                              zorder=plot_z_order)  # lw=3
        ts = ax.transData
        tr = Affine2D().rotate_around(x, y, theta - 0.5 * np.pi)
        t = tr + ts
        car_polygon.set_transform(t)
        ax.add_patch(car_polygon)
    elif type == 'point':
        size = 0.07
        car_circle_inner = plt.Circle(pose[0:2], size, facecolor=facecolor, edgecolor=edgecolor,
                                      zorder=plot_z_order, lw=2.5, fill=True)
        ax.add_patch(car_circle_inner)
        # ax.plot(*pose[0:2], marker='o', color='b', zorder=plot_z_order)

    # ax.plot(*pose[0:2], color='darkblue', marker='o', markersize=100 * size, zorder=plot_z_order)


def plot_lidar(ax, pose, lidar_points, plot_z_order=99):
    for pt in lidar_points:
        ax.plot([pose[0], pt[0]], [pose[1], pt[1]], linewidth=1.5, alpha=0.2, color='m', zorder=plot_z_order)
        ax.plot(pt[0], pt[1], marker='s', color='grey',
                markerfacecolor='m', markersize=3.0, markeredgewidth=0.4, zorder=plot_z_order)


def plot_all_log_data(robot, skip=10, folder='snaps', only_final=False):  # 总2
    # [k * time_step, np.copy(self.pose), np.copy(self.lidar_points),
    #                           np.array([linear_v, angular_v]), np.array(grad),
    #                           len(self.world_model.obstacles)]
    full_obstacles = list(robot.world_model.obstacles)
    traj = []
    grad_all = []
    k = 0
    T_stamp = np.arange(0, len(robot.log), skip)
    for t_stamp in T_stamp:
        data = robot.log[t_stamp]
        t = data[0]
        robot._pose = np.array(data[1])
        traj.append(np.array(data[1]))
        robot._lidar_points = np.array(data[2])
        linear_v, angular_v = data[3][:]
        grad_all.append(np.array(data[4]))
        robot.world_model._obstacles = full_obstacles[0:data[5]]
        # robot.compute_nf()
        # plot
        if (not only_final) or (t_stamp == T_stamp[-1]):
            ax = plot_robot_in_world(robot)
            k += 1
            ax.set_title('Time %.2f s | v: %.2f, $\omega$: %.2f' % (
                float(t), float(linear_v), float(angular_v)))
            # save fig
            fig_name = 'NF %02d.png' % k
            plt.savefig(join(folder, fig_name), bbox_inches='tight')
            plt.clf()
            plt.close()
    print('|||||*%d* snapshots saved under %s||||||||' % (k, folder))
