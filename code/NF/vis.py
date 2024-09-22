import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from NF.utils import compute_squicle_length_ray_plt, distance
from NF.geometry import Workspace
from ROBOT.vis import plot_robot, plot_lidar
from ROBOT.lidar import Lidar
from NF.geometry import World


def plot_init_world(ax, world):
    # ax = fig.add_subplot(111, projection='3d')
    # world_config = '../complex_world/auto_config/real_tsp_world.yaml'
    # world = World(world_config)
    xmin, xmax = world.workspace[0][0].x_limits()
    ymin, ymax = world.workspace[0][0].y_limits()
    ax.set_xlim([xmin - 1.0, xmax + 1.0])
    ax.set_ylim([ymin - 1.0, ymax + 1.0])
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    start_pose = np.array([3.2, 0.4, -np.pi])   #2.8,0.4,np.pi
    ax.plot(start_pose[0], start_pose[1], '*r', markersize=10.0)
    # world = World('../complex_world/tsp_config/tsp_world_contour_1.yaml')
    # plot_fill_workspace(ax, world.workspace)
    # plot_fill_obstacle(ax, world.obstacles)
    plot_init_workspace_boundary(ax, world.workspace)
    # plot_workspace_boundary(ax, world.workspace, line_color = 'k',
    #                         line = (0, (1, 1)), line_width = 1.6, z_order=15)
    # plot_obstacle_boundary(ax, world.obstacles, line_color = 'k',
    #                        line = (0, (1, 1)), line_width = 1.6, z_order=15)
    # new_world = World('../complex_world/auto_config/new_tsp_world.yaml')
    # plot_new_obstacle(ax, new_world.obstacles)
    return ax


def plot_world(ax, world):
    # ax = fig.add_subplot(111, projection='3d')
    # world_config = '../complex_world/auto_config/real_tsp_world.yaml'
    # world = World(world_config)
    xmin, xmax = world.workspace[0][0].x_limits()
    ymin, ymax = world.workspace[0][0].y_limits()
    ax.set_xlim([xmin - 1.0, xmax + 1.0])
    ax.set_ylim([ymin - 1.0, ymax + 1.0])
    ax.set_aspect('equal', 'box')
    ax.set_axis_off()
    # start_pose = np.array([3.2, 0.4, -np.pi])   #2.8,0.4,np.pi
    # ax.plot(start_pose[0], start_pose[1], '*r', markersize=10.0)
    # world = World('../complex_world/tsp_config/tsp_world_contour_1.yaml')
    plot_fill_workspace(ax, world.workspace)
    plot_fill_obstacle(ax, world.obstacles)
    plot_workspace_boundary(ax, world.workspace)
    plot_obstacle_boundary(ax, world.obstacles)
    plot_workspace_boundary(ax, world.workspace, line_color = 'k',
                            line = (0, (1, 1)), line_width = 1.6, z_order=15)
    plot_obstacle_boundary(ax, world.obstacles, line_color = 'k',
                           line = (0, (1, 1)), line_width = 1.6, z_order=15)
    # new_world = World('../complex_world/auto_config/new_tsp_world.yaml')
    # plot_new_obstacle(ax, new_world.obstacles)
    return ax

def plot_inflate_world(ax, world):
    plot_fill_inflate_workspace(ax, world.workspace)
    plot_fill_inflate_obstacle(ax, world.obstacles)
    plot_inflate_workspace_boundary(ax, world.workspace, line_color = 'b',
                            line = '-', line_width = 1.0, z_order=15)
    plot_inflate_obstacle_boundary(ax, world.obstacles, line_color = 'b',
                           line = '-', line_width = 1.0, z_order=15)
    return ax

def plot_init_workspace_boundary(ax, workspace, line_color = 'k', line = '-',line_width = 2.0, z_order = 10):
    ws_i = workspace[0][0]
    i = 0
    if ws_i.type == 'Workspace':
        plot_squircle(ax, ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s, line_color=line_color,
                      line_style=line, line_width=line_width, plot_z_order=z_order + i)
        plot_squircle(ax, ws_i.center, ws_i.width / 2 + 0.2, ws_i.height / 2 + 0.2, ws_i.theta, ws_i.s, line_color=line_color,
                      line_style=line, line_width=line_width, plot_z_order=z_order + i)  #ws_i.height / 2 + 0.18  0.1
    elif ws_i.type == 'Circle':
        inner_boundary = plt.Circle((ws_i.center[0], ws_i.center[1]), radius=ws_i.radius, color='k',
                                    fill=False, linestyle=line, ec="black", linewidth=line_width, zorder=z_order + i)
        ax.add_patch(inner_boundary)


def plot_workspace_boundary(ax, workspace, line_color = 'k', line = '-',line_width = 2.0, z_order = 10):
    ws_i = workspace[0][0]
    i = 0
    if ws_i.type == 'Workspace':
        plot_squircle(ax, ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s, line_color=line_color,
                      line_style=line, line_width=line_width, plot_z_order=z_order + i)
        plot_squircle(ax, ws_i.center, ws_i.width / 2 + 0.2, ws_i.height / 2 + 0.2, ws_i.theta, ws_i.s, line_color=line_color,
                      line_style=line, line_width=line_width, plot_z_order=z_order + i)  #ws_i.height / 2 + 0.18  0.1
    elif ws_i.type == 'Circle':
        inner_boundary = plt.Circle((ws_i.center[0], ws_i.center[1]), radius=ws_i.radius, color='k',
                                    fill=False, linestyle=line, ec="black", linewidth=line_width, zorder=z_order + i)
        ax.add_patch(inner_boundary)
    for ws in workspace:
        i = 0
        for ws_i in ws[1:]:
            i += 1
            if ws_i.type == 'Rectangular':
                plot_squircle(ax, ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s, line_color = line_color,
                                line_style = line, line_width = line_width, plot_z_order = z_order + i)
            elif ws_i.type == 'Circle':
                inner_boundary = plt.Circle((ws_i.center[0], ws_i.center[1]), radius = ws_i.radius, color='k',
                                            fill=False, linestyle=line, ec="black", linewidth = line_width, zorder = z_order + i)
                ax.add_patch(inner_boundary)


def plot_obstacle_boundary(ax, obstacles, line_color = 'k', line = '-', line_width = 2.0, z_order = 10):
    for obs in obstacles:
        i = 0
        for obs_i in obs:
            if obs_i.type == 'Rectangular':
                plot_squircle(ax, obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s, line_color = line_color,
                              line_style = line, line_width = line_width, plot_z_order = z_order + i)
            elif obs_i.type == 'Circle':
                inner_boundary = plt.Circle((obs_i.center[0], obs_i.center[1]), radius = obs_i.radius, color='k',
                                            fill=False, linestyle=line, ec="black", linewidth = line_width, zorder = z_order + i)
                ax.add_patch(inner_boundary)
            i += 1


def plot_inflate_workspace_boundary(ax, workspace, line_color = 'k', line = '-',line_width = 2.0, z_order = 10):
    for ws in workspace:
        i = 0
        for ws_i in ws[1:]:
            i += 1
            plot_squircle(ax, ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s, line_color = line_color,
                            line_style = line, line_width = line_width, plot_z_order = z_order + i)


def plot_inflate_obstacle_boundary(ax, obstacles, line_color = 'k', line = '-', line_width = 2.0, z_order = 10):
    for obs in obstacles:
        i = 0
        for obs_i in obs:
            plot_squircle(ax, obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s, line_color = line_color,
                            line_style = line, line_width = line_width, plot_z_order = z_order + i)
            i += 1

def plot_new_obstacle(ax, obstacles, line_color = 'k', line = '-', line_width = 1.5, z_order = 10):
    for obs in obstacles:
        z_order = 10
        i = 2
        for obs_i in obs:
            xmin, xmax = obs_i.x_limits()
            ymin, ymax = obs_i.y_limits()
            xs = np.linspace(xmin - 0.2, xmax + 0.2, 200)
            ys = np.linspace(ymin - 0.2, ymax + 0.2, 200)
            xv, yv = np.meshgrid(xs, ys)
            zv = []
            threshold = 0.001
            for xx, yy in zip(xv.ravel(), yv.ravel()):
                q = np.array([xx, yy])
                if distance(obs_i.center, q) < \
                        compute_squicle_length_ray(obs_i.width / 2, obs_i.height / 2, q - obs_i.center, obs_i.theta, obs_i.s) - threshold:
                    zz = 0
                elif (distance(obs_i.center, q) >=
                      compute_squicle_length_ray(obs_i.width / 2, obs_i.height / 2, q - obs_i.center, obs_i.theta, obs_i.s) - threshold) and \
                        (distance(obs_i.center, q) <=
                         compute_squicle_length_ray(obs_i.width / 2, obs_i.height / 2, q - obs_i.center, obs_i.theta, obs_i.s) + threshold):
                    zz = 1.0
                elif distance(obs_i.center, q) > \
                        compute_squicle_length_ray(obs_i.width / 2, obs_i.height / 2, q - obs_i.center, obs_i.theta, obs_i.s) + threshold:
                    zz = 2.0
                else:
                    zz = 3.0
                zv.append(zz)
            zv = np.asarray(zv).reshape(xv.shape)
            ax.contourf(xv, yv, zv, levels=[0.0, 1.0], colors=('#838bc5'), zorder = z_order + i)
            i += 1
    for obs in obstacles:
        z_order = 10
        i = 2
        for obs_i in obs:
            if obs_i.type == 'Rectangular':
                plot_squircle(ax, obs_i.center, obs_i.width / 2, obs_i.height / 2, line_color = line_color,
                              line_style = line, line_width = line_width, plot_z_order = z_order + i)
            elif obs_i.type == 'Circle':
                inner_boundary = plt.Circle((obs_i.center[0], obs_i.center[1]), radius = obs_i.radius, color='k',
                                            fill=False, linestyle=line, ec="black", linewidth = line_width, zorder = z_order + i)
                ax.add_patch(inner_boundary)
            i += 1

def plot_fill_workspace(ax, workspace):
    z_order = 10
    ws_i = workspace[0][0]
    i = 0
    xmin, xmax = ws_i.x_limits()
    ymin, ymax = ws_i.y_limits()
    xs = np.linspace(xmin - 0.5, xmin + 0.3, 100)
    ys = np.linspace(ymin - 0.5, ymax + 0.5, 500)
    xv, yv = np.meshgrid(xs, ys)
    zv = []
    infla_ws = Workspace('Rectangular', ws_i.center, ws_i.width + 0.2 + 0.12, ws_i.height + 0.2 + 0.12, ws_i.theta, ws_i.s) #ws_i.width + 0.36, ws_i.height + 0.36
    for xx, yy in zip(xv.ravel(), yv.ravel()):
        q = np.array([xx, yy])
        threshold = 0.001
        if distance(ws_i.center, q) < \
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold:
            zz = 0
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold):
            zz = 1.0
        elif (distance(ws_i.center, q) >
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold) and \
                (distance(ws_i.center, q) <
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold):
            zz = 1.5
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold):
            zz = 2.0
        elif distance(ws_i.center, q) > \
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold:
            zz = 2.5
        else:
            zz = 3.0
        zv.append(zz)
    zv = np.asarray(zv).reshape(xv.shape)
    ax.contourf(xv, yv, zv, levels=[1.0, 2.0], colors=('gray'), zorder = z_order + i)

    xs = np.linspace(xmax - 0.3, xmax + 0.5, 100)
    ys = np.linspace(ymin - 0.5, ymax + 0.5, 500)
    xv, yv = np.meshgrid(xs, ys)
    zv = []
    for xx, yy in zip(xv.ravel(), yv.ravel()):
        q = np.array([xx, yy])
        threshold = 0.001
        if distance(ws_i.center, q) < \
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold:
            zz = 0
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold):
            zz = 1.0
        elif (distance(ws_i.center, q) >
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold) and \
                (distance(ws_i.center, q) <
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold):
            zz = 1.5
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold):
            zz = 2.0
        elif distance(ws_i.center, q) > \
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold:
            zz = 2.5
        else:
            zz = 3.0
        zv.append(zz)
    zv = np.asarray(zv).reshape(xv.shape)
    ax.contourf(xv, yv, zv, levels=[1.0, 2.0], colors=('gray'), zorder=z_order + i)

    xs = np.linspace(xmin - 0.5, xmax + 0.5, 500)
    ys = np.linspace(ymin - 0.5, ymin + 0.3, 100)
    xv, yv = np.meshgrid(xs, ys)
    zv = []
    for xx, yy in zip(xv.ravel(), yv.ravel()):
        q = np.array([xx, yy])
        threshold = 0.001
        if distance(ws_i.center, q) < \
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold:
            zz = 0
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold):
            zz = 1.0
        elif (distance(ws_i.center, q) >
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold) and \
                (distance(ws_i.center, q) <
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold):
            zz = 1.5
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold):
            zz = 2.0
        elif distance(ws_i.center, q) > \
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold:
            zz = 2.5
        else:
            zz = 3.0
        zv.append(zz)
    zv = np.asarray(zv).reshape(xv.shape)
    ax.contourf(xv, yv, zv, levels=[1.0, 2.0], colors=('gray'), zorder=z_order + i)

    xs = np.linspace(xmin - 0.5, xmax + 0.5, 500)
    ys = np.linspace(ymax - 0.3, ymax + 0.5, 100)
    xv, yv = np.meshgrid(xs, ys)
    zv = []
    for xx, yy in zip(xv.ravel(), yv.ravel()):
        q = np.array([xx, yy])
        threshold = 0.001
        if distance(ws_i.center, q) < \
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold:
            zz = 0
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold) and \
                (distance(ws_i.center, q) <=
                    compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold):
            zz = 1.0
        elif (distance(ws_i.center, q) >
                compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold) and \
                (distance(ws_i.center, q) <
                    compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold):
            zz = 1.5
        elif (distance(ws_i.center, q) >=
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) - threshold) and \
                (distance(ws_i.center, q) <=
                 compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold):
            zz = 2.0
        elif distance(ws_i.center, q) > \
                compute_squicle_length_ray_plt(q - infla_ws.center, infla_ws.width / 2, infla_ws.height / 2, infla_ws.theta, infla_ws.s) + threshold:
            zz = 2.5
        else:
            zz = 3.0
        zv.append(zz)
    zv = np.asarray(zv).reshape(xv.shape)
    ax.contourf(xv, yv, zv, levels=[1.0, 2.0], colors=('gray'), zorder=z_order + i)

    for ws in workspace:
        i = 0
        for ws_i in ws[1:]:
            i += 1
            xmin, xmax = ws_i.x_limits()
            ymin, ymax = ws_i.y_limits()
            xs = np.linspace(xmin - 0.2, xmax + 0.2, 200)
            ys = np.linspace(ymin - 0.2, ymax + 0.2, 200)
            xv, yv = np.meshgrid(xs, ys)
            zv = []
            threshold = 0.001
            for xx, yy in zip(xv.ravel(), yv.ravel()):
                q = np.array([xx, yy])
                if distance(ws_i.center, q) < \
                        compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold:
                    zz = 0
                elif (distance(ws_i.center, q) >=
                        compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold) and \
                        (distance(ws_i.center, q) <=
                         compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold):
                    zz = 1.0
                elif distance(ws_i.center, q) > \
                         compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold:
                    zz = 2.0
                else:
                    zz = 3.0
                zv.append(zz)
            zv = np.asarray(zv).reshape(xv.shape)
            ax.contourf(xv, yv, zv, levels=[0.0, 1.0], colors=('lightgray'), zorder = z_order + i)

def plot_fill_obstacle(ax, obstacles):
    for obs in obstacles:
        z_order = 10
        i = 0
        for obs_i in obs:
            xmin, xmax = obs_i.x_limits()
            ymin, ymax = obs_i.y_limits()
            xs = np.linspace(xmin - 0.2, xmax + 0.2, 200)
            ys = np.linspace(ymin - 0.2, ymax + 0.2, 200)
            xv, yv = np.meshgrid(xs, ys)
            zv = []
            threshold = 0.001
            for xx, yy in zip(xv.ravel(), yv.ravel()):
                q = np.array([xx, yy])
                if distance(obs_i.center, q) < \
                        compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) - threshold:
                    zz = 0
                elif (distance(obs_i.center, q) >=
                      compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) - threshold) and \
                        (distance(obs_i.center, q) <=
                         compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) + threshold):
                    zz = 1.0
                elif distance(obs_i.center, q) > \
                        compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) + threshold:
                    zz = 2.0
                else:
                    zz = 3.0
                zv.append(zz)
            zv = np.asarray(zv).reshape(xv.shape)
            ax.contourf(xv, yv, zv, levels=[0.0, 1.0], colors=('lightgray'), zorder = z_order + i)
            i += 1
    
def plot_fill_inflate_workspace(ax, workspace):
    z_order=1
    for ws in workspace:
        i = 0
        for ws_i in ws[1:]:
            i += 1
            xmin, xmax = ws_i.x_limits()
            ymin, ymax = ws_i.y_limits()
            xs = np.linspace(xmin - 0.2, xmax + 0.2, 200)
            ys = np.linspace(ymin - 0.2, ymax + 0.2, 200)
            xv, yv = np.meshgrid(xs, ys)
            zv = []
            threshold = 0.001
            for xx, yy in zip(xv.ravel(), yv.ravel()):
                q = np.array([xx, yy])
                if distance(ws_i.center, q) < \
                        compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold:
                    zz = 0
                elif (distance(ws_i.center, q) >=
                        compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) - threshold) and \
                        (distance(ws_i.center, q) <=
                        compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold):
                    zz = 1.0
                elif distance(ws_i.center, q) > \
                        compute_squicle_length_ray_plt(q - ws_i.center, ws_i.width / 2, ws_i.height / 2, ws_i.theta, ws_i.s) + threshold:
                    zz = 2.0
                else:
                    zz = 3.0
                zv.append(zz)
            zv = np.asarray(zv).reshape(xv.shape)
            ax.contourf(xv, yv, zv, levels=[0.0, 1.0], colors=('#838bc5'), zorder = z_order + i) #cornflowerblue

def plot_fill_inflate_obstacle(ax, obstacles):
    for obs in obstacles:
        z_order = 1
        i = 0
        for obs_i in obs:
            xmin, xmax = obs_i.x_limits()
            ymin, ymax = obs_i.y_limits()
            xs = np.linspace(xmin - 0.2, xmax + 0.2, 200)
            ys = np.linspace(ymin - 0.2, ymax + 0.2, 200)
            xv, yv = np.meshgrid(xs, ys)
            zv = []
            threshold = 0.001
            for xx, yy in zip(xv.ravel(), yv.ravel()):
                q = np.array([xx, yy])
                if distance(obs_i.center, q) < \
                        compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) - threshold:
                    zz = 0
                elif (distance(obs_i.center, q) >=
                      compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) - threshold) and \
                        (distance(obs_i.center, q) <=
                         compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) + threshold):
                    zz = 1.0
                elif distance(obs_i.center, q) > \
                        compute_squicle_length_ray_plt(q - obs_i.center, obs_i.width / 2, obs_i.height / 2, obs_i.theta, obs_i.s) + threshold:
                    zz = 2.0
                else:
                    zz = 3.0
                zv.append(zz)
            zv = np.asarray(zv).reshape(xv.shape)
            ax.contourf(xv, yv, zv, levels=[0.0, 1.0], colors=('#838bc5'), zorder = z_order + i)
            i += 1

def plot_squircle(ax, center, width, height, theta, s, line_color = 'k', line_style = '-', line_width = 1.5, plot_z_order = 10):
    angle_list = np.linspace(0, 2 * np.pi, 1000)
    traj_x = []
    traj_y = []
    for angle_i in angle_list:
        q = np.array([np.cos(angle_i), np.sin(angle_i)])
        rho_i = compute_squicle_length_ray_plt(q, width, height, theta, s)
        traj_x_i = center[0] + rho_i * np.cos(angle_i)
        traj_y_i = center[1] + rho_i * np.sin(angle_i)
        # rotated_x_i = (traj_x_i - center[0]) * np.cos(theta) - (traj_y_i - center[1]) * np.sin(theta) + \
        #               center[0]
        # rotated_y_i = (traj_x_i - center[0]) * np.sin(theta) + (traj_y_i - center[1]) * np.cos(theta) + \
        #               center[1]
        traj_x.append(traj_x_i)
        traj_y.append(traj_y_i)
    ax.plot(traj_x, traj_y, color = line_color, linestyle = line_style, linewidth = line_width, zorder = plot_z_order)


# def plot_task_area(ax, ec_color = "purple", face_color = "violet", line_width = 1.5):
#     task_area_A = [2.7, 1.2, 0.25]
#     task_area_B = [3.6, 2.7, 0.25]
#     task_area_C = [2.88, 4.6, 0.25]
#     task_area_D = [2.15, 3.4, 0.25]
#     task_area_E = [1.0, 3.15, 0.25]
#     task_area_F = [0.5, 1.8, 0.25]
#     circle_A = plt.Circle(task_area_A[0: 2], radius=task_area_A[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     circle_B = plt.Circle(task_area_B[0: 2], radius=task_area_B[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     circle_C = plt.Circle(task_area_C[0: 2], radius=task_area_C[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     circle_D = plt.Circle(task_area_D[0: 2], radius=task_area_D[2], fill=True,
#                           ec='orange', facecolor = 'peachpuff', linewidth=line_width, zorder = 29)
#     # ec = ec_color, facecolor = face_color
#     circle_E = plt.Circle(task_area_E[0: 2], radius=task_area_E[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     # circle_F = plt.Circle(task_area_F[0: 2], radius=task_area_F[2], fill=True,
#     #                       ec='orange', facecolor = 'peachpuff', linewidth=line_width, zorder = 29)
#     circle_F = plt.Circle(task_area_F[0: 2], radius=task_area_F[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder=29)
#     ax.add_patch(circle_A)
#     ax.add_patch(circle_B)
#     ax.add_patch(circle_C)
#     ax.add_patch(circle_D)
#     ax.add_patch(circle_E)
#     ax.add_patch(circle_F)
#     plt.text(task_area_A[0] - 0.1 + 0.01, task_area_A[1] - 0.1 + 0.01, 'A', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_B[0] - 0.1 + 0.02, task_area_B[1] - 0.1 + 0.01, 'B', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_C[0] - 0.1, task_area_C[1] - 0.1 + 0.01, 'C', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_D[0] - 0.1 + 0.01, task_area_D[1] - 0.1 + 0.01, 'D', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_E[0] - 0.1 + 0.01, task_area_E[1] - 0.1 + 0.01, 'E', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_F[0] - 0.1 + 0.03, task_area_F[1] - 0.1, 'F', fontsize=10, color='k', zorder = 30)
#     plt.text(0.6, 5.3, 'F', fontsize=10, color='k', zorder=30)
#     plt.text(1.2, 5.3, 'E', fontsize=10, color='k', zorder=30)
#     plt.text(1.8, 5.3, 'A', fontsize=10, color='k', zorder=30)
#     plt.text(2.4, 5.3, 'B', fontsize=10, color='k', zorder=30)
#     plt.text(3.0, 5.3, 'C', fontsize=10, color='k', zorder=30)
#     plt.text(3.6, 5.3, 'D', fontsize=10, color='k', zorder=30)
#     plt.quiver(0.3 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     # plt.quiver(0.3 - 0.05, 5.4, 1.0, 0.0, color="orange")
#     plt.quiver(0.9 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(1.5 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(2.1 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(2.7 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(3.3 - 0.05, 5.4, 1.0, 0.0, color="orange")
#     # "purple"


# def plot_task_area(ax, ec_color = "purple", face_color = "violet", line_width = 1.5):
#     task_area_r_1 = [1.0, 4.65, 0.25]
#     task_area_r_2 = [3.3, 3.55, 0.25]
#     task_area_r_3 = [1.4, 2.0, 0.25]
#     task_area_r_4 = [1.2, 0.9, 0.25]
#     task_area_r_5 = [3.5, 1.4, 0.25]
#     circle_r_1 = plt.Circle(task_area_r_1[0: 2], radius=task_area_r_1[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_2 = plt.Circle(task_area_r_2[0: 2], radius=task_area_r_2[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_3 = plt.Circle(task_area_r_3[0: 2], radius=task_area_r_3[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_4 = plt.Circle(task_area_r_4[0: 2], radius=task_area_r_4[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_5 = plt.Circle(task_area_r_5[0: 2], radius=task_area_r_5[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     ax.add_patch(circle_r_1)
#     ax.add_patch(circle_r_2)
#     ax.add_patch(circle_r_3)
#     ax.add_patch(circle_r_4)
#     ax.add_patch(circle_r_5)
#     plt.text(task_area_r_1[0] - 0.1, task_area_r_1[1] - 0.08, '$r_1$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_2[0] - 0.1, task_area_r_2[1] - 0.08, '$r_2$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_3[0] - 0.1, task_area_r_3[1] - 0.08, '$r_3$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_4[0] - 0.1, task_area_r_4[1] - 0.08, '$r_4$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_5[0] - 0.1, task_area_r_5[1] - 0.08, '$r_5$', fontsize=10, color='k', zorder = 30)
#     plt.text(1.2, 5.3, '$r_1$', fontsize=10, color='k', zorder=30)
#     plt.text(1.8, 5.3, '$r_3$', fontsize=10, color='k', zorder=30)
#     plt.text(2.4, 5.3, '$r_4$', fontsize=10, color='k', zorder=30)
#     plt.text(3.0, 5.3, '$r_5$', fontsize=10, color='k', zorder=30)
#     # plt.quiver(0.3 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(0.9 - 0.05, 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(1.5 - 0.05, 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(2.1 - 0.05, 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(2.7 - 0.05, 5.4, 1.0, 0.0, color="purple")
#     # "purple"

def plot_task_area(ax, ec_color = "black", face_color = "deepskyblue", line_width = 1.0):

    # pick_up_regions
    face_color = "deepskyblue"
    task_area_r_1 = [0.4, 0.4, 0.35, 0.35]
    task_area_r_2 = [2.2, 3.5, 0.35, 0.35]
    circle_r_1 = plt.Rectangle((task_area_r_1[0]-task_area_r_1[2] / 2, task_area_r_1[1]-task_area_r_1[3] / 2),
                               width=task_area_r_1[2], height = task_area_r_1[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
    circle_r_2 = plt.Rectangle((task_area_r_2[0]-task_area_r_2[2] / 2, task_area_r_2[1]-task_area_r_2[3] / 2),
                               width=task_area_r_2[2], height = task_area_r_2[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)

    # deliver_regions
    face_color = "plum"
    task_area_r_3 = [0.9, 3.5, 0.35, 0.35]
    task_area_r_4 = [6.3, 0.5, 0.35, 0.35]
    task_area_r_5 = [6.4, 3.0, 0.35, 0.35]
    circle_r_3 = plt.Rectangle((task_area_r_3[0]-task_area_r_3[2] / 2, task_area_r_3[1]-task_area_r_3[3] / 2),
                               width=task_area_r_3[2], height=task_area_r_3[3], fill=True,
                               ec=ec_color, facecolor=face_color, linewidth=line_width, zorder=29)
    circle_r_4 = plt.Rectangle((task_area_r_4[0]-task_area_r_4[2] / 2, task_area_r_4[1]-task_area_r_4[3] / 2),
                               width=task_area_r_4[2], height = task_area_r_4[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
    circle_r_5 = plt.Rectangle((task_area_r_5[0]-task_area_r_5[2] / 2, task_area_r_5[1]-task_area_r_5[3] / 2),
                               width=task_area_r_5[2], height = task_area_r_5[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)

    # # start_region
    # face_color = "red"
    # task_area_r_6 = [2.8, 0.4, 0.5, 0.5]
    # circle_r_6 = plt.Rectangle((task_area_r_6[0]-task_area_r_6[2] / 2, task_area_r_6[1]-task_area_r_6[3] / 2),
    #                            width=task_area_r_6[2], height=task_area_r_6[3], fill=True,
    #                            ec=ec_color, facecolor=face_color, linewidth=line_width, zorder=29)

    # urgent_regions
    face_color = "orange"
    task_area_r_6 = [4.6, 0.5, 0.35, 0.35]
    circle_r_6 = plt.Rectangle((task_area_r_6[0]-task_area_r_6[2] / 2, task_area_r_6[1]-task_area_r_6[3] / 2),
                               width=task_area_r_6[2], height=task_area_r_6[3], fill=True,
                               ec=ec_color, facecolor=face_color, linewidth=line_width, zorder=29)

    ax.add_patch(circle_r_1)
    ax.add_patch(circle_r_2)
    ax.add_patch(circle_r_3)
    ax.add_patch(circle_r_4)
    ax.add_patch(circle_r_5)
    ax.add_patch(circle_r_6)
    ax.text(task_area_r_1[0] - 0.13, task_area_r_1[1] - 0.05, '$p_1$', fontsize=13, color='k', zorder = 30)
    ax.text(task_area_r_2[0] - 0.13, task_area_r_2[1] - 0.05, '$p_2$', fontsize=13, color='k', zorder = 30)
    ax.text(task_area_r_3[0] - 0.14, task_area_r_3[1] - 0.08, '$d_1$', fontsize=13, color='k', zorder = 30)
    ax.text(task_area_r_4[0] - 0.14, task_area_r_4[1] - 0.08, '$d_2$', fontsize=13, color='k', zorder = 30)
    ax.text(task_area_r_5[0] - 0.14, task_area_r_5[1] - 0.08, '$d_3$', fontsize=13, color='k', zorder = 30)
    ax.text(task_area_r_6[0] - 0.13, task_area_r_6[1] - 0.08, '$u_1$', fontsize=13, color='k', zorder=30)


def plot_task_area_1(ax, ec_color = "black", face_color = "deepskyblue", line_width = 1.0):

    # pick_up_regions
    face_color = "plum"
    task_area_r_1 = [2.7, 0.5, 0.4, 0.4]
    task_area_r_2 = [3.5, 3.0, 0.4, 0.4]
    task_area_r_3 = [0.5, 3.5, 0.4, 0.4]
    task_area_r_4 = [0.6, 2.2, 0.4, 0.4]
    circle_r_1 = plt.Rectangle((task_area_r_1[0]-task_area_r_1[2] / 2, task_area_r_1[1]-task_area_r_1[3] / 2),
                               width=task_area_r_1[2], height = task_area_r_1[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
    circle_r_2 = plt.Rectangle((task_area_r_2[0]-task_area_r_2[2] / 2, task_area_r_2[1]-task_area_r_2[3] / 2),
                               width=task_area_r_2[2], height = task_area_r_2[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
    circle_r_3 = plt.Rectangle((task_area_r_3[0]-task_area_r_3[2] / 2, task_area_r_3[1]-task_area_r_3[3] / 2),
                               width=task_area_r_3[2], height=task_area_r_3[3], fill=True,
                               ec=ec_color, facecolor=face_color, linewidth=line_width, zorder=29)
    circle_r_4 = plt.Rectangle((task_area_r_4[0]-task_area_r_4[2] / 2, task_area_r_4[1]-task_area_r_4[3] / 2),
                               width=task_area_r_4[2], height = task_area_r_4[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)

    ax.add_patch(circle_r_1)
    ax.add_patch(circle_r_2)
    ax.add_patch(circle_r_3)
    ax.add_patch(circle_r_4)
    plt.text(task_area_r_1[0] - 0.16, task_area_r_1[1] - 0.09, '$d_1$', fontsize=13, color='k', zorder = 30)
    plt.text(task_area_r_2[0] - 0.16, task_area_r_2[1] - 0.09, '$d_2$', fontsize=13, color='k', zorder = 30)
    plt.text(task_area_r_3[0] - 0.16, task_area_r_3[1] - 0.09, '$d_3$', fontsize=13, color='k', zorder = 30)
    plt.text(task_area_r_4[0] - 0.16, task_area_r_4[1] - 0.09, '$d_4$', fontsize=13, color='k', zorder = 30)

def plot_task_area_hard(ax, ec_color = "black", face_color = "deepskyblue", line_width = 1.0):

    # pick_up_regions
    # r_1: (1.0m, 0.3m)
    # r_2: (1.85m, 0.84m)
    # r_3: (0.74m, 1.16m)
    # r_4: (0.18m, 1.83m)
    face_color = "plum"
    # task_area_r_1 = [1.05, 0.3, 0.2, 0.2]
    # task_area_r_2 = [1.83, 0.8, 0.2, 0.2]
    # task_area_r_3 = [0.75, 1.1, 0.2, 0.2]
    # task_area_r_4 = [0.25, 1.8, 0.2, 0.2]
    task_area_r_1 = [1.0, 0.3, 0.2, 0.2]
    task_area_r_2 = [1.85, 0.84, 0.2, 0.2]
    task_area_r_3 = [0.74, 1.16, 0.2, 0.2]
    task_area_r_4 = [0.18, 1.83, 0.2, 0.2]
    circle_r_1 = plt.Rectangle((task_area_r_1[0]-task_area_r_1[2] / 2, task_area_r_1[1]-task_area_r_1[3] / 2),
                               width=task_area_r_1[2], height = task_area_r_1[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
    circle_r_2 = plt.Rectangle((task_area_r_2[0]-task_area_r_2[2] / 2, task_area_r_2[1]-task_area_r_2[3] / 2),
                               width=task_area_r_2[2], height = task_area_r_2[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
    circle_r_3 = plt.Rectangle((task_area_r_3[0]-task_area_r_3[2] / 2, task_area_r_3[1]-task_area_r_3[3] / 2),
                               width=task_area_r_3[2], height=task_area_r_3[3], fill=True,
                               ec=ec_color, facecolor=face_color, linewidth=line_width, zorder=29)
    circle_r_4 = plt.Rectangle((task_area_r_4[0]-task_area_r_4[2] / 2, task_area_r_4[1]-task_area_r_4[3] / 2),
                               width=task_area_r_4[2], height = task_area_r_4[3], fill=True,
                          ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)

    ax.add_patch(circle_r_1)
    ax.add_patch(circle_r_2)
    ax.add_patch(circle_r_3)
    ax.add_patch(circle_r_4)
    plt.text(task_area_r_1[0] - 0.08, task_area_r_1[1] - 0.05, '$r_1$', fontsize=10, color='k', zorder = 30)
    plt.text(task_area_r_2[0] - 0.08, task_area_r_2[1] - 0.05, '$r_2$', fontsize=10, color='k', zorder = 30)
    plt.text(task_area_r_3[0] - 0.08, task_area_r_3[1] - 0.05, '$r_3$', fontsize=10, color='k', zorder = 30)
    plt.text(task_area_r_4[0] - 0.08, task_area_r_4[1] - 0.05, '$r_4$', fontsize=10, color='k', zorder = 30)


# def plot_task_area(ax, ec_color = "purple", face_color = "violet", line_width = 1.5):
#     task_area_A = [2.6, 0.4, 0.25]
#     task_area_B = [3.5, 2.9, 0.25]
#     task_area_C = [2.3, 4.65, 0.25]
#     task_area_D = [0.4, 3.8, 0.25]
#     task_area_E = [0.65, 2.1, 0.25]
#     circle_A = plt.Circle(task_area_A[0: 2], radius=task_area_A[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     circle_B = plt.Circle(task_area_B[0: 2], radius=task_area_B[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     circle_C = plt.Circle(task_area_C[0: 2], radius=task_area_C[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     circle_D = plt.Circle(task_area_D[0: 2], radius=task_area_D[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     # ec = ec_color, facecolor = face_color
#     # ec = 'dodgerblue', facecolor = 'lightskyblue'
#     circle_E = plt.Circle(task_area_E[0: 2], radius=task_area_E[2], fill=True,
#                           ec='sienna', facecolor='rosybrown', linewidth=line_width, zorder = 29)
#     ax.add_patch(circle_A)
#     ax.add_patch(circle_B)
#     ax.add_patch(circle_C)
#     ax.add_patch(circle_D)
#     ax.add_patch(circle_E)
#     plt.text(task_area_A[0] - 0.1 + 0.01, task_area_A[1] - 0.1 + 0.01, 'A', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_B[0] - 0.1 + 0.02, task_area_B[1] - 0.1 + 0.01, 'B', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_C[0] - 0.1, task_area_C[1] - 0.1 + 0.01, 'C', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_D[0] - 0.1 + 0.01, task_area_D[1] - 0.1 + 0.01, 'D', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_E[0] - 0.1 + 0.01, task_area_E[1] - 0.1 + 0.01, 'E', fontsize=10, color='k', zorder = 30)
#     plt.text(0.6 + 0.35, 5.3, 'E', fontsize=10, color='k', zorder=30)
#     plt.text(1.2 + 0.35, 5.3, 'C', fontsize=10, color='k', zorder=30)
#     plt.text(1.8 + 0.35, 5.3, 'D', fontsize=10, color='k', zorder=30)
#     plt.text(2.4 + 0.35, 5.3, 'B', fontsize=10, color='k', zorder=30)
#     plt.text(3.0 + 0.35, 5.3, 'A', fontsize=10, color='k', zorder=30)
#     # plt.quiver(0.3 - 0.05, 5.4, 1.0, 0.0, color="orange")
#     plt.quiver(0.3 + 0.35 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(0.9 + 0.35 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(1.5 + 0.35 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(2.1 + 0.35 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     plt.quiver(2.7 + 0.35 - 0.05, 5.4, 1.0, 0.0, color="sienna")
#     # "purple"

# def plot_task_area(ax, ec_color = "purple", face_color = "violet", line_width = 1.5):
#     task_area_r_1 = [2.6, 0.4, 0.25]
#     task_area_r_2 = [3.5, 2.9, 0.25]
#     task_area_r_3 = [2.3, 4.65, 0.25]
#     task_area_r_4 = [0.4, 3.8, 0.25]
#     task_area_r_5 = [0.65, 2.1, 0.25]
#     circle_r_1 = plt.Circle(task_area_r_1[0: 2], radius=task_area_r_1[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_2 = plt.Circle(task_area_r_2[0: 2], radius=task_area_r_2[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_3 = plt.Circle(task_area_r_3[0: 2], radius=task_area_r_3[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_4 = plt.Circle(task_area_r_4[0: 2], radius=task_area_r_4[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     circle_r_5 = plt.Circle(task_area_r_5[0: 2], radius=task_area_r_5[2], fill=True,
#                           ec = ec_color, facecolor = face_color, linewidth=line_width, zorder = 29)
#     ax.add_patch(circle_r_1)
#     ax.add_patch(circle_r_2)
#     ax.add_patch(circle_r_3)
#     ax.add_patch(circle_r_4)
#     ax.add_patch(circle_r_5)
#     plt.text(task_area_r_1[0] - 0.1, task_area_r_1[1] - 0.08, '$r_1$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_2[0] - 0.1, task_area_r_2[1] - 0.08, '$r_2$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_3[0] - 0.1, task_area_r_3[1] - 0.08, '$r_3$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_4[0] - 0.1, task_area_r_4[1] - 0.08, '$r_4$', fontsize=10, color='k', zorder = 30)
#     plt.text(task_area_r_5[0] - 0.1, task_area_r_5[1] - 0.08, '$r_5$', fontsize=10, color='k', zorder = 30)
#     plt.text(0.0 + 0.35 - 0.05 + 0.05, 5.3, '$r_5$', fontsize=10, color='k', zorder=30)
#     plt.text(0.6 + 0.35 + 0.05, 5.3, '$r_3$', fontsize=10, color='k', zorder=30)
#     plt.text(1.2 + 0.35 + 0.05, 5.3, '$r_4$', fontsize=10, color='k', zorder=30)
#     plt.text(1.8 + 0.35 + 0.05, 5.3, '$r_2$', fontsize=10, color='k', zorder=30)
#     plt.text(2.4 + 0.35 + 0.05, 5.3+0.02, '$q_t$', fontsize=10, color='k', zorder=30)
#     plt.text(3.0 + 0.35 + 0.05 + 0.03, 5.3, '$r_1$', fontsize=10, color='k', zorder=30)
#     # plt.quiver(0.3 - 0.05, 5.4, 1.0, 0.0, color="orange")
#     plt.quiver(0.3 + 0.35, 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(0.9 + 0.35, 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(1.5 + 0.35, 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(2.1 + 0.35 , 5.4, 1.0, 0.0, color="purple")
#     plt.quiver(2.7 + 0.35 + 0.03, 5.4, 1.0, 0.0, color="purple")
#     # "purple"

def plot_robot_lidar(ax, nf, current_pose):
    plot_robot(ax, current_pose, size=0.09, facecolor='b', edgecolor='k', plot_z_order  = 50) #size=0.04 0.1
    lidar = Lidar(radius=0.5) #0.25 0.8
    lidar_points, detected_obstacles = lidar.get_measurements(current_pose, nf.world)
    plot_lidar(ax, current_pose, lidar_points=lidar_points, plot_z_order = 49)


def plot_contour(ax, nf):
    xx, yy = nf.world.workspace[0][0].workspace_meshgrid(resolution=0.05)
    zz_nav = nf.evaluate_potential(xx, yy, threshold=0.1)
    # contour_levels = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.5, 0.63, 0.7, 0.9, 1.0]
    contour_levels = [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9,1.0]
    ctour = ax.contour(xx, yy, zz_nav, contour_levels, linewidths = 1.0, zorder=1)
    # plt.contour(xx, yy, zz_nav, contour_levels, linewidths=1.0, zorder=30)
    # plt.colorbar(ctour, shrink=0.6)
    # plt.colorbar()
    # ax.plot_surface(xx, yy, zz_nav, lw=0.5, rstride=1, cstride=1)
    # ax.contour(xx, yy, zz_nav, contour_levels, linewidths = 1.0, linestyles="solid", zorder=1)
    # ax.contour(xx, yy, zz_nav, contour_levels, linewidths = 1.0, colors="k", linestyles="solid", zorder=1)
    return ax


# def plot_vector_field(ax, nf):
#     xx, yy = nf.world.workspace[0][0].workspace_meshgrid(resolution=0.2, threshold=0.1)
#     # grad_x, grad_y = nf.evaluate_gradient(xx, yy)
#     grad_x, grad_y = nf.evaluate_gradient(xx, yy, threshold = 0.05)
#     ax.quiver(xx, yy, grad_x, grad_y, units='xy', width=0.03, headwidth=2.4,
#                   scale=1 / 0.24, color='#6488ea', zorder=30)


def plot_vector_field(ax, nf):
    xx, yy = nf.world.workspace[0][0].workspace_meshgrid(resolution=0.2, threshold=0.1)
    colors = plt.get_cmap('viridis', 1000)
    for qx, qy in zip(xx.ravel(), yy.ravel()):
        q = np.array([qx, qy])
        # gradient = nf.compute_gradient_point(q)
        potential = nf.compute_potential_at_point(q)
        gradient = nf.compute_mapped_gradient(q)
        if np.linalg.norm(gradient) == 0:
            normalized_grad = np.array([[0], [0]])
        else:
            normalized_grad = - gradient / np.linalg.norm(gradient)
            grad_x = normalized_grad[0][0]
            grad_y = normalized_grad[1][0]
            potential_color_i = int(1000 * potential)
            ax.quiver(qx, qy, grad_x, grad_y, units='xy', width=0.021, headwidth=3.8,
                        scale=1 / 0.21, color=colors([potential_color_i]), zorder=30)


def resave_potential(nf):
    x = []
    y = []
    gradient_x = []
    gradient_y = []
    data_file = '../static_NF/evaluate_gradient.txt'
    with open(data_file, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            data = list(line.rstrip().split(','))
            x.append(float(data[0]))
            y.append(float(data[1]))
            gradient_x.append(float(data[2]))
            gradient_y.append(float(data[3]))
    f.close
    with open(data_file, 'w', encoding='utf-8') as f:
        for i in range(len(x)):
            length = (gradient_x[i] ** 2 + gradient_y[i] ** 2) ** 0.5
            if length != 0.0:
                q = np.array([x[i], y[i]])
                potential_i = nf.compute_potential_at_point(q)
                f.write(
                    str(x[i]) + ',' + str(y[i]) + ',' + str(gradient_x[i]) + ','
                    + str(gradient_y[i]) + ',' + str(potential_i) + '\n')
    f.close()

def plot_saved_vector_field(ax, nf):
    x = []
    y = []
    gradient_x = []
    gradient_y = []
    potential = []
    data_file = '../static_NF/evaluate_gradient.txt'
    colors = plt.get_cmap('viridis', 1000)
    with open(data_file, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            data = list(line.rstrip().split(','))
            x.append(float(data[0]))
            y.append(float(data[1]))
            gradient_x.append(float(data[2]))
            gradient_y.append(float(data[3]))
            potential.append(float(data[4]))
    f.close
    for i, x_i in enumerate(x):
        length = (gradient_x[i] ** 2 + gradient_y[i] ** 2) ** 0.5
        gradient_x_i = gradient_x[i] / length
        gradient_y_i = gradient_y[i] / length
        potential_i = int(1000 * potential[i])
        ax.quiver(x[i], y[i], gradient_x_i, gradient_y_i, units='xy', width=0.03, headwidth=2.4,
                  scale=1 / 0.24, color=colors([potential_i]), zorder=30)
        # '#6488ea'


def plot_path(ax, nf, start,  **kwargs):
    x_traj, y_traj, theta_traj = nf.compute_vector_follow_path(start, **kwargs)
    # print(path)
    ax.plot(x_traj, y_traj, '-r', linewidth = 1.5,zorder=35)

def plot_path_1(ax, nf, start,  **kwargs):
    path, vel_list, acc_list = nf.calculate_path(start, **kwargs)
    # print(path)
    ax.plot(path[:, 0], path[:, 1], linestyle='-', color = 'royalblue', linewidth = 1.5)

def plot_path_2(ax, nf, start,  **kwargs):
    path, vel_list, acc_list = nf.calculate_path(start, **kwargs)
    # print(path)
    ax.plot(path[:, 0], path[:, 1], linestyle='-', color = 'violet', linewidth = 1.5)

def plot_path_3(ax, nf, start,  **kwargs):
    path, vel_list, acc_list = nf.calculate_path(start, **kwargs)
    # print(path)
    ax.plot(path[:, 0], path[:, 1], linestyle='-', color = 'cyan', linewidth = 1.5)


def plot_pose(ax, nf, start,  **kwargs):
    all_pose, step_num = nf.calculate_nf_pose(start, **kwargs)
    for i in range(step_num):
        x = all_pose[i][0]
        y = all_pose[i][1]
        theta = all_pose[i][2]
        ax.arrow(x, y, 0.15 * np.cos(theta), 0.15 * np.sin(theta), 
                 width = 0.001, head_width = 0.03, length_includes_head = True, color = 'k')
    # ax.plot(all_pose[:, 0], all_pose[:, 1], '-r', linewidth = 1.0)


def plot_trajectory(ax, nf, start_pose,  **kwargs):
    x_traj, y_traj = nf.compute_nf_ego_path(start_pose, **kwargs)
    ax.plot(x_traj, y_traj, '-r', linewidth = 1.0)


def plot_vector_follow_trajectory(ax, nf, start_pose, **kwargs):
    x_traj, y_traj = nf.compute_vector_follow_path(start_pose, **kwargs)
    ax.plot(x_traj, y_traj, '-r', linewidth=1.5, zorder = 35)


def plot_saved_trajectory(ax, nf, linestyle='-', color='r', data_file = '../complex_world/dynamic_results/selection_trajectory_data_1.txt'):
    final_trajectory_x = []
    final_trajectory_y = []
    final_theta = []
    with open(data_file, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            data = list(line.rstrip().split(','))
            final_trajectory_x.append(float(data[0]))
            final_trajectory_y.append(float(data[1]))
            final_theta.append(float(data[2]))
    f.close
    ax.plot(final_trajectory_x, final_trajectory_y, linestyle=linestyle, color=color, linewidth=1.8, zorder = 35)
    # current_pose = [final_trajectory_x[-1], final_trajectory_y[-1], final_theta[-1]]
    # plot_robot(ax, current_pose, size=0.1, facecolor='b', edgecolor='k', plot_z_order  = 50)
    # lidar = Lidar()
    # lidar_points, detected_obstacles = lidar.get_measurements(current_pose, nf.world)
    # plot_lidar(ax, current_pose, lidar_points=lidar_points, plot_z_order = 49)


def plot_path_on_contour(ax, nf, start_pose, **kwargs):
    # nf.save_vector_follow_path(start_pose,
    #                            save_path = '../complex_world/auto_results/RRT/tsp_trajectory_data_22.txt')
    # nf.save_vector_follow_path(start_pose,
    #                            save_path='../static_NF/path_1.txt')
    plot_world(ax, nf.world)
    # plot_task_area_1(ax)
    plot_contour(ax, nf)
    # nf.save_evaluate_gradient(threshold=0.0)
    # resave_potential(nf)
    # goal_pose = [1.2, 3.8, np.pi / 2]
    # ax.plot(goal_pose[0], goal_pose[1], '^', color = 'darkorange', markersize=6.0, zorder = 35)
    # start_pose = np.array([1.73,1.33,np.pi / 2])
    # ax.plot(start_pose[0], start_pose[1], '*r', markersize=6.0, zorder = 35)
    # plot_saved_vector_field(ax, nf)
    # plot_vector_field(ax,nf)
    # current_pose = [3.8, 2.8, 2*np.pi/5]
    # plot_robot_lidar(ax, nf, current_pose)

    # plot_saved_trajectory(ax, nf, linestyle='-', color='orange',
    #                       data_file='../complex_world/auto_results/RRT/tsp_trajectory_data.txt')
    # plot_saved_trajectory(ax, nf, linestyle='-', color='g',
    #                       data_file='../complex_world/auto_results/ONF/tsp_trajectory_data.txt')
    # plot_saved_trajectory(ax, nf, linestyle='-', color='b',
    #                       data_file='../complex_world/auto_results/NF/tsp_trajectory_data.txt')
    # plot_saved_trajectory(ax, nf, linestyle='-', color='r',
    #                       data_file='../complex_world/auto_results/NF_RRT/tsp_trajectory_data.txt')

    # plot_saved_trajectory(ax, nf,
    #                       data_file='../complex_world/auto_results/RRT/tsp_trajectory_data_22.txt')
    # plot_saved_trajectory(ax, nf,
    #                       data_file='../complex_world/dynamic_results/selection_trajectory_1/selection_trajectory_data_4.txt')
    # plot_saved_trajectory(ax, nf,
    #                       data_file='../complex_world/dynamic_results/selection_trajectory_1/selection_trajectory_data_5.txt')
    # v_list = plot_path(ax, nf, start[0:2], **kwargs)
    # plot_pose(ax, nf, start,  **kwargs)
    # plot_trajectory(ax, nf, start[0:2], **kwargs)
    # start = np.array([0.5, 0.4, 0.0])
    # plot_path(ax, nf, start_pose)
    # start_pose = np.array([0.3, 2.9, 0.0])
    # ax.plot(start_pose[0], start_pose[1], '*',color='royalblue', markersize=6.0, zorder=35)
    # plot_path_1(ax, nf, start_pose)
    # start_pose = np.array([2.2, 4.6, 0.0])
    # ax.plot(start_pose[0], start_pose[1], '*', color='violet', markersize=6.0, zorder=35)
    # plot_path_2(ax, nf, start_pose)
    # start_pose = np.array([0.4, 4.5, 3.1])
    # ax.plot(start_pose[0], start_pose[1], '*', color='cyan', markersize=6.0, zorder=35)
    # plot_path_3(ax, nf, start_pose)

def compute_total_cost(data_file = '../complex_world/auto_results/NF_RRT/tsp_trajectory_data.txt'):
    final_trajectory_x = []
    final_trajectory_y = []
    final_theta = []
    with open(data_file, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            data = list(line.rstrip().split(','))
            final_trajectory_x.append(float(data[0]))
            final_trajectory_y.append(float(data[1]))
            final_theta.append(float(data[2]))
    f.close
    distance = 0
    for i in range(len(final_trajectory_x)-1):
        distance_i = np.sqrt((final_trajectory_x[i+1] - final_trajectory_x[i])**2 +
                             (final_trajectory_y[i+1] - final_trajectory_y[i])**2)
        distance += distance_i
    print(distance)

def plot_complexity(fig):
    ax = fig.add_subplot(1, 1, 1)
    obs_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dt = 0.013887665855480876 - 0.0076837723291595435
    compute_time = [0.013887665855480876,
                    0.01579788397458266,
                    0.018536261331724216,
                    0.02118649932329413,
                    0.023777417580157858,
                    0.02728601114864571,
                    0.029959823139921203,
                    0.03462324673925646,
                    0.03731925117994823,
                    0.04293165392667758,
                    0.04524969165652134,
                    0.051969585064709604]
    compute_time_1 = [
                    0.0076837723291595435 + dt,
                    0.008554179988799199 + dt,
                    0.010022093880449943 + dt,
                    0.012619018516650074 + dt,
                    0.013352976084914741 + dt,
                    0.014110567241561156 + dt,
                    0.014982380259302285 + dt,
                    0.016036600668812424 + dt,
                    0.017721990955188102 + dt,
                    0.01862100829730467 + dt,
                    0.019752671650081342 + dt,
                    0.024001286118313567 + dt]
    # font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23}
    ax.set_xlim([-0.1, 11.1])
    ax.set_ylim([0.010, 0.055])
    ax.set_aspect(80)
    plot_1 = ax.plot(obs_num, compute_time, linewidth = 2.0, c='limegreen', zorder = 1, label = '              ') #'Nominal Method'
    plot_2 = ax.plot(obs_num, compute_time_1, linewidth = 2.0, c='violet', zorder=1, label='                   ') #Incremental Method
    legend = plt.legend(['                ', '               '], fontsize=14)
    plt.scatter(obs_num, compute_time, marker = 'o', c='green', zorder = 2)
    plt.scatter(obs_num, compute_time_1, marker='s', c='m', zorder = 2)
    plt.grid(linestyle='--', alpha=0.5)
    # plt.legend()
    x_major_locator = MultipleLocator(1.0)
    y_major_locator = MultipleLocator(0.005)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xlabel('      ', fontsize=14) #Number of Environment Updates
    ax.set_ylabel('       ', fontsize=14) #Computation Time (s)


