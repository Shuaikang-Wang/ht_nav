import pickle
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import datetime
from ENV.vis import plot_occupancy_world, plot_trajectory, plot_data, plot_cluster, plot_init_data, plot_cluster_segment, plot_fitting_squircle, plot_polygon_list, plot_nm_path, plot_fitting_world, plot_nf
from ENV.construct_forest import ConstructForest

file_path = 'DATA/pickle_data/'
current_date_execution = datetime.datetime.now().strftime("execution_%Y_%m_%d")

folder_path_execution = os.path.join(file_path, current_date_execution)

with open(folder_path_execution + '/execution.pickle', 'rb') as f:
    execution_data = pickle.load(f)

fig = plt.figure(figsize=(18, 6))

grid_resolution=0.05
grid_size_x = abs(execution_data[0].forest_world.workspace[0][0].x_limits()[0] - \
                  execution_data[0].forest_world.workspace[0][0].x_limits()[1])
grid_size_y = abs(execution_data[0].forest_world.workspace[0][0].y_limits()[0] - \
                  execution_data[0].forest_world.workspace[0][0].y_limits()[1])
grid_size_x = int(grid_size_x / grid_resolution)
grid_size_y = int(grid_size_y / grid_resolution)
grid_map = np.zeros((grid_size_y, grid_size_x))

grid_wall = np.zeros((grid_size_y, grid_size_x))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
print("overall frame: ", len(execution_data))

plot_step = len(execution_data)

import matplotlib.colors as mcolors


def generate_colors(num_colors, r_range=(0.2, 0.9), g_range=(0.2, 0.9), b_range=(0.2, 0.9)):
    COLORS = []
    for _ in range(num_colors):
        r = np.random.uniform(r_range[0], r_range[1])
        g = np.random.uniform(g_range[0], g_range[1])
        b = np.random.uniform(b_range[0], b_range[1])
        color = (r, g, b)
        COLORS.append(color)
    return COLORS

COLORS = generate_colors(30)
COLORS = [(0.6655945812790935, 0.7991128931718545, 0.30916315164416913), (0.32829267037101884, 0.7451305003948241, 0.36280186596035074), (0.854229488115634, 0.5990835866803179, 0.20781223608871766), (0.8742568931262042, 0.6297996970464653, 0.49028682060664874), (0.20930653280720707, 0.36417626588239316, 0.5931371973553741), (0.324754507817581, 0.5088792895304222, 0.7694473256897623), (0.5763588604189889, 0.2808405161036487, 0.860610929917361), (0.6644150453023188, 0.7783679150362854, 0.8980454591587148), (0.47674735247496336, 0.2744716725177803, 0.7072411914884495), (0.6222016891706936, 0.7600144323151621, 0.3606482762722216), (0.2013759982641025, 0.7166964813151788, 0.831061282743023), (0.32310868193888787, 0.5748996172398056, 0.3037129178331264), (0.7770521875936589, 0.3856652812161864, 0.6392810779995997), (0.7973099041784191, 0.31543512634425985, 0.594249132537819), (0.644351792216931, 0.5963216697298116, 0.38269319382637046), (0.6449494311678624, 0.8980627331361042, 0.29682668290826875), (0.6910142022335852, 0.8885519016543781, 0.8136110956402887), (0.6037358058808033, 0.37428284757035013, 0.5318987636451674), (0.8905749919868624, 0.8582759306533658, 0.5567107403719688), (0.5489150602942576, 0.7487437735882794, 0.29816960173778656), (0.2502914774042717, 0.7448088588223913, 0.4547877682541195), (0.6229053609193845, 0.32005331696094574, 0.8604402766939614), (0.726627001298221, 0.5396248912414816, 0.2897533029280746), (0.33676673420322356, 0.4613465231659416, 0.6635673730543519), (0.6313634857587729, 0.6697070552715317, 0.6206241638664551), (0.8077059923769898, 0.6094351178175885, 0.44613905831444)]



for i in range(86, 87):
    # if i == 508 or i == 520:plot_fitting_world(ax2, execution_data[i].forest_world))
    #     continue
    ax1.clear()
    ax2.clear()

    # reexecution
    squircle_data = execution_data[i].sq_esti.fit_squircle_group(execution_data[i].forest_world.all_cluster_segments)
    execution_data[i].forest_world.squircle_data = squircle_data
    execution_data[i].construct_forest = ConstructForest(execution_data[i].forest_world.squircle_data)

    plot_data(ax1, execution_data[i].robot, execution_data[i].forest_world)
    plot_trajectory(ax1, execution_data[i].trajectory)
    plot_trajectory(ax2, execution_data[i].trajectory)

    plot_cluster(ax2, execution_data[i].forest_world, COLORS)
    plot_init_data(ax2, execution_data[i].robot, execution_data[i].forest_world)
    plot_nf(ax2, execution_data[i])
    # plot_cluster_segment(ax2, execution_data[i].forest_world)
    # plot_fitting_squircle(ax2, execution_data[i].forest_world)
    plot_fitting_world(ax2, execution_data[i].forest_world, execution_data[i].construct_forest.forest_world)
    # plot_polygon_list(ax2, execution_data[i].all_polygon_list)
    plot_nm_path(ax2, execution_data[i].path)
    # plot_occupancy_world(ax2, execution_data[i], grid_map, grid_wall, grid_resolution)
    file_path = './DATA/nf_data'
    current_date = datetime.datetime.now().strftime("snap_%Y_%m_%d")
    folder_path = os.path.join(file_path, current_date)
    # folder_path = './DATA/figure_data/snap_test'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = f"{i}.png"
    # print("path", os.path.join(folder_path, file_name))
    plt.savefig(os.path.join(folder_path, file_name), dpi=1200)
    print("=============frame " + str(i) + " is saved==============")
