import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import copy
from NF.geometry import World
from EXEC.execution import Execution
from ENV.geometry import InitWorld
from ROBOT.robot import Robot

import pickle
import datetime

# robot start pose
start_pose = np.array([3.2, 0.4, -np.pi])

# task areas
goal_1 = [0.4, 0.4, -np.pi / 2]
goal_2 = [2.2, 3.5, -0.2]

goal_pose_list = [goal_1, goal_2]

# world and robot config
init_world_config = './CONFIG/workspace_boundary.yaml'
forest_config = './CONFIG/forest_world.yaml'
robot_config = './CONFIG/robot_config.yaml'

init_world = InitWorld(init_world_config)
forest_world = World(forest_config)
# robot_world = RobotWorld()
robot = Robot(start_pose, forest_world, init_world, robot_config=robot_config)

main_execute = Execution(robot, forest_world)

main_execute.robot.goal_list = goal_pose_list

max_step = 10000
current_frame = 0

execution_data = []

# set data path
file_path = 'DATA/pickle_data/'
current_date_execution = datetime.datetime.now().strftime("execution_%Y_%m_%d")
folder_path_execution = os.path.join(file_path, current_date_execution)
if not os.path.exists(folder_path_execution):
    os.makedirs(folder_path_execution)

main_execute.goal_index = 0
main_execute.robot.set_start(start_pose)
main_execute.robot.set_goal(goal_pose_list[main_execute.goal_index])

for frame in range(0, max_step):
    current_frame = frame
    print("=============current frame is: " + str(current_frame) + "==============")

    # move to next goal
    if np.linalg.norm(np.asarray(goal_pose_list[-1][:2]) - main_execute.robot.pose[:2]) <= 0.1:
        main_execute.goal_index = -1
        main_execute.robot.set_goal(goal_pose_list[main_execute.goal_index])
    else:
        if np.linalg.norm(np.asarray(goal_pose_list[main_execute.goal_index][:2]) - robot.pose[:2]) <= 0.1:
            main_execute.goal_index += 1
            main_execute.robot.set_goal(goal_pose_list[main_execute.goal_index])

    # save data every 20 step
    if current_frame % 20 == 0:
        with open(folder_path_execution + '/execution.pickle', 'wb') as f:
            pickle.dump(execution_data, f)

    if np.linalg.norm(np.asarray(goal_pose_list[-1][:2]) - robot.pose[:2]) <= 0.1:
        main_execute.trajectory[0].append(goal_pose_list[-1][0])
        main_execute.trajectory[1].append(goal_pose_list[-1][1])
        with open(folder_path_execution + '/execution.pickle', 'wb') as f:
            pickle.dump(execution_data, f)
        break

    try:
        main_execute.one_step_forward(current_frame)
    except:
        print("************Error and Stop************")
        with open(folder_path_execution + '/execution.pickle', 'wb') as f:
            pickle.dump(execution_data, f)

        main_execute.one_step_forward(current_frame)
        break
    else:
        # add data to data_list in order to plot in the end.

        main_execute_copy = copy.deepcopy(main_execute)
        execution_data.append(main_execute_copy)
