"""
Transformation utils
"""

import numpy as np
import math
import torch
import pdb
def x_to_world(pose):

    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


import torch


#TODO:jinlong
def x_to_world_tensor(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : torch.Tensor
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : torch.Tensor
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose

    # used for rotation matrix
    # print(yaw)

    c_y = torch.cos(torch.deg2rad(yaw))
    s_y = torch.sin(torch.deg2rad(yaw))
    c_r = torch.cos(torch.deg2rad(roll))
    s_r = torch.sin(torch.deg2rad(roll))
    c_p = torch.cos(torch.deg2rad(pitch))
    s_p = torch.sin(torch.deg2rad(pitch))

    matrix = torch.eye(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix



def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or
        transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or
         transformation matrix x2->world

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """

    # print(type(x1), type(x2))
    

    if isinstance(x1, list) and isinstance(x2, list):

        

        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)

        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)


    elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):#TODO:jinlong 
        # print('11111111111111111111 ', x1)
        # print('22222222222222222222 ', x2)
        x1_to_world = x_to_world_tensor(x1)
        x2_to_world = x_to_world_tensor(x2)
        # print('33333333333333333333 ', x1_to_world)
        # print('444444444444444444444 ', x2_to_world)        
        world_to_x2 = torch.inverse(x2_to_world)
        #world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = torch.mm(world_to_x2, x1_to_world)
        # print('555555555555555555555 ', transformation_matrix)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x1_to_world = torch.tensor(x1_to_world)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
        #transformation_matrix = torch.mm(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    # pdb.set_trace()
    return transformation_matrix


def dist_two_pose(cav_pose, ego_pose):
    """
    Calculate the distance between agent by given there pose.
    """
    if isinstance(cav_pose, list):
        distance = \
            math.sqrt((cav_pose[0] -
                       ego_pose[0]) ** 2 +
                      (cav_pose[1] - ego_pose[1]) ** 2)
    else:
        distance = \
            math.sqrt((cav_pose[0, -1] -
                       ego_pose[0, -1]) ** 2 +
                      (cav_pose[1, -1] - ego_pose[1, -1]) ** 2)
    return distance


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous
