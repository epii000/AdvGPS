

import numpy as np
import math
import torch




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

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x1_to_world = torch.tensor(x1_to_world)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
        # transformation_matrix = torch.mm(world_to_x2, x1_to_world)

    return transformation_matrix



def matrix_to_pose(given_matrix):
    """
    Convert a 4x4 transformation matrix to pose [x, y, z, roll, yaw, pitch].

    Parameters
    ----------
    given_matrix : np.ndarray
        The 4x4 transformation matrix.

    Returns
    -------
    poses : list
        A list of all possible poses [x, y, z, roll, yaw, pitch].

    """

    # Extract translation components
    x = given_matrix[0, 3]
    y = given_matrix[1, 3]
    z = given_matrix[2, 3]

    # Extract rotation components
    c_p = given_matrix[0, 0] / np.sqrt(given_matrix[0, 0]**2 + given_matrix[1, 0]**2)
    s_p = given_matrix[2, 0]
    c_y = given_matrix[1, 1] / np.sqrt(given_matrix[0, 0]**2 + given_matrix[1, 0]**2)
    s_y = given_matrix[0, 1] / np.sqrt(given_matrix[0, 0]**2 + given_matrix[1, 0]**2)
    c_r = given_matrix[2, 2] / np.sqrt(given_matrix[0, 2]**2 + given_matrix[1, 2]**2)
    s_r = given_matrix[2, 1] / np.sqrt(given_matrix[0, 2]**2 + given_matrix[1, 2]**2)

    # Calculate possible solutions for pitch
    pitch_1 = np.arctan2(s_p, c_p)
    pitch_2 = np.arctan2(-s_p, -c_p)

    # Calculate possible solutions for yaw
    yaw_1 = np.arctan2(s_y, c_y)
    yaw_2 = np.arctan2(-s_y, -c_y)

    # Calculate possible solutions for roll
    roll_1 = np.arctan2(s_r, c_r)
    roll_2 = np.arctan2(-s_r, -c_r)

    # Create a list to store all possible poses
    poses = []

    # Generate all possible combinations of pitch, yaw, and roll
    for pitch in [pitch_1, pitch_2]:
        for yaw in [yaw_1, yaw_2]:
            for roll in [roll_1, roll_2]:
                pose = [x, y, z, np.degrees(roll), np.degrees(yaw), np.degrees(pitch)]
                poses.append(pose)

    return poses


import numpy as np

def distance_between_poses(pose1, pose2):
    """
    Calculate the Euclidean distance between two poses.

    Parameters
    ----------
    pose1 : list or np.ndarray
        The first pose [x, y, z, roll, yaw, pitch].
    pose2 : list or np.ndarray
        The second pose [x, y, z, roll, yaw, pitch].

    Returns
    -------
    distance : float
        The Euclidean distance between the two poses.

    """

    # Convert the poses to numpy arrays (if they are lists)
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    # Calculate the position distance using the first three elements (x, y, z)
    position_distance = np.linalg.norm(pose1[:3] - pose2[:3])

    # Calculate the orientation distance using the last three elements (roll, yaw, pitch)
    orientation_distance = np.linalg.norm(pose1[3:] - pose2[3:])

    # Calculate the total Euclidean distance as a combination of position and orientation distances
    distance = np.sqrt(position_distance**2 + orientation_distance**2)

    return distance


if __name__=='__main__':


    cav_pose  = [2.3387e+02, -1.6008e+02,  1.9324e+00,  2.2712e-02,  1.5660e+02,
         8.0357e-02]
    
    ego_pose  = [1.9034e+02, -1.1936e+02,  1.9319e+00, -1.1169e-02, -2.6640e+01,
         1.0880e-01]


    matrix = x1_to_x2(cav_pose, ego_pose)

    transfer_pose = x1_to_x2(cav_pose, matrix)

