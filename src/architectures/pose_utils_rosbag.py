"""Plotting + data format and reference frame conversion utilities specific to
the pose estimation problem - while dealing with rosbag datasets
"""

from geometry_msgs.msg import Pose, Point
import ros_utils
import pose_utils
from pose_utils import DEG_TO_RAD
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"


def plot_orientation(gt_poses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['r'] / DEG_TO_RAD for pose in gt_poses],
        mode='markers',
        name='roll'
    ))
    fig.add_trace(go.Scatter(
        y=[ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['p'] / DEG_TO_RAD for pose in gt_poses],
        mode='markers',
        name='pitch'
    ))
    fig.add_trace(go.Scatter(
        y=[ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] / DEG_TO_RAD for pose in gt_poses],
        mode='markers',
        name='yaw'
    ))
    return fig


def plot_positions(gt_poses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=[pose.position.x for pose in gt_poses],
        mode='markers',
        name='x'
    ))
    fig.add_trace(go.Scatter(
        y=[pose.position.y for pose in gt_poses],
        mode='markers',
        name='y'
    ))
    fig.add_trace(go.Scatter(
        y=[pose.position.z for pose in gt_poses],
        mode='markers',
        name='z'
    ))
    return fig

def viz_positions_ros(gt_poses, viz_heading_offset, name='ros', footprint='processing_plant', subset=None):
    return pose_utils.viz_positions(
        x_list=[pose.position.x for pose in gt_poses],
        y_list=[pose.position.y for pose in gt_poses],
        yaw_list=[ros_utils.angle_in_range(ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] - viz_heading_offset, np.pi) for pose in gt_poses],
        name=name,
        footprint=footprint,
        subset=subset
    )

def viz_orientations_ros(fig, gt_poses, subset=None):
    return pose_utils.viz_orientations(
        fig=fig,
        x_list=[pose.position.x for pose in gt_poses],
        y_list=[pose.position.y for pose in gt_poses],
        yaw_list=[ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] for pose in gt_poses],
        subset=subset
    )


def make_label_from_pose(pose, type):
    '''used by:
    1. RosbagParser -> convert from rover ground truth to optical ground truth and then to label
    '''
    d = np.sqrt(pose.position.x ** 2 + pose.position.y ** 2)
    theta = np.arctan2(pose.position.y, pose.position.x)    # range -pi;pi and correct quadrant
    if type == 'optical':
        yaw = (ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] - theta - np.pi / 2.)
    elif type == 'rover':
        yaw = (ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] - theta - np.pi)
    else:
        print(f"in make_label_from_pose(): type {type} not recognized")
    if yaw > np.pi:
        yaw = yaw - (2. * np.pi)
    if yaw < -np.pi:
        yaw = yaw + (2. * np.pi)
    if yaw < -np.pi or yaw > np.pi:
        print(f'[error] out of bounds yaw = {yaw / DEG_TO_RAD} deg')
    return d, theta, yaw


def make_pose_from_label(label, type, target_object):
    d, theta, yaw = label
    # defining z: correct only on flat ground and with sensor mount horizontal
    if type == 'optical':
        r = - np.pi / 2.
        if target_object == 'processing_plant':
            z = 1.704
        elif 'small_' in target_object:  # it's a rover
            z = 0.507
        elif target_object == 'world':
            z = 2.154
        else:
            print(f'in make_pose_from_label(): target_object {target_object} not supported')
    else:
        print(f'in make_pose_from_label(): type {type} not supported')
        r = 0.

    optical_pose = Pose(
        position=Point(
            x=d * np.cos(theta),
            y=d * np.sin(theta),
            z=z
        ),
        # true only on flat ground and with sensor mount horizontal
        orientation=ros_utils.euler_dict_to_quaternion_msg({
            'r': r,
            'p': 0.,
            'y': yaw + theta + np.pi / 2.
        })
    )
    return optical_pose

def convert_optical_labels_to_rover_labels(
    d_optical, theta_optical, yaw_optical,
    rover_name='small_hauler_1'
):
    d_rover, theta_rover, yaw_rover = [], [], []
    for d, theta, yaw in zip(d_optical, theta_optical, yaw_optical):
        optical_pose = make_pose_from_label(
            [d, theta, yaw],
            type='optical',
            target_object='small_excavator_1'
        )
        rover_pose = ros_utils.optical_pose_to_rover_pose(
            ros_utils.inverse_transform_stamped(ros_utils.make_tf_optical_in_rover(rover_name)),
            optical_pose=optical_pose
        )
        rover_d, rover_theta, rover_yaw = make_label_from_pose(rover_pose, type='rover')
        d_rover.append(rover_d)
        theta_rover.append(rover_theta)
        yaw_rover.append(rover_yaw)
    return d_rover, theta_rover, yaw_rover
