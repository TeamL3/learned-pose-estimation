from ros_utils import ros_utils
import numpy as np
import pickle
import io
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "plotly_white"

DEG_TO_RAD = np.pi / 180.0
RAD_TO_SCALED = 1 / np.pi  # ensure angle is within [-1, 1]
MAX_DEPTH = 25  # m
METERS_TO_SCALED = 1 / MAX_DEPTH  # ensure depth is within [0, 1]
INTENSITY_TO_SCALED = 1 / 255.  # # ensure pixel intensities are within [0, 1]


def which_target(tf_data_path):
    if 'hopper' in tf_data_path:
        target='processing_plant'
    elif 'scout' in tf_data_path:
        target='small_scout_1'
    else:
        target='small_excavator_1'
    return target


def show_rgbd(img, figsize=(20, 5), format='rgbd'):
    # TODO: refactor conditions, this grew a bit too bespoke ><
    n_channels = np.atleast_3d(img).shape[-1]
    print(f"img shape: {img.shape}, found {n_channels} channels")
    if 'h' in format:  # 'hsv' or subset
        if format == 'hsvd':
            fig, ax = plt.subplots(3, 2, figsize=(figsize[0], figsize[1] * 3))
            ax = ax.ravel()
            h = ax[0].imshow(img[:, :, 0])
            plt.colorbar(h, ax=ax[0])
            ax[0].set_title("Hue")

            s = ax[2].imshow(img[:, :, 1])
            plt.colorbar(s, ax=ax[2])
            ax[2].set_title("Saturation")

            v = ax[4].imshow(img[:, :, 2])
            plt.colorbar(v, ax=ax[4])
            ax[4].set_title("Value")

            d = ax[1].imshow(img[:, :, -1])
            plt.colorbar(d, ax=ax[1])
            ax[1].set_title("Depth")
        elif format == 'hvd' or format == 'hv' or format == 'hd':
            fig, ax = plt.subplots(2, 2, figsize=(figsize[0], figsize[1] * 2))
            ax = ax.ravel()
            h = ax[0].imshow(img[:, :, 0])
            plt.colorbar(h, ax=ax[0])
            ax[0].set_title("Hue")

            if format == 'hvd' or format == 'hv':
                v = ax[2].imshow(img[:, :, 1])
                plt.colorbar(v, ax=ax[2])
                ax[2].set_title("Value")
            if 'd' in format:
                d = ax[1].imshow(img[:, :, -1])
                plt.colorbar(d, ax=ax[1])
                ax[1].set_title("Depth")
        else:
            fig, ax = plt.subplots(3, 1, figsize=(figsize[0], figsize[1] * 3))
            ax = ax.ravel()
            h = ax[0].imshow(img[:, :, 0])
            plt.colorbar(h, ax=ax[0])
            ax[0].set_title("Hue")
            s = ax[1].imshow(img[:, :, 1])
            plt.colorbar(s, ax=ax[1])
            ax[1].set_title("Saturation")
            v = ax[2].imshow(img[:, :, 2])
            plt.colorbar(v, ax=ax[2])
            ax[2].set_title("Value")

    elif 'd' in format:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        d = ax[1].imshow(img[:, :, -1])
        plt.colorbar(d, ax=ax[1])
        ax[1].set_title("Depth")
        if n_channels == 4:
            im = ax[0].imshow(img[:, :, :3])
            ax[0].set_title("RGB")
            plt.colorbar(im, ax=ax[0])
        elif format == 'rbd':
            rgb = np.zeros((img.shape[0], img.shape[1], 3))
            rgb[:, :, 0] = img[:, :, 0]
            rgb[:, :, 2] = img[:, :, 1]
            im = ax[0].imshow(rgb)
            ax[0].set_title("Red and Blue")
            plt.colorbar(im, ax=ax[0])
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if n_channels == 1:
            d = ax.imshow(img)
            plt.colorbar(d, ax=ax)
            ax.set_title("single channel")
        elif n_channels == 3:
            im = ax.imshow(img)
            plt.colorbar(im, ax=ax)
            ax.set_title("RGB")
        elif format == 'rb':
            rgb = np.zeros((img.shape[0], img.shape[1], 3))
            rgb[:, :, 0] = img[:, :, 0]
            rgb[:, :, 2] = img[:, :, 1]
            im = ax.imshow(rgb)
            ax.set_title("Red and Blue")
            plt.colorbar(im, ax=ax)
        else:
            print(f"n_channels = {n_channels} with format = {format} are not supported")

    return fig, ax


def plotly2array(fig):
    # convert a Plotly interactive figure to a numpy array (to save as a static image)
    # from https://community.plotly.com/t/converting-byte-object-to-numpy-array/40189/2
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    arr = np.asarray(img)
    if np.max(arr) > 1:  # convert from 0-255 range to 0-1 range
        arr = arr/255.
    return arr


def hist_labels(labels, title='label histogram', figsize=(20, 5)):
    rows = 2
    fig, ax = plt.subplots(rows, 3, figsize=(figsize[0], figsize[1] * rows))
    ax = ax.ravel()
    ax[0].hist([d for d, _, _ in labels])
    ax[0].set_title("d (raw)")
    ax[1].hist([theta for _, theta, _ in labels])
    ax[1].set_title("theta (raw)")
    ax[2].hist([yaw for _, _, yaw in labels])
    ax[2].set_title("yaw (raw)")

    ax[3].hist([d / METERS_TO_SCALED for d, _, _ in labels])
    ax[3].set_title("d (meters)")
    ax[4].hist([theta / RAD_TO_SCALED / DEG_TO_RAD for _, theta, _ in labels])
    ax[4].set_title("theta (degrees)")
    ax[5].hist([yaw / RAD_TO_SCALED / DEG_TO_RAD for _, _, yaw in labels])
    ax[5].set_title("yaw (degrees)")
    fig.suptitle(title, fontsize=16)
    return fig


def subplot_labels(labels, viz='raw', subset=50, title='label subplots'):
    if subset is None or subset > len(labels):
        subset = len(labels)

    # create stacked subplots with each output
    if viz == 'raw':
        units = ['raw'] * 3
    elif viz == 'human':
        units = ['meters', 'degrees', 'degrees']
    else:
        print(f"viz '{viz}' not supported. only 'raw' or 'human' are valid")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[f'distance ({units[0]})', f'theta ({units[1]})', f'yaw ({units[2]})']
    )
    fig.add_trace(go.Scatter(
            y=labels[:subset, 0] if viz == 'raw' else labels[:subset, 0]/ METERS_TO_SCALED,
            mode='markers',
            name='d',
            ),
            row=1, col=1
        )
    fig.add_trace(go.Scatter(
            y=labels[:subset, 1] if viz == 'raw' else labels[:subset, 1]/ RAD_TO_SCALED / DEG_TO_RAD,
            mode='markers',
            name='theta',
            ),
            row=2, col=1
        )
    fig.add_trace(go.Scatter(
            y=labels[:subset, 2] if viz == 'raw' else labels[:subset, 2]/ RAD_TO_SCALED / DEG_TO_RAD,
            mode='markers',
            name='yaw',
            ),
            row=3, col=1
        )
    fig.update_layout(
        # height=800,
        # width=1000,
        title_text=title
    )
    return fig


def hist_errors(d_true, theta_true, yaw_true, d_list, theta_list, yaw_list, n_bins=20, figsize=(20, 5)):
    error_d = [(true - pred) for true, pred in zip(d_true, d_list)]
    error_theta = []
    for true, pred in zip(theta_true, theta_list):
        delta = (true - pred) / DEG_TO_RAD
        if delta < - 180.:
            error_theta.append(delta + 360)
        elif delta > 180:
            error_theta.append(delta - 360)
        else:
            error_theta.append(delta)
    error_yaw = [(true - pred) / DEG_TO_RAD for true, pred in zip(yaw_true, yaw_list)]

    rows = 2
    fig, ax = plt.subplots(rows, 3, figsize=(figsize[0], figsize[1] * rows))
    ax = ax.ravel()
    hist = ax[0].hist(np.abs(error_d), bins=n_bins)
    ax[0].set_title("absolute error distance (meters)")
    mean = np.mean(np.abs(error_d))
    median = np.median(np.abs(error_d))
    ax[0].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    ax[0].axvline(median, color='g', linestyle='dotted', linewidth=2)
    ax[0].legend((f'mean    = {mean:.2f}', f'median = {median:.2f}'), numpoints=1, loc=1)

    hist = ax[1].hist(np.abs(error_theta), bins=n_bins)
    ax[1].set_title("absolute error theta (degrees)")
    median = np.median(np.abs(error_theta))
    mean = np.mean(np.abs(error_theta))
    ax[1].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    ax[1].axvline(median, color='g', linestyle='dotted', linewidth=2)
    ax[1].legend((f'mean    = {mean:.2f}', f'median = {median:.2f}'), numpoints=1, loc=1)

    hist = ax[2].hist(np.abs(error_yaw), bins=n_bins)
    ax[2].set_title("absolute error yaw (degrees)")
    median = np.median(np.abs(error_yaw))
    mean = np.mean(np.abs(error_yaw))
    ax[2].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    ax[2].axvline(median, color='g', linestyle='dotted', linewidth=2)
    ax[2].legend((f'mean    = {mean:.2f}', f'median = {median:.2f}'), numpoints=1, loc=1)

    hist = ax[3].hist(error_d, bins=n_bins)
    ax[3].set_title("error distance (meters)")
    mean = np.mean(error_d)
    median = np.median(error_d)
    ax[3].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    ax[3].axvline(median, color='g', linestyle='dotted', linewidth=2)
    ax[3].legend((f'mean    = {mean:.2f}', f'median = {median:.2f}'), numpoints=1, loc=1)

    hist = ax[4].hist(error_theta, bins=n_bins)
    ax[4].set_title("error theta (degrees)")
    mean = np.mean(error_theta)
    median = np.median(error_theta)
    ax[4].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    ax[4].axvline(median, color='g', linestyle='dotted', linewidth=2)
    ax[4].legend((f'mean    = {mean:.2f}', f'median = {median:.2f}'), numpoints=1, loc=1)

    hist = ax[5].hist(error_yaw, bins=n_bins)
    ax[5].set_title("error yaw (degrees)")
    mean = np.mean(error_yaw)
    median = np.median(error_yaw)
    ax[5].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    ax[5].axvline(median, color='g', linestyle='dotted', linewidth=2)
    ax[5].legend((f'mean    = {mean:.2f}', f'median = {median:.2f}'), numpoints=1, loc=1)
    return fig


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
    return viz_positions(
        x_list=[pose.position.x for pose in gt_poses],
        y_list=[pose.position.y for pose in gt_poses],
        yaw_list=[ros_utils.angle_in_range(ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] - viz_heading_offset, np.pi) for pose in gt_poses],
        name=name,
        footprint=footprint,
        subset=subset
    )


def viz_positions(x_list, y_list, yaw_list, name='predict', footprint='small_excavator_1', subset=None):
    if subset is None or subset > len(x_list):
        subset = len(x_list)
    fig = go.Figure(
        data=go.Scatter(
            x=x_list[:subset],
            y=y_list[:subset],
            customdata=[yaw / DEG_TO_RAD for yaw in yaw_list],  # convert to degrees
            mode='markers',
            name=name,
            hovertemplate="yaw = %{customdata:.1f} deg"
        )
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    if footprint == 'processing_plant':
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-1.8, y0=-1.8, x1=1.8, y1=1.8,
            line_color="LightSeaGreen",
        )
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=-1.4, y0=-3.8, x1=1.4, y1=-2.1,
            line_color="LightSeaGreen",
        )
        fig.add_annotation(
            x=0,
            y=0,
            text="PP",
            xref="x",
            yref="y",
            showarrow=False,
            font_size=20
        )
    elif 'small_' in footprint:
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=-1.43 / 2, y0=-1.29 / 2, x1=1.43 / 2, y1=1.29 / 2,
            line_color="LightSeaGreen",
        )
        if 'excavator' in footprint:
            fig.add_shape(
                type="rect",
                xref="x", yref="y",
                x0=1.43 / 2, y0=-0.25, x1=1.43 / 2 + 1, y1=0.25,
                line_color="LightSeaGreen",
            )
        fig.add_annotation(
            x=0,
            y=0,
            text=footprint.split('_')[1],
            xref="x",
            yref="y",
            showarrow=False,
            font_size=20
        )
    else:
        print(f"footprint '{footprint}' not supported")
    return fig


def viz_positions_additional(fig, x_list, y_list, name='ground truth', footprint='processing_plant', subset=None):
    if subset is None or subset > len(x_list):
        subset = len(x_list)
    fig.add_trace(
        go.Scatter(
            x=x_list[:subset],
            y=y_list[:subset],
            mode='markers',
            marker_color='red',
            name=name
        )
    )
    return fig


def viz_orientations_ros(fig, gt_poses, subset=None):
    return viz_orientations(
        fig=fig,
        x_list=[pose.position.x for pose in gt_poses],
        y_list=[pose.position.y for pose in gt_poses],
        yaw_list=[ros_utils.quaternion_msg_to_euler_dict(pose.orientation)['y'] for pose in gt_poses],
        subset=subset
    )


def viz_orientations(fig, x_list, y_list, yaw_list, subset=None):
    if subset is None or subset > len(x_list):
        subset = len(x_list)
    # add arrows to show the yaw orientation (rover is pointing towards the processing plant)
    annotations = []
    xscale, yscale = 20, 20  # empirical scale factor for length of arrows
    for x, y, yaw in zip(x_list[:subset], y_list[:subset], yaw_list[:subset]):
        annotations.append(dict(
            x=x,
            y=y,
            ax=xscale * np.cos(yaw),
            ay=yscale * np.sin(yaw) * -1,  # -1 is just because plotly arrows go towards the point instead of outwards (I think)
            showarrow=True,
            arrowcolor="blue",
            arrowsize=1,
            arrowwidth=1,
            arrowhead=0,
            opacity=0.5
        ))
    fig.update_layout(
        annotations=annotations,
        width=800, height=800
    )
    return fig, annotations


def viz_orientations_additional(fig, annotations, x_list, y_list, yaw_list, subset=None):
    if subset is None or subset > len(x_list):
        subset = len(x_list)
    # add arrows to show the yaw orientation (rover is pointing towards the processing plant)
    additional_annotations = []
    xscale, yscale = 20, 20  # empirical scale factor for length of arrows
    for x, y, yaw in zip(x_list[:subset], y_list[:subset], yaw_list[:subset]):
        additional_annotations.append(dict(
            x=x,
            y=y,
            ax=xscale * np.cos(yaw),
            ay=yscale * np.sin(yaw) * -1,  # -1 is just because plotly arrows go towards the point instead of outwards (I think)
            showarrow=True,
            arrowcolor="red",
            arrowsize=1,
            arrowwidth=1,
            arrowhead=0,
            opacity=0.5
        ))
    fig.update_layout(
        annotations=annotations + additional_annotations,
        width=800, height=800
    )
    return fig, annotations + additional_annotations


def viz_link(fig, x0_list, y0_list, x1_list, y1_list, subset=None):
    if subset is None or subset > len(x0_list):
        subset = len(x0_list)
    # add arrows to show link between predict and ground truth
    for x0, y0, x1, y1 in zip(x0_list[:subset], y0_list[:subset], x1_list[:subset], y1_list[:subset]):
        fig.add_shape(
            type='line',
            xref='x', yref='y',
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(
                color="grey",
                width=1,
                dash="dot",
            )
        )
    return fig


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


from geometry_msgs.msg import Pose, Point
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


def compare_optical_poses(
    d_true, theta_true, yaw_true,
    d_pred, theta_pred, yaw_pred,
    yaw_viz_offset,
    footprint,
    subset=None
):
    # recommend: yaw_viz_offset=np.pi/2 to have the arrows point to the target_object
    # for visualizations: convert back to cartesian coordinates, in the reference frame of the lander
    x_true = [r * np.cos(theta) for r, theta in zip(d_true, theta_true)]
    y_true = [r * np.sin(theta) for r, theta in zip(d_true, theta_true)]
    yaw_true = [yaw + theta + np.pi / 2. for yaw, theta in zip(yaw_true, theta_true)]
    x_pred = [r * np.cos(theta) for r, theta in zip(d_pred, theta_pred)]
    y_pred = [r * np.sin(theta) for r, theta in zip(d_pred, theta_pred)]
    yaw_pred = [yaw + theta + np.pi / 2. for yaw, theta in zip(yaw_pred, theta_pred)]
    fig = viz_positions(x_pred, y_pred, yaw_pred, footprint=footprint, name='predict', subset=subset)
    fig, annotations = viz_orientations(fig, x_pred, y_pred, [y + yaw_viz_offset for y in yaw_pred], subset=subset)
    fig = viz_positions_additional(fig, x_true, y_true, name='ground truth', subset=subset)
    fig, annotations = viz_orientations_additional(fig, annotations, x_true, y_true, [y + yaw_viz_offset for y in yaw_true], subset=subset)
    fig = viz_link(fig, x_pred, y_pred, x_true, y_true, subset=subset)
    return fig


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


def compare_optical_and_rover_poses(
    d_optical, theta_optical, yaw_optical,
    d_rover, theta_rover, yaw_rover,
    subset=None
):
    x_list = [r * np.cos(theta) for r, theta in zip(d_optical, theta_optical)]
    y_list = [r * np.sin(theta) for r, theta in zip(d_optical, theta_optical)]
    yaw_list = [yaw + theta + np.pi / 2. for yaw, theta in zip(yaw_optical, theta_optical)]
    x_rover = [r * np.cos(theta) for r, theta in zip(d_rover, theta_rover)]
    y_rover = [r * np.sin(theta) for r, theta in zip(d_rover, theta_rover)]
    yaw_rover = [yaw + theta + np.pi for yaw, theta in zip(yaw_rover, theta_rover)]

    fig = viz_positions(x_list, y_list, yaw_list, name='optical', subset=subset)
    fig, annotations = viz_orientations(fig, x_list, y_list, [y + np.pi / 2. for y in yaw_list], subset=subset)  # add pi/2 for vizualization to make the arrow point to the front of the rover
    fig = viz_positions_additional(fig, x_rover, y_rover, name='rover', subset=subset)
    fig, annotations = viz_orientations_additional(fig, annotations, x_rover, y_rover, yaw_rover, subset=subset)
    fig = viz_link(fig, x_list, y_list, x_rover, y_rover, subset=subset)
    return fig


def plot_optical_poses(
    d_optical, theta_optical, yaw_optical,
    title='training',
    subset=None,
    footprint='processing_plant'
):
    x_list = [r * np.cos(theta) for r, theta in zip(d_optical, theta_optical)]
    y_list = [r * np.sin(theta) for r, theta in zip(d_optical, theta_optical)]
    yaw_list = [yaw + theta + np.pi / 2. for yaw, theta in zip(yaw_optical, theta_optical)]

    fig = viz_positions(x_list, y_list, yaw_list, name=title, subset=subset, footprint=footprint)
    fig, annotations = viz_orientations(fig, x_list, y_list, [y + np.pi / 2. for y in yaw_list], subset=subset)  # add pi/2 for vizualization to make the arrow point to the front of the rover
    fig.update_layout(title=title)
    return fig


def compare_each_output(
    d_true, theta_true, yaw_true,
    d_list, theta_list, yaw_list,
    subset=50):
    if subset > len(d_true):
        subset = len(d_true)

    # preliminary: scale the angles to degrees for humans
    theta_true = [_ / DEG_TO_RAD for _ in theta_true]
    yaw_true = [_ / DEG_TO_RAD for _ in yaw_true]
    theta_list = [_ / DEG_TO_RAD for _ in theta_list]
    yaw_list = [_ / DEG_TO_RAD for _ in yaw_list]

    # create stacked subplots with each output
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=['distance (meters)', 'theta (degress)', 'orientation (degrees)']
    )
    fig.add_trace(go.Scatter(
        y=d_list[:subset],
        mode='markers',
        name='d_pred',
        marker_color='blue'
        ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        y=d_true[:subset],
        mode='markers',
        name='d_true',
        marker_color='red'
        ),
        row=1, col=1
    )
    for x0, y0, x1, y1 in zip(range(len(d_list[:subset])), d_list[:subset], range(len(d_true[:subset])), d_true[:subset]):
        fig.add_shape(
            type='line',
            xref='x', yref='y',
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(
                color="grey",
                width=1,
                dash="dot",
            ),
            row=1, col=1
        )
    fig.add_trace(go.Scatter(
        y=theta_list[:subset],
        mode='markers',
        name='theta_pred',
        marker_color='blue'
        ),
        row=2, col=1
    )
    fig.add_trace(go.Scatter(
        y=theta_true[:subset],
        mode='markers',
        name='theta_true',
        marker_color='red'
        ),
        row=2, col=1
    )
    for x0, y0, x1, y1 in zip(range(len(theta_list[:subset])), theta_list[:subset], range(len(theta_true[:subset])), theta_true[:subset]):
        fig.add_shape(
            type='line',
            xref='x', yref='y',
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(
                color="grey",
                width=1,
                dash="dot",
            ),
            row=2, col=1
        )
    fig.add_trace(go.Scatter(
        y=yaw_list[:subset],
        mode='markers',
        name='yaw_pred',
        marker_color='blue'
        ),
        row=3, col=1
    )
    fig.add_trace(go.Scatter(
        y=yaw_true[:subset],
        mode='markers',
        name='yaw_true',
        marker_color='red'
        ),
        row=3, col=1
    )
    for x0, y0, x1, y1 in zip(range(len(yaw_list[:subset])), yaw_list[:subset], range(len(yaw_true[:subset])), yaw_true[:subset]):
        fig.add_shape(
            type='line',
            xref='x', yref='y',
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(
                color="grey",
                width=1,
                dash="dot",
            ),
            row=3, col=1
        )

    fig.update_layout(
        height=800,
        width=1000,
        title_text="Comparison of prediction and ground truth for each output"
    )
    return fig
