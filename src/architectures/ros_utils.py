import numpy as np
import rospy
from tf import transformations as ts
import tf2_geometry_msgs
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, PoseArray, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
from std_msgs.msg import Header


class ros_utils():
    ''' Static helper methods for manipulating the rosbags used for building this dataset
    and converting gazebo ground truth poses
    '''
    @staticmethod
    def angle_in_range(angle, range_max):
        # reduce the angle
        angle = angle % (range_max * 2)
        # force it to be the positive remainder, so that 0 <= angle < 360
        angle = (angle + range_max * 2) % (range_max * 2)
        # force into the minimum absolute value residue class, so that -180 < angle <= 180
        if (angle > range_max):
            angle -= range_max * 2
        return angle

    @staticmethod
    def dict_to_point(d):
        # addressing necessary to avoid bugs in unordered dictionaries
        p = Point(
            x=d['x'],
            y=d['y'],
            z=d['z']
        )
        return p

    @staticmethod
    def point_to_dict(p):
        d = {
            'x': p.x,
            'y': p.y,
            'z': p.z,
        }
        return d

    @staticmethod
    def pose_to_dict(p):
        d = {
            'position': ros_utils.point_to_dict(p.position),
            'orientation': ros_utils.quaternion_msg_to_dict(p.orientation)
        }
        return d

    @staticmethod
    def dict_to_pose(d):
        pose = Pose(
            position=ros_utils.dict_to_point(d['position']),
            orientation=ros_utils.dict_to_quaternion_msg(d['orientation'])
        )
        return pose

    @staticmethod
    def euler_dict_to_quaternion_msg(d):
        # d must be a dictionary with euler angles with these specific keys
        q = ts.quaternion_from_euler(d['r'], d['p'], d['y'], axes='sxyz')
        return Quaternion(*q)

    @staticmethod
    def dict_to_quaternion_msg(d):
        q = Quaternion(
            x=d['x'],
            y=d['y'],
            z=d['z'],
            w=d['w']
        )
        return q

    @staticmethod
    def yaw_to_quaternion_msg(yaw):
        q = ts.quaternion_from_euler(0.0, 0.0, float(yaw), axes='sxyz')
        return Quaternion(*q)

    @staticmethod
    def quaternion_msg_to_euler_dict(q):
        # q must be a Quaternion ROS msg
        q_list = [
            q.x,
            q.y,
            q.z,
            q.w
        ]
        euler = ts.euler_from_quaternion(q_list, axes='sxyz')  # axis = 's'tatic frame
        q_dict = {
            'r': euler[0],
            'p': euler[1],
            'y': euler[2],
        }
        return q_dict

    @staticmethod
    def quaternion_msg_to_quaternion(msg):
        # d must be a dictionary with euler angles with these specific keys
        q_list = [msg.x, msg.y, msg.z, msg.w]
        return q_list

    @staticmethod
    def quaternion_msg_to_dict(msg):
        # msg must be a Quaternion ROS msg
        q_dict = {
            'x': msg.x,
            'y': msg.y,
            'z': msg.z,
            'w': msg.w,
        }
        return q_dict

    @staticmethod
    def quaternion_msg_to_wxyz_array(msg):
        ''' careful: quaternion w/x/y/z convention of transforms3d, not ROS
        '''
        # msg must be a Quaternion ROS msg
        q = np.array([msg.w, msg.x, msg.y, msg.z])
        return q

    @staticmethod
    def pose_to_pose_with_covariance_stamped(pose, frame):
        pose_with_cov_stamped = PoseWithCovarianceStamped()
        pose_with_cov_stamped.pose.pose = pose
        pose_with_cov_stamped.header.frame_id = frame
        return pose_with_cov_stamped

    @staticmethod
    def init_pose_wcs():
        pose_wcs = PoseWithCovarianceStamped()
        pose_wcs.pose.pose.orientation = Quaternion(w=1.0)
        return pose_wcs

    @staticmethod
    def init_pose():
        return Pose(
            orientation=Quaternion(w=1.0)
        )

    @staticmethod
    def pprint_deg_from_rad(rads, precision=1):
        '''for quick printout of radian angles in degrees
        assumes an iterable of floats in radians
        returns a list of strings in degrees to the given precision
        '''
        return [f'{rad * 180 / np.pi:.{precision}f}' for rad in rads]

    @staticmethod
    def pprint_pose(pose, precision=1):
        '''for quick printout of pose with quaternion in rpy (in deg)
        '''
        print(f'x: {pose.position.x:.{precision}f} m \ny: {pose.position.y:.{precision}f} m \nz: {pose.position.z:.{precision}f} m')
        rpy = ros_utils.quaternion_msg_to_euler_dict(pose.orientation).values()
        print('roll: {} deg \npitch: {} deg \nyaw: {} deg'.format(*ros_utils.pprint_deg_from_rad(list(rpy), precision=precision)))

    @staticmethod
    def inverse_transform(orig_transform):
        ''' only available in C++ API
        This python version is inspired by
        https://www.hepeng.me/ros-tf-whoever-wrote-the-python-tf-api-f-ked-up-the-concept/
        '''
        trans = orig_transform.translation
        rot = orig_transform.rotation

        transform_m = ts.concatenate_matrices(
            ts.translation_matrix((trans.x, trans.y, trans.z)),
            ts.quaternion_matrix((rot.x, rot.y, rot.z, rot.w))
        )

        inversed_trans_m = ts.inverse_matrix(transform_m)

        inversed_transform = Transform(
            translation=Vector3(*ts.translation_from_matrix(inversed_trans_m)),
            rotation=Quaternion(*ts.quaternion_from_matrix(inversed_trans_m))
        )
        return inversed_transform

    @staticmethod
    def inverse_transform_stamped(orig_transform):
        inversed_transform = ros_utils.inverse_transform(orig_transform.transform)

        return TransformStamped(
            header=Header(
                seq=orig_transform.header.seq,
                stamp=orig_transform.header.stamp,
                frame_id=orig_transform.child_frame_id,
            ),
            child_frame_id=orig_transform.header.frame_id,
            transform=inversed_transform
        )

    @staticmethod
    def position_delta(position_ref, position_target):
        delta = Point()
        delta.x = position_target.x - position_ref.x
        delta.y = position_target.y - position_ref.y
        delta.z = position_target.z - position_ref.z
        return delta

    @staticmethod
    def rpy_delta(orientation_ref, orientation_target):
        rpy_ref = ros_utils.quaternion_msg_to_euler_dict(orientation_ref).values()
        rpy_target = ros_utils.quaternion_msg_to_euler_dict(orientation_target).values()
        rpy_delta = np.array(list(rpy_ref)) - np.array(list(rpy_target))
        return rpy_delta

    @staticmethod
    def distance_squared(position_ref, position_target, horizontal=False):
        delta = ros_utils.position_delta(position_ref, position_target)
        if horizontal:
            return delta.x**2 + delta.y**2
        else:
            return delta.x**2 + delta.y**2 + delta.z**2

    @staticmethod
    def local_pose_to_world(pose_local, ref_world):
        pose_world = tf2_geometry_msgs.do_transform_pose(
            PoseStamped(
                Header(),
                pose_local
            ),
            TransformStamped(
                transform=Transform(
                    translation=ref_world.position,
                    rotation=ref_world.orientation
                )
            )
        )
        return pose_world.pose
    @staticmethod
    def optical_pose_to_rover_pose(tf_rover_in_optical, optical_pose):
        ''' returns the pose of the rover in the same reference frame that the optical_pose is expressed
        Get the proper transform with:
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        tf_rover_in_optical = tfBuffer.lookup_transform(
            source_frame=f'{rover_name}_small_chassis',
            target_frame=f'{rover_name}_left_camera_optical',
            time=rospy.Time()
        )
        '''
        rover_pose = ros_utils.local_pose_to_world(
            Pose(
                position=tf_rover_in_optical.transform.translation,
                orientation=tf_rover_in_optical.transform.rotation
            ),
            optical_pose
        )
        return rover_pose

    @staticmethod
    def rover_pose_to_optical_pose(tf_optical_in_rover, rover_pose):
        ''' returns the pose of the left camera optical frame in the same reference frame that the rover_pose is expressed
        Get the proper transform with:
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        tf_optical_in_rover = tfBuffer.lookup_transform(
            target_frame=f'{rover_name}_small_chassis',
            source_frame=f'{rover_name}_left_camera_optical',
            time=rospy.Time()
        )
        '''
        optical_pose = ros_utils.local_pose_to_world(
            Pose(
                position=tf_optical_in_rover.transform.translation,
                orientation=tf_optical_in_rover.transform.rotation
            ),
            rover_pose
        )
        return optical_pose

    @staticmethod
    def make_tf_optical_in_rover(rover_name, stamp=None):
        ''' quick and dirty to get the transform between camera and rover frame
        Dirty because: assumes sensor mount pan and tilt are zero
        Tested to cause a few mm and a few degrees error
        The clean way is to query the lookup_transform server (see above)
        '''
        tf_optical_in_rover = TransformStamped(
            header=Header(
                frame_id=f'{rover_name}_small_chassis'
            ) if stamp is None else Header(
                frame_id=f'{rover_name}_small_chassis',
                stamp=stamp
            ),
            child_frame_id=f'{rover_name}_left_camera_optical',
            transform=Transform(
                translation=Point(
                    x=0.589,
                    y=0.581,
                    z=0.509
                ),
                rotation=Quaternion(
                    x=-0.5,
                    y=0.5,
                    z=-0.5,
                    w=0.5
                )
            )
        )
        return tf_optical_in_rover

# for unit tests
if __name__ == "__main__":
    try:
        # unit tests placeholder
        pass

    except rospy.ROSInterruptException:
        pass
