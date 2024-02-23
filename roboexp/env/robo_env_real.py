from .robot import XARM7
from .camera import RS_D455
from roboexp.utils import rpy_to_rotation_matrix, quat2rpy
import pickle
import numpy as np
import open3d as o3d
import time


class RobotExplorationReal:
    def __init__(
        self,
        calibrate_path="calibrate.pkl",
        gripper_length=0.172,
        offset=[0, 0.00856, 0.01065],
        REPLAY_FLAG=False,
    ):
        self.REPLAY_FLAG = REPLAY_FLAG
        if not self.REPLAY_FLAG:
            # If replaying, no need to use the robot to take observations
            self.robot = XARM7()
            self.camera = RS_D455(WH=[640, 480], depth_threshold=[0, 1])
        self.gripper_length = gripper_length
        self.offset = np.array(offset)
        # Initialize the calibration of the wrist camera
        self._init_calibration(calibrate_path)

    def _init_calibration(self, calibrate_path):
        if not self.REPLAY_FLAG:
            # If replaying, the calibration info has been saved into the observation
            with open(calibrate_path, "rb") as f:
                calibrate_data = pickle.load(f)
            R_cam2gripper = calibrate_data["R_cam2gripper"]
            t_cam2gripper = calibrate_data["t_cam2gripper"]
        self.cam2gripper = np.eye(4)
        if not self.REPLAY_FLAG:
            self.cam2gripper[:3, :3] = R_cam2gripper
            self.cam2gripper[:3, 3] = t_cam2gripper

    def get_observations(self, visualize=False, **kwargs):
        # Get the observations
        points, colors, depths, mask = self.camera.get_observations()
        # Get the camera2base transformation for the current pose
        cam2base = self._get_cam2base()
        # Save all the observations
        observations = {
            "wrist": {
                "position": points,
                "rgb": colors,
                "depths": depths,
                "mask": mask,
                "c2w": cam2base,
                "intrinsic": self.camera.intrinsic_matrix,
                "dist_coef": self.camera.dist_coef,
            }
        }
        if visualize:
            # Get current valid points
            valid_points = points[mask]
            valid_colors = colors[mask]
            valid_points = np.concatenate(
                [valid_points, np.ones([valid_points.shape[0], 1])], axis=1
            )
            valid_points = np.dot(cam2base, valid_points.T).T[:, :3]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(valid_points)
            cloud.colors = o3d.utility.Vector3dVector(valid_colors)

            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            o3d.visualization.draw_geometries([cloud, coordinate])
        return observations

    # Only run one action each time, to keep the same interface as the simulation environment
    # To keep the same interface, there are some extra code to do the conversion
    def run_action(
        self, action_code=0, action_parameters=[], for_camera=False, speed=200, **kwargs
    ):
        print(f"Running action {action_code} with parameters {action_parameters}")
        if action_code == 1:
            # move the end effector, parameters: qpos
            xyz = action_parameters[:3] * 1000
            rpy = quat2rpy(action_parameters[3:])
            # Need to do a 30-degree rotation in pitch if the movement is for the camera
            if for_camera:
                rpy[1] += 30
            self.robot.move_to_pose(list(xyz) + list(rpy), speed=speed)
            if for_camera:
                time.sleep(1)
        elif action_code == 2:
            # Open the gripper
            half_open = False
            if len(action_parameters) > 0:
                half_open = True
            self.robot.open_gripper(half_open=half_open)
        elif action_code == 3:
            # Close the gripper
            self.robot.close_gripper()
        elif action_code == 4:
            # Make the robot back to the default position
            self.robot.reset()

    def _get_cam2base(self):
        current_pose = self.robot.get_current_pose()
        print("Current pose: ", current_pose)
        R_cur_gripper2base = rpy_to_rotation_matrix(
            current_pose[3], current_pose[4], current_pose[5]
        )
        # Need to make srue the unit of the pose is in meters
        t_cur_gripper2base = np.array(current_pose[:3]) / 1000
        current_gripper2base = np.eye(4)
        current_gripper2base[:3, :3] = R_cur_gripper2base
        current_gripper2base[:3, 3] = t_cur_gripper2base
        # Get the camera pose in the base frame
        cam2base = np.dot(current_gripper2base, self.cam2gripper)
        # Improve the calibration based on the offset
        cam2base[:3, 3] += self.offset
        return cam2base
