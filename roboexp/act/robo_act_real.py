from .robo_act import RoboAct
import numpy as np
from roboexp.utils import get_pose_look_at

COUNT_TIME = False
if COUNT_TIME:
    import time


class RoboActReal(RoboAct):
    def __init__(
        self,
        robo_exp,
        robo_percept,
        robo_memory,
        object_level_labels,
        base_dir,
        REPLAY_FLAG=False,
    ):
        super().__init__(
            robo_exp,
            robo_percept,
            robo_memory,
            object_level_labels,
            base_dir=base_dir,
            gripper_length=robo_exp.gripper_length,
            open_num=10,
            prismatic_open_unit=0.015,
            revolute_open_unit=5 / 180 * np.pi,
            REPLAY_FLAG=REPLAY_FLAG,
        )
        # Rewrite the camera poses
        camera_positions = {
            "camera_0": np.array([0.2, 0, 0.7]),
            "camera_1": np.array([0.25, 0.5, 0.4]),
            "camera_2": np.array([0.2, 0, 0.7]),
            "camera_3": np.array([0.25, -0.5, 0.4]),
        }
        target_far = np.array([0.8, 0, 0.2])
        target_low = np.array([0.8, 0, 0])
        self.camera_poses = [
            get_pose_look_at(
                eye=np.array(camera_positions["camera_0"]), target=target_far
            ),
            get_pose_look_at(
                eye=np.array(camera_positions["camera_1"]), target=target_far
            ),
            get_pose_look_at(
                eye=np.array(camera_positions["camera_2"]), target=target_low
            ),
            get_pose_look_at(
                eye=np.array(camera_positions["camera_3"]), target=target_far
            ),
        ]
        self.extra_alignment = False
