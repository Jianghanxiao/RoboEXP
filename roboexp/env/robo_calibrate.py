from .robot import XARM7
from .camera import RS_D455
from roboexp.utils import rpy_to_rotation_matrix
import cv2
import time
import numpy as np
import pickle


class RoboCalibrate:
    def __init__(self):
        self.robot = XARM7()
        self.camera = RS_D455(WH=[640, 480], depth_threshold=[0, 2])
        # Initialize the calibration board, this should corresponds to the real board
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (6, 9),
            squareLength=0.03,
            markerLength=0.022,
            dictionary=self.dictionary,
        )
        # Get the camera intrinsic parameters
        self.intrinsic_matrix, self.dist_coef = (
            self.camera.intrinsic_matrix,
            self.camera.dist_coef,
        )
        # Specify the poses used for the hand-eye calirbation
        self.poses = [
            # Pose 1
            [263.731323, 26.548101, 387.849731, -173.282892, 23.043217, -10.037705],
            [268.941742, -11.555555, 410.964691, -175.391262, 31.465467, -10.371338],
            [277.991882, -10.138917, 429.85141, -173.874184, 18.252946, 6.556585],
            [263.773407, -5.113694, 424.882446, -172.188485, 25.204127, -3.061944],
            [286.275574, 0.48078, 388.695129, 169.20475, 19.140916, 0.776644],
            [268.292938, 12.322101, 396.480072, -179.760409, 25.735603, 5.932634],
            # Pose 2
            [363.941864, 165.311935, 365.18042, 170.33531, 23.462622, -89.093734],
            [361.088531, 163.330643, 361.326599, 175.971153, 25.606, -84.645118],
            [385.088837, 166.307434, 358.266205, 164.065376, 20.410017, -86.535477],
            [375.216125, 177.15918, 369.692169, 174.871532, 23.472076, -90.972119],
            [363.056671, 151.710983, 356.220184, 169.952632, 20.056273, -95.139126],
            [342.381866, 146.086578, 347.81427, 167.615021, 24.641712, -101.520616],
            # Pose 3
            [318.695923, -183.728088, 377.404907, -168.434752, 26.44057, 50.166122],
            [319.03186, -172.396683, 373.445709, -178.302919, 28.056712, 49.804929],
            [321.078705, -172.923706, 399.83429, -164.069673, 33.00701, 45.33626],
            [305.544403, -179.392273, 360.520477, -165.835185, 25.209054, 52.448843],
            [303.150604, -200.736572, 366.663544, 179.644958, 19.041909, 46.242622],
            [328.160797, -174.039917, 394.734589, -177.424575, 22.076923, 52.370233],
        ]

    def set_calibration_poses(self, poses):
        self.poses = poses

    def calibrate(self, visualize=True):
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []
        rgbs = []
        depths = []
        point_list = []
        masks = []

        for pose in self.poses:
            # Move to the pose and wait for 5s to make it stable
            self.robot.move_to_pose(pose=pose, wait=True, ignore_error=True)
            time.sleep(5)

            # Calculate the markers
            points, colors, depth_img, mask = self.camera.get_observations()
            calibration_img = colors.copy()
            calibration_img *= 255
            calibration_img = calibration_img.astype(np.uint8)
            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_RGB2BGR)
            # calibration_img, depth_img = self.camera.get_observations(only_raw=True)

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                image=calibration_img,
                dictionary=self.dictionary,
                parameters=None,
            )

            # Calculate the charuco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=calibration_img,
                board=self.board,
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.dist_coef,
            )

            print("number of corners: ", len(charuco_corners))

            if visualize:
                cv2.aruco.drawDetectedCornersCharuco(
                    image=calibration_img,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids,
                )
                cv2.imshow("cablibration", calibration_img)
                cv2.waitKey(1)

            rvec = None
            tvec = None
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                self.intrinsic_matrix,
                self.dist_coef,
                rvec,
                tvec,
            )
            if not retval:
                raise ValueError("pose estimation failed")

            reprojected_points, _ = cv2.projectPoints(
                self.board.getChessboardCorners()[charuco_ids, :],
                rvec,
                tvec,
                self.intrinsic_matrix,
                self.dist_coef,
            )

            # Reshape for easier handling
            reprojected_points = reprojected_points.reshape(-1, 2)
            charuco_corners = charuco_corners.reshape(-1, 2)

            # Calculate the error
            error = np.sqrt(
                np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
            ).mean()

            print("Reprojection Error:", error)

            # if error < 0.3:
            print("Pose estimation succeed!")
            # Save the transformation of board2cam
            R_board2cam.append(cv2.Rodrigues(rvec)[0])
            t_board2cam.append(tvec[:, 0])
            # Save the transformation of the gripper2base
            current_pose = self.robot.get_current_pose()
            print("Current pose: ", current_pose)
            R_gripper2base.append(
                rpy_to_rotation_matrix(
                    current_pose[3], current_pose[4], current_pose[5]
                )
            )
            t_gripper2base.append(np.array(current_pose[:3]) / 1000)
            # Save the rgb and depth images
            rgbs.append(colors)
            depths.append(depth_img)
            point_list.append(points)
            masks.append(mask)

        R_base2gripper = []
        t_base2gripper = []
        for i in range(len(R_gripper2base)):
            R_base2gripper.append(R_gripper2base[i].T)
            t_base2gripper.append(-R_gripper2base[i].T @ t_gripper2base[i])

        # Do the robot-world hand-eye calibration
        (
            R_base2board,
            t_base2board,
            R_gripper2cam,
            t_gripper2cam,
        ) = cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_board2cam,
            t_world2cam=t_board2cam,
            R_base2gripper=R_base2gripper,
            t_base2gripper=t_base2gripper,
            R_base2world=None,
            t_base2world=None,
            R_gripper2cam=None,
            t_gripper2cam=None,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        R_cam2gripper = R_gripper2cam.T
        t_cam2gripper = -R_gripper2cam.T @ t_gripper2cam[:, 0]

        R_board2base = R_base2board.T
        t_board2base = -R_base2board.T @ t_base2board[:, 0]

        results = {}
        results["R_cam2gripper"] = R_cam2gripper
        results["t_cam2gripper"] = t_cam2gripper
        results["R_board2base"] = R_board2base
        results["t_board2base"] = t_board2base
        results["R_gripper2base"] = R_gripper2base
        results["t_gripper2base"] = t_gripper2base
        results["R_board2cam"] = R_board2cam
        results["t_board2cam"] = t_board2cam
        results["rgbs"] = rgbs
        results["depths"] = depths
        results["point_list"] = point_list
        results["masks"] = masks
        results["poses"] = self.poses

        self._save_results(results, "calibrate.pkl")

        print(R_cam2gripper)
        print(t_cam2gripper)

    def test_pose(self):
        for pose in self.poses:
            # Move to the pose and wait for 5s to make it stable
            self.robot.move_to_pose(pose=pose, wait=True, ignore_error=True)
            time.sleep(5)

            current_pose = self.robot.get_current_pose()
            print("Current pose: ", current_pose)

            points, colors, depth_img, mask = self.camera.get_observations()
            calibration_img = colors.copy()
            calibration_img *= 255
            calibration_img = calibration_img.astype(np.uint8)
            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_RGB2BGR)

            cv2.imshow("cablibration", calibration_img)
            cv2.waitKey(0)

    def _save_results(self, results, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
