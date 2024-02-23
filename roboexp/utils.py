import numpy as np
import torch
import matplotlib.pyplot as plt
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, mat2euler, euler2quat, euler2mat
import matplotlib.transforms as mtransforms
import open3d as o3d
import datetime
import cv2


# Calculate the transformation matrix through the lookAt logic
# Borrow the logic from https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#create-and-mount-a-camera
def get_pose_look_at(
    eye, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0])
):
    # Convert to numpy array
    if type(eye) is list:
        eye = np.array(eye)
    if type(target) is list:
        target = np.array(target)
    if type(up) is list:
        up = np.array(up)
    up /= np.linalg.norm(up)
    # Calcualte the rotation matrix
    front = target - eye
    front /= np.linalg.norm(front)
    left = np.cross(up, front)
    left = left / np.linalg.norm(left)
    new_up = np.cross(front, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([front, left, new_up], axis=1)
    mat44[:3, 3] = eye
    return mat44


# The function is to calculate the qpos of the end effector given the front and up vector
# front is defined as the direction of the gripper
# Up is defined as the reverse direction with special shape
def get_pose_from_front_up_end_effector(front, up):
    # Convert to numpy array
    if type(front) is list:
        front = np.array(front)
    if type(up) is list:
        up = np.array(up)
    front = front / np.linalg.norm(front)
    up = -up / np.linalg.norm(up)
    left = np.cross(up, front)
    left = left / np.linalg.norm(left)
    new_up = np.cross(front, left)

    rotation_mat = np.eye(3)
    rotation_mat[:3, :3] = np.stack([left, new_up, front], axis=1)
    quat = mat2quat(rotation_mat)

    return quat


def quat2rpy(quat):
    # Convert to numpy array
    if type(quat) is list:
        quat = np.array(quat)
    # Convert to rpy
    rpy = np.array(quat2euler(quat, axes="sxyz")) / np.pi * 180
    return rpy


def rpy2quat(roll, pitch, yaw):
    # TODO: no use for now, if use, need some test to check if it's corect
    # Convert to quat
    quat = euler2quat(
        roll / 180 * np.pi, pitch / 180 * np.pi, yaw / 180 * np.pi, axes="sxyz"
    )
    return quat


def rotation_matrix_to_rpy(R):
    # TODO: no use for now, if use, need some test to check if it's corect
    # Ensure R is a numpy array
    if type(R) is list:
        R = np.array(R)

    rpy = np.array(mat2euler(R, axes="sxyz")) / np.pi * 180
    return rpy


def rpy_to_rotation_matrix(roll, pitch, yaw):
    R = euler2mat(
        roll / 180 * np.pi, pitch / 180 * np.pi, yaw / 180 * np.pi, axes="sxyz"
    )

    return R


# colors = {}
# Functions used for visualization of the segmentaiton
def display_image(img, boxes=None, labels=None, masks=None):
    # global colors

    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    if masks is not None:
        for mask in masks:
            _show_mask(mask, plt.gca(), random_color=True)
    # if masks is not None:
    #     for mask, label in zip(masks, labels):
    #         if label.split(":")[0] not in colors:
    #             colors[label.split(":")[0]] = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #         _show_mask(mask, plt.gca(), random_color=False, color=colors[label.split(":")[0]])
    if boxes is not None and labels is not None:
        for box, label in zip(boxes, labels):
            _show_box(box, plt.gca(), label)
    plt.axis("off")
    plt.show()


# Funcitons used to savw the image with boxes
def save_image(img, boxes, save_path):
    img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(save_path, img)


def _show_mask(
    mask, ax, random_color=False, color=np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def _show_box(box, ax, label, no_label=False):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    if not no_label:
        ax.text(x0, y0, label)


def get_bbx_from_mask(mask):
    # Find the coordinates of non-zero (white) pixels in the mask
    non_zero_pixels = np.argwhere(mask > 0)
    if np.shape(non_zero_pixels)[0] == 0:
        return None

    # Calculate the minimum and maximum X and Y coordinates
    min_x = min(non_zero_pixels[:, 1])
    max_x = max(non_zero_pixels[:, 1])
    min_y = min(non_zero_pixels[:, 0])
    max_y = max(non_zero_pixels[:, 0])

    # Create the XYWH format bounding box
    bbox = [min_x, min_y, max_x, max_y]
    return bbox


# Functions used for visualization of the DenseCLIP
def display_denseCLIP(img, dense_feats, dense_confidences, text_feats):
    # Calculate the similarity scores between the pixel-wise clip feature and the text features
    _simfunc = torch.nn.CosineSimilarity(dim=-1)
    # Only calculate the similarity for the pixels with valid clip feature
    valid_mask = dense_confidences > 0
    _sim = torch.zeros_like(torch.from_numpy(dense_confidences)).float()
    _sim[valid_mask] = _simfunc(
        torch.from_numpy(dense_feats[valid_mask]).float(),
        torch.from_numpy(text_feats).float(),
    )

    # Display the image
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("Image")
    # Display the similarity
    ax[1].imshow(_sim.numpy(), vmin=0, vmax=1)
    ax[1].axis("off")
    ax[1].set_title("Raw Similarity")

    # Do scaling to highlight the high similarity
    _sim[valid_mask] = (_sim[valid_mask] - _sim[valid_mask].min()) / (
        _sim[valid_mask].max() - _sim[valid_mask].min() + 1e-12
    )
    im = ax[2].imshow(_sim.numpy())
    ax[2].axis("off")
    # Add the color bar
    cax = _add_right_cax(ax[2], pad=0.02, width=0.02)
    fig.colorbar(im, cax)
    ax[2].set_title("Scaled Similarity")

    plt.show()


def _add_right_cax(ax, pad, width):
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad, axpos.y0, axpos.x1 + pad + width, axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)
    return cax


def visualize_pc(
    pc, color=None, show_coordinate=False, instances=None, extra=None, index_to_pcd=None
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    display = [pcd]
    if extra is not None:
        display += extra
    if show_coordinate:
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()
        display.append(coordinate)
    o3d.visualization.draw_geometries(display)

    if instances is not None:
        instance_pcds = []
        for instance in instances:
            instance_pcd = o3d.geometry.PointCloud()
            instance_pcd.points = o3d.utility.Vector3dVector(
                instance.index_to_pcd(instance.voxel_indexes)
                if index_to_pcd is None
                else index_to_pcd(instance["voxel_indexes"])
            )
            instance_pcd.paint_uniform_color(np.array([0, 1, 0]))
            instance_pcds.append(instance_pcd)
            print(
                instance.instance_id
                if index_to_pcd is None
                else instance["instance_id"]
            )
            o3d.visualization.draw_geometries([pcd, instance_pcd])
        print("Number of instances: ", len(instance_pcds))


# Borrow ideas and codes from H. Sánchez's answer
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def getArrowMesh(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.08,
        cylinder_height=1.0 * vec_len,
        cylinder_radius=0.04,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = _caculate_align_mat(vec_Arr / vec_len)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow


def _get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array(
        [
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ]
    )
    return qCross_prod_mat


def _caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = _get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = _get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = (
            np.eye(3, 3)
            + z_c_vec_mat
            + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
        )
    qTrans_Mat *= scale
    return qTrans_Mat


def _init_low_level_memory(lower_bound, higher_bound, voxel_size, voxel_num):
    def pcd_to_voxel(pcds):
        if type(pcds) == list:
            pcds = np.array(pcds)
        # The pc is in numpy array with shape (..., 3)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.floor((pcds - lower_bound) / voxel_size).astype(np.int32)
        return voxels

    def voxel_to_pcd(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The pc is in numpy array with shape (..., 3)
        pcds = voxels * voxel_size + lower_bound
        return pcds

    def voxel_to_index(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        indexes = (
            voxels[..., 0] * voxel_num[1] * voxel_num[2]
            + voxels[..., 1] * voxel_num[2]
            + voxels[..., 2]
        )
        return indexes

    def index_to_voxel(indexes):
        if type(indexes) == list:
            indexes = np.array(indexes)
        # The index is in numpy array with shape (...,)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.zeros((indexes.shape + (3,)), dtype=np.int32)
        voxels[..., 2] = indexes % voxel_num[2]
        indexes = indexes // voxel_num[2]
        voxels[..., 1] = indexes % voxel_num[1]
        voxels[..., 0] = indexes // voxel_num[1]
        return voxels

    def pcd_to_index(pcds):
        # The pc is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        voxels = pcd_to_voxel(pcds)
        indexes = voxel_to_index(voxels)
        return indexes

    def index_to_pcd(indexes):
        # The index is in numpy array with shape (...,)
        # The pc is in numpy array with shape (..., 3)
        voxels = index_to_voxel(indexes)
        pcds = voxel_to_pcd(voxels)
        return pcds

    return (
        pcd_to_voxel,
        voxel_to_pcd,
        voxel_to_index,
        index_to_voxel,
        pcd_to_index,
        index_to_pcd,
    )


def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return current_time
