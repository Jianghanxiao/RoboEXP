from roboexp.utils import visualize_pc, _init_low_level_memory, getArrowMesh
from .instance import myInstance
from .scene_graph import ActionSceneGraph
import numpy as np
import pickle
import open3d as o3d
import cv2

COUNT_TIME = False
if COUNT_TIME:
    import time

node_label_counts = {}


class RoboMemory:
    def __init__(
        self,
        lower_bound,
        higher_bound,
        voxel_size=0.02,
        real_camera=False,
        iou_thres=0.05,
        similarity_thres=0.75,
        base_dir=None,
    ):
        # The lower bound and higher bound are the boundaries of the workspace
        # voxel_size is the size of the voxel
        # Initialize the utility functions
        if type(lower_bound) == list:
            lower_bound = np.array(lower_bound)
        if type(higher_bound) == list:
            higher_bound = np.array(higher_bound)
        # Initialize the voxel grid used to store the voxels, corresponding instance, and instance-level features
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.voxel_size = voxel_size
        self.voxel_num = ((higher_bound - lower_bound) / voxel_size).astype(np.int32)

        (
            self.pcd_to_voxel,
            self.voxel_to_pcd,
            self.voxel_to_index,
            self.index_to_voxel,
            self.pcd_to_index,
            self.index_to_pcd,
        ) = _init_low_level_memory(
            lower_bound, higher_bound, voxel_size, voxel_num=self.voxel_num
        )
        # Initialize the instances in the low-level memory
        self.memory_instances = []
        self.memory_scene = {}
        self.memory_scene_avg = {}
        # The real camera option will influence the way to do the depth test
        self.real_camera = real_camera
        # Set the instnace merging threshold
        self.iou_thres = iou_thres
        self.similarity_thres = similarity_thres

        # The high-level action-conditioned scene graph
        self.action_scene_graph = None
        self.instance_node_mapping = {}
        # Record the latest observations
        self.latest_observations = None
        # Save the stuffs into the base directory
        self.base_dir = base_dir

    def get_scene_pcd(self):
        return self.index_to_pcd(np.array(list(self.memory_scene.keys()))), np.array(
            list(self.memory_scene.values())
        )

    def update_memory(
        self,
        observations,
        observation_attributes,
        object_level_labels,
        # Used to move the interior contents in memory for the drawer casae
        direct_move=None,
        # Used to filter the pixels in the observations
        filter_masks=None,
        # In the real setting, need extra_alignment
        extra_alignment=False,
        # Option to update scene graph,
        update_scene_graph=False,
        scene_graph_option=None,
        visualize=False,
    ):
        # Merge the observations in current timestep
        if COUNT_TIME:
            start = time.time()
        merged_instances, merged_scene = self._merge_observations(
            observations,
            observation_attributes,
            object_level_labels,
            filter_masks=filter_masks,
            extra_alignment=extra_alignment,
            visualize=visualize,
        )
        if COUNT_TIME:
            print(f"Memory: Merging the observations takes {time.time() - start}")
            start = time.time()
        # Update the memory
        self._update_with_current_observations(
            merged_instances,
            merged_scene,
            object_level_labels,
            observations,
            direct_move=direct_move,
            visualize=visualize,
        )
        if COUNT_TIME:
            print(f"Memory: Updating the memory takes {time.time() - start}")
            start = time.time()

        if update_scene_graph:
            self._update_scene_graph(scene_graph_option, object_level_labels)
            if COUNT_TIME:
                print(f"Memory: Updating the scene graph takes {time.time() - start}")

    def _update_scene_graph(self, scene_graph_option, object_level_labels):
        # Update the scene graph based on the memory and previous scene graph
        if scene_graph_option is None:
            # This is the case the scene graph is not initialized
            assert self.action_scene_graph is None
            # Initialize the action-conditioned scene graph, with table as the root
            root_instance = None
            part_instance_parent = {}
            for instance in self.memory_instances:
                if instance.label == "table":
                    root_instance = instance
                elif instance.label not in object_level_labels:
                    # Find the max IoU object-level instance, associate the part and the object-level instance
                    parent_object = self._find_parent_object(
                        instance, object_level_labels
                    )
                    part_instance_parent[
                        instance.instance_id
                    ] = parent_object.instance_id

            if root_instance is None:
                root_instance = myInstance(
                    "table",
                    1,
                    None,
                    None,
                    index_to_pcd=self.index_to_pcd,
                )

            # The node id is different from the instance id to make the node be consistent with the real-world setting
            self.action_scene_graph = ActionSceneGraph(
                self._get_node_id(root_instance.label),
                root_instance.label,
                root_instance,
                base_dir=self.base_dir,
            )
            # No need to further check the table in the tabletop environment
            self.action_scene_graph.root.explored = True

            self.instance_node_mapping[
                root_instance.instance_id
            ] = self.action_scene_graph.root.node_id

            # Add other objects to the scene graph
            # Loop the object node first, then the part node
            for instance in self.memory_instances:
                if instance != root_instance:
                    if instance.instance_id not in part_instance_parent.keys():
                        # Judge if the object is on the table or on other objects
                        instance_lower_bound = np.min(
                            instance.index_to_pcd(instance.voxel_indexes), axis=0
                        )
                        if instance_lower_bound[2] < 0.15:
                            # The object is near the table
                            node = self.action_scene_graph.add_object(
                                self.action_scene_graph.root,
                                self._get_node_id(instance.label),
                                instance.label,
                                instance,
                                parent_relation="on",
                            )
                            self.instance_node_mapping[
                                instance.instance_id
                            ] = node.node_id

            # For other objects not directly on the table, find the object beneath it
            for instance in self.memory_instances:
                if (
                    instance.instance_id not in self.instance_node_mapping.keys()
                    and instance.instance_id not in part_instance_parent.keys()
                ):
                    # Find the instance beneath the object
                    instance_center = np.mean(
                        instance.index_to_pcd(instance.voxel_indexes), axis=0
                    )
                    flag = False
                    for other_instance in self.memory_instances:
                        if (
                            other_instance != root_instance
                            and other_instance != instance
                        ):
                            other_instance_center = np.mean(
                                other_instance.index_to_pcd(
                                    other_instance.voxel_indexes
                                ),
                                axis=0,
                            )
                            if (
                                np.abs(other_instance_center[0] - instance_center[0])
                                < 0.1
                                and np.abs(
                                    other_instance_center[1] - instance_center[1]
                                )
                                < 0.1
                                and other_instance_center[2] < instance_center[2]
                            ):
                                # The other instance is beneath the current instance
                                node = self.action_scene_graph.add_object(
                                    self.action_scene_graph.object_nodes[
                                        self.instance_node_mapping[
                                            other_instance.instance_id
                                        ]
                                    ],
                                    self._get_node_id(instance.label),
                                    instance.label,
                                    instance,
                                    parent_relation="on",
                                )
                                flag = True
                                break
                    if not flag:
                        node = self.action_scene_graph.add_object(
                            self.action_scene_graph.root,
                            self._get_node_id(instance.label),
                            instance.label,
                            instance,
                            parent_relation="on",
                        )
                    self.instance_node_mapping[instance.instance_id] = node.node_id
            for instance in self.memory_instances:
                if instance != root_instance:
                    if instance.instance_id in part_instance_parent.keys():
                        # If the node is a part, need to reassociate the parent
                        parent_node = self.action_scene_graph.object_nodes[
                            self.instance_node_mapping[
                                part_instance_parent[instance.instance_id]
                            ]
                        ]
                        node = self.action_scene_graph.add_object(
                            parent_node,
                            self._get_node_id(instance.label),
                            instance.label,
                            instance,
                            parent_relation="belong",
                            is_part=True,
                        )
                        self.instance_node_mapping[instance.instance_id] = node.node_id
            # Analyze the attributes of the handles
            for instance in self.memory_instances:
                if instance.label == "handle":
                    node = self.action_scene_graph.object_nodes[
                        self.instance_node_mapping[instance.instance_id]
                    ]
                    parent_node = self.action_scene_graph.object_nodes[
                        self.instance_node_mapping[
                            part_instance_parent[instance.instance_id]
                        ]
                    ]
                    sibling_instances = [
                        child_node.instance
                        for child_node in parent_node.children.values()
                        if child_node.node_id != node.node_id
                    ]
                    # Detect the handle attributes and add to the node
                    (
                        node.handle_center,
                        node.handle_direction,
                        node.open_direction,
                        node.joint_type,
                        joint_info,
                    ) = self._get_handle_info(
                        instance,
                        parent_instance=parent_node.instance,
                        sibling_instances=sibling_instances,
                        visualize=False,
                    )
                    if node.joint_type == "revolute":
                        node.node_label = "door_handle"
                        node.joint_axis = joint_info["joint_axis"]
                        node.joint_origin = joint_info["joint_origin"]
                        node.side_direction = joint_info["side_direction"]
                    elif node.joint_type == "prismatic":
                        node.node_label = "drawer_handle"
        elif scene_graph_option["type"] == "handle":
            instance_id = scene_graph_option["handle_instance"].instance_id
            node_id = self.instance_node_mapping[instance_id]
            # Get the node
            node = self.action_scene_graph.object_nodes[node_id]
            node.explored = True
            # Find the new handle instances cloest to the handle center and associate it with the node
            handle_position = node.handle_center
            min_distance = None
            target_instance = None
            for instance in self.memory_instances:
                if instance.instance_id not in self.instance_node_mapping.keys():
                    distance = (
                        (
                            (
                                handle_position
                                - np.mean(
                                    instance.index_to_pcd(instance.voxel_indexes),
                                    axis=0,
                                )
                            )
                            ** 2
                        ).sum()
                    ) ** 0.5
                    if target_instance is None:
                        target_instance = instance
                        min_distance = distance
                    else:
                        if distance < min_distance:
                            target_instance = instance
                            min_distance = distance

            # Update the instance in this node with new instance
            del self.instance_node_mapping[node.instance.instance_id]
            node.update_instance(target_instance)
            self.instance_node_mapping[target_instance.instance_id] = node_id
            # Add the action node
            # Judge if the action node exists
            intervention_flag = False
            if len(node.actions) == 0:
                pre_conditions = []
                for pre_node in node.pre_nodes:
                    pre_conditions.append(list(pre_node.actions.values())[0])
                action_node = self.action_scene_graph.add_action(
                    node, self._get_node_id("open"), "open", preconditions=pre_conditions
                )
            else:
                action_node = list(node.actions.values())[0]
                intervention_flag = True
            # Add the interior instances, for all instance in the bbx of the parent object, then add it to the scene graph
            parent_node = node.parent
            parent_points = parent_node.instance.index_to_pcd(
                parent_node.instance.voxel_indexes
            )
            min_bound = np.min(parent_points, axis=0)
            max_bound = np.max(parent_points, axis=0)
            for instance in self.memory_instances:
                if instance.instance_id not in self.instance_node_mapping.keys():
                    instance_points = instance.index_to_pcd(instance.voxel_indexes)
                    interior_mask = np.logical_and(
                        np.all(instance_points >= min_bound, axis=1),
                        np.all(instance_points <= max_bound, axis=1),
                    )
                    # Judge if the instance has some points inside the parent object
                    if interior_mask.sum() > 5:
                        new_node = self.action_scene_graph.add_object(
                            action_node,
                            self._get_node_id(instance.label),
                            instance.label,
                            instance,
                            parent_relation="inside",
                        )
                        self.instance_node_mapping[
                            instance.instance_id
                        ] = new_node.node_id
                        new_node.explored = True
            if intervention_flag:
                to_delete = []
                for object_node in self.action_scene_graph.object_nodes.values():
                    if object_node.parent == action_node and object_node.instance.deleted:
                        del self.instance_node_mapping[object_node.instance.instance_id]
                        to_delete.append(object_node.node_id)
                        object_node.delete()
                        del object_node
                for node_id in to_delete:
                    del self.action_scene_graph.object_nodes[node_id]
        elif scene_graph_option["type"] == "pick_away":
            node = scene_graph_option["node"]
            action_node = self.action_scene_graph.add_action(
                node, self._get_node_id("pick"), "pick"
            )
            interior_instance = []
            for instance in self.memory_instances:
                if instance.instance_id not in self.instance_node_mapping.keys():
                    interior_instance.append(instance)

            # Add the interior instances node
            for instance in interior_instance:
                new_node = self.action_scene_graph.add_object(
                    action_node,
                    self._get_node_id(instance.label),
                    instance.label,
                    instance,
                    parent_relation="under",
                )
                self.instance_node_mapping[instance.instance_id] = new_node.node_id
        elif scene_graph_option["type"] == "pick_back":
            node = scene_graph_option["node"]
            position = node.original_position
            # Find the new instnaces cloest to the original position and associate it with the node
            min_distance = None
            target_instance = None
            for instance in self.memory_instances:
                if instance.instance_id not in self.instance_node_mapping.keys():
                    distance = (
                        (
                            (
                                position
                                - np.mean(
                                    instance.index_to_pcd(instance.voxel_indexes),
                                    axis=0,
                                )
                            )
                            ** 2
                        ).sum()
                    ) ** 0.5
                    if target_instance is None:
                        target_instance = instance
                        min_distance = distance
                    else:
                        if distance < min_distance:
                            target_instance = instance
                            min_distance = distance

            del self.instance_node_mapping[node.instance.instance_id]
            node.update_instance(target_instance)
            self.instance_node_mapping[target_instance.instance_id] = node.node_id
        elif scene_graph_option["type"] == "check":
            old_instances = scene_graph_option["old_instances"]
            old_instance_ids = [instance.instance_id for instance in old_instances]
            # Find the new instances
            new_instances = []
            for instance in self.memory_instances:
                if instance.instance_id not in old_instance_ids:
                    new_instances.append(instance)
            root = self.action_scene_graph.root

            part_instance_parent = {}
            for instance in new_instances:
                if instance.label not in object_level_labels:
                    # Find the max IoU object-level instance, associate the part and the object-level instance
                    parent_object = self._find_parent_object(
                        instance, object_level_labels
                    )
                    part_instance_parent[
                        instance.instance_id
                    ] = parent_object.instance_id
            
            # Loop the object node first, then the part node
            for instance in new_instances:
                if instance.instance_id not in part_instance_parent.keys():
                    # Judge if the object is on the table or on other objects
                    instance_lower_bound = np.min(
                        instance.index_to_pcd(instance.voxel_indexes), axis=0
                    )
                    if instance_lower_bound[2] < 0.15:
                        # The object is near the table
                        node = self.action_scene_graph.add_object(
                            self.action_scene_graph.root,
                            self._get_node_id(instance.label),
                            instance.label,
                            instance,
                            parent_relation="on",
                        )
                        self.instance_node_mapping[
                            instance.instance_id
                        ] = node.node_id

            for instance in new_instances:
                if instance.instance_id in part_instance_parent.keys():
                    # If the node is a part, need to reassociate the parent
                    parent_node = self.action_scene_graph.object_nodes[
                        self.instance_node_mapping[
                            part_instance_parent[instance.instance_id]
                        ]
                    ]
                    node = self.action_scene_graph.add_object(
                        parent_node,
                        self._get_node_id(instance.label),
                        instance.label,
                        instance,
                        parent_relation="belong",
                        is_part=True,
                    )
                    self.instance_node_mapping[instance.instance_id] = node.node_id
            # Analyze the attributes of the handles
            for instance in new_instances:
                if instance.label == "handle":
                    node = self.action_scene_graph.object_nodes[
                        self.instance_node_mapping[instance.instance_id]
                    ]
                    parent_node = self.action_scene_graph.object_nodes[
                        self.instance_node_mapping[
                            part_instance_parent[instance.instance_id]
                        ]
                    ]
                    sibling_instances = [
                        child_node.instance
                        for child_node in parent_node.children.values()
                        if child_node.node_id != node.node_id
                    ]
                    # Detect the handle attributes and add to the node
                    (
                        node.handle_center,
                        node.handle_direction,
                        node.open_direction,
                        node.joint_type,
                        joint_info,
                    ) = self._get_handle_info(
                        instance,
                        parent_instance=parent_node.instance,
                        sibling_instances=sibling_instances,
                        visualize=False,
                    )
                    if node.joint_type == "revolute":
                        node.node_label = "door_handle"
                        node.joint_axis = joint_info["joint_axis"]
                        node.joint_origin = joint_info["joint_origin"]
                        node.side_direction = joint_info["side_direction"]
                    elif node.joint_type == "prismatic":
                        node.node_label = "drawer_handle"
        print(self.instance_node_mapping)

        print("Visualizing the scene graph")
        self.action_scene_graph.visualize()

    def _find_parent_object(self, part, object_level_labels):
        for instance in self.memory_instances:
            if instance.label in object_level_labels:
                intersection = len(
                    set(part.voxel_indexes).intersection(set(instance.voxel_indexes))
                )
                if intersection / len(part.voxel_indexes) > 0.2:
                    return instance
        raise ValueError("Cannot find the parent object")

    def _get_node_id(self, label):
        global node_label_counts
        if label not in node_label_counts:
            node_label_counts[label] = 0
        else:
            node_label_counts[label] += 1
        return f"{label}_{node_label_counts[label]}"

    def _get_handle_info(
        self, instance, parent_instance, sibling_instances, visualize=False
    ):
        # Get the handle pose
        handle_pcd = instance.index_to_pcd(instance.voxel_indexes)
        # Currently the handle points are kind of clean
        handle_center = np.mean(handle_pcd, axis=0)
        # PCA to get the handle most obvious direction
        centralized_points = handle_pcd - handle_center
        cov_matrix = np.cov(centralized_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(
            cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10
        )
        handle_direction = eigenvectors[:, -1]
        floor_normal = np.array([0, 0, 1])
        # Make the handle direciton prependicular or parallel to the floor
        if np.abs(np.dot(handle_direction, floor_normal)) > 0.7:
            handle_direction = floor_normal
        else:
            handle_direction = np.cross(
                floor_normal, np.cross(handle_direction, floor_normal)
            )
            handle_direction /= np.linalg.norm(handle_direction)

        # Get the neightbour points of the handle
        voxel_indexes = instance.voxel_indexes
        neighbours, valid_mask = self._get_voxel_neighbours(voxel_indexes, 5)
        # Only consider the neighbours on the parent instance
        valid_mask *= np.isin(neighbours, list(parent_instance.voxel_indexes))
        all_neighbour_indexes = np.array(list(set(neighbours[valid_mask].tolist())))
        all_neighbour_indexes = all_neighbour_indexes[
            ~np.isin(all_neighbour_indexes, voxel_indexes)
        ]
        neighbour_pcd = self.index_to_pcd(all_neighbour_indexes).tolist()
        # Estimate the normals
        neighbour_pc = o3d.geometry.PointCloud()
        neighbour_pc.points = o3d.utility.Vector3dVector(neighbour_pcd)
        neighbour_pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=20)
        )
        neighbour_normals = np.asarray(neighbour_pc.normals)
        rounded_normals = np.round(neighbour_normals, decimals=3)
        # Remove the normals prependicular to the floor
        mask = np.abs(np.dot(rounded_normals, floor_normal)) < 0.5
        rounded_normals = rounded_normals[mask]
        # Remove the normals parallel to the handle direction
        mask = np.abs(np.dot(rounded_normals, handle_direction)) < 0.5
        rounded_normals = rounded_normals[mask]
        unique, counts = np.unique(rounded_normals, axis=0, return_counts=True)
        open_direction = unique[np.argmax(counts)]
        # Use heuristic to determine the reference direction
        reference_direction = np.zeros(3) - handle_center
        reference_direction /= np.linalg.norm(reference_direction)
        if np.dot(open_direction, reference_direction) < 0:
            open_direction = -open_direction
        # Make the open direction perpendicular to the [0, 0, 1]
        open_direction = np.cross(floor_normal, np.cross(open_direction, floor_normal))
        open_direction /= np.linalg.norm(open_direction)
        # Refine the handle direction using the open direction
        # Use heuristic to judge the joint type, if vertical, revolute, or prismatic
        if np.abs(np.dot(handle_direction, floor_normal)) < 0.7:
            handle_direction = np.cross(open_direction, floor_normal)
            joint_type = "prismatic"
        else:
            joint_type = "revolute"

        joint_info = {}
        if joint_type == "revolute":
            joint_axis = handle_direction
            # Get the side_direction
            side_direction = np.cross(handle_direction, open_direction)
            # Get the distance of the handle center to the parent node boundary in the side direction
            parent_points = parent_instance.index_to_pcd(parent_instance.voxel_indexes)
            # Get the distance between the handle center to the parent node boundary
            distances = np.dot(parent_points - handle_center, side_direction)
            max_distance = np.max(distances)
            min_distance = np.min(distances)
            # Judge if there is another handle on the side
            flag_current = False
            flag_reverse = False
            for sibling_instance in sibling_instances:
                if sibling_instance.label == "handle":
                    distance_up = np.dot(
                        np.mean(
                            sibling_instance.index_to_pcd(
                                sibling_instance.voxel_indexes
                            ),
                            axis=0,
                        )
                        - handle_center,
                        floor_normal,
                    )
                    if np.abs(distance_up) > 0.07:
                        continue
                    distance_current = np.dot(
                        np.mean(
                            sibling_instance.index_to_pcd(
                                sibling_instance.voxel_indexes
                            ),
                            axis=0,
                        )
                        - handle_center,
                        side_direction,
                    )
                    if distance_current > 0:
                        flag_current = True
                    distance_reverse = np.dot(
                        np.mean(
                            sibling_instance.index_to_pcd(
                                sibling_instance.voxel_indexes
                            ),
                            axis=0,
                        )
                        - handle_center,
                        -side_direction,
                    )
                    if distance_reverse > 0:
                        flag_reverse = True

            if flag_current:
                side_direction = -side_direction
                joint_axis = -joint_axis
            elif not flag_reverse and max_distance < np.abs(min_distance):
                side_direction = -side_direction
                joint_axis = -joint_axis

            # Find the points in the farthest open directions
            open_distances = np.dot(parent_points, open_direction)
            max_distance = np.dot(handle_center, open_direction)
            potential_points = parent_points[
                np.logical_and(
                    open_distances < max_distance, open_distances > max_distance - 0.05
                )
            ]
            # Find the points in the farthest side directions
            side_distances = np.dot(potential_points, side_direction)
            max_index = np.argmax(side_distances)
            joint_info["side_direction"] = side_direction
            joint_info["joint_origin"] = potential_points[max_index]
            joint_info["joint_axis"] = joint_axis

        if visualize:
            scene_pcd, scene_color = self.get_scene_pcd()
            # Test the estimated normal of current scen first
            scene_pc = o3d.geometry.PointCloud()
            scene_pc.points = o3d.utility.Vector3dVector(scene_pcd)
            scene_pc.colors = o3d.utility.Vector3dVector(scene_color)
            o3d.visualization.draw_geometries(
                [scene_pc, neighbour_pc], point_show_normal=True
            )
            arrow1 = getArrowMesh(
                origin=handle_center,
                end=handle_center + handle_direction,
                color=[1, 0, 0],
            )
            arrow2 = getArrowMesh(
                origin=handle_center,
                end=handle_center + open_direction,
                color=[0, 1, 0],
            )
            extra = [arrow1, arrow2]
            if joint_type == "revolute":
                arrow3 = getArrowMesh(
                    origin=joint_info["joint_origin"],
                    end=joint_info["joint_origin"] + joint_info["joint_axis"],
                    color=[0, 0, 0],
                )
                extra.append(arrow3)
            visualize_pc(scene_pcd, scene_color, extra=extra)

        return handle_center, handle_direction, open_direction, joint_type, joint_info

    def _merge_observations(
        self,
        observations,
        observation_attributes,
        object_level_labels,
        filter_masks=None,
        extra_alignment=False,
        visualize=False,
    ):
        if COUNT_TIME:
            start = time.time()

        # Align the observations to the first camera if there are multiple viewpoint
        if extra_alignment:
            # Deprecated extra_alignment in xarm7 setting
            assert extra_alignment == False
            observations = self._align_observations(observations)

        # Record the latest observations for use
        self.latest_observations = observations

        # This functions is to merge observations from different viewpoint in the same timestep, which means there is no need to understand dynamic changes
        # This function will create a voxel_grid for current timestep
        merged_instances = []
        merged_scene = {}
        _merged_scene_avg = {}
        for name, obs in observations.items():
            if "wrist" in name and "wrist" in filter_masks:
                filter_mask = filter_masks["wrist"]
            elif name in filter_masks:
                filter_mask = filter_masks[name]
            else:
                filter_mask = None
            color, pixel_index_mapping, mask = self._preprocess_observation(
                obs, filter_mask
            )
            (
                obj_labels,
                obj_confidences,
                obj_masks,
                obj_feats,
                robotarm_mask,
            ) = self._preprocess_observation_attributes(
                observation_attributes[name], mask
            )

            # Get the scene based on current observation, filter out the robot arm and do some other processing
            # The mask is also updated here
            scene, mask = self._get_scene_from_observation(
                color, pixel_index_mapping, mask, robotarm_mask
            )

            instances = []
            # Construct the instances from current observation
            for obj_label, obj_confidence, obj_mask, obj_feat in zip(
                obj_labels, obj_confidences, obj_masks, obj_feats
            ):
                if len(list(set(pixel_index_mapping[obj_mask * mask]))) == 0:
                    continue
                instance = myInstance(
                    obj_label,
                    obj_confidence,
                    list(set(pixel_index_mapping[obj_mask * mask])),
                    obj_feat,
                    index_to_pcd=self.index_to_pcd,
                )
                instances.append(instance)

            if visualize and False:
                print(f"Visualizing the PC from camera:{name}")
                visualize_pc(
                    self.index_to_pcd(np.array(list(scene.keys()))),
                    np.array(list(scene.values())),
                    instances=instances,
                )

            # Merge the scene with the merged_scene
            merged_instances, merged_scene = self._merge_scene(
                merged_instances,
                merged_scene,
                _merged_scene_avg,
                instances,
                scene,
            )

        if COUNT_TIME:
            print(f"Memory: Merging: the merge process takes {time.time() - start}")
            start = time.time()
        # Make each voxel only belonging to one object-level label
        merged_instances, merged_scene = self._filter_merged_scene(
            merged_instances, merged_scene, object_level_labels
        )
        if COUNT_TIME:
            print(f"Memory: Merging: the filtering takes {time.time() - start}")
        if visualize and False:
            print(f"Visualizing the merged PC")
            visualize_pc(
                self.index_to_pcd(np.array(list(merged_scene.keys()))),
                np.array(list(merged_scene.values())),
                instances=merged_instances,
            )

        return merged_instances, merged_scene

    def _update_with_current_observations(
        self,
        merged_instances,
        merged_scene,
        object_level_labels,
        observations,
        direct_move=None,
        visualize=False,
    ):
        # Clean the old stuffs from the old memory
        # Update the memory based on the new observation depth test: delete some out-of-date voxels
        if visualize and False:
            print(f"Visualizing the memory PC Before deleting out-of-date voxels")
            visualize_pc(
                self.index_to_pcd(np.array(list(self.memory_scene.keys()))),
                np.array(list(self.memory_scene.values())),
                instances=self.memory_instances,
            )

        if COUNT_TIME:
            start = time.time()

        self._update_memory_with_new_observations(
            observations, merged_scene, direct_move=direct_move
        )

        if visualize and False:
            print(f"Visualizing the merged PC")
            visualize_pc(
                self.index_to_pcd(np.array(list(merged_scene.keys()))),
                np.array(list(merged_scene.values())),
                instances=merged_instances,
            )

        if visualize and False:
            print(f"Visualizing the memory PC after deleting out-of-date voxels")
            visualize_pc(
                self.index_to_pcd(np.array(list(self.memory_scene.keys()))),
                np.array(list(self.memory_scene.values())),
                instances=self.memory_instances,
            )

        # Merge the scene with the merged_scene
        self.memory_instances, self.memory_scene = self._merge_scene(
            self.memory_instances,
            self.memory_scene,
            self.memory_scene_avg,
            merged_instances,
            merged_scene,
        )

        if COUNT_TIME:
            print(f"Memory: Updating: the merge process takes {time.time() - start}")
            start = time.time()

        # Make each voxel only belonging to one object-level label
        self.memory_instances, self.memory_scene = self._filter_merged_scene(
            self.memory_instances, self.memory_scene, object_level_labels
        )

        if COUNT_TIME:
            print(f"Memory: Updating: the filtering takes {time.time() - start}")

        # Merge the table instances for the tabletop environment
        # Find all the table instances
        table_instances = []
        for instance in self.memory_instances:
            if instance.label == "table":
                table_instances.append(instance)

        if len(table_instances) > 1:
            if self.action_scene_graph is None:
                main_table = table_instances[0]
            else:
                main_table = self.action_scene_graph.root.instance
            # Merge the table instances
            for instance in table_instances:
                if instance != main_table:
                    main_table.merge_instance(instance)
                    self.memory_instances.remove(instance)
                    instance.deleted = True

        if visualize:
            print(f"Visualizing the memory PC")
            visualize_pc(
                self.index_to_pcd(np.array(list(self.memory_scene.keys()))),
                np.array(list(self.memory_scene.values())),
                instances=self.memory_instances,
            )

    def _align_observations(self, observations):
        # Align the observations to the first camera if there are multiple viewpoint
        names = list(observations.keys())
        target = None
        sources = {}
        for name, obs in observations.items():
            # Get the rgb, point cloud, and the camera pose
            color = obs["rgb"]
            position = obs["position"]
            mask = obs["mask"]
            c2w = obs["c2w"]
            # Need to filter the points outside the memory range
            position_world = position @ c2w[:3, :3].T + c2w[:3, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(position_world[mask])
            pcd.colors = o3d.utility.Vector3dVector(color[mask])
            if name == names[0]:
                target = pcd
            else:
                sources[name] = pcd

        total_pcd = o3d.geometry.PointCloud()
        if len(sources) > 0:
            # When there is more than one camera
            for name, source in sources.items():
                transformation = self._calculate_alignment_colored_icp(source, target)
                observations[name]["c2w"] = np.dot(
                    transformation, observations[name]["c2w"]
                )
                total_pcd += source.transform(transformation)

        total_pcd += target
        # Align the pcd with the memory
        if len(self.memory_scene) != 0:
            memory_pcd = o3d.geometry.PointCloud()
            memory_pcd.points = o3d.utility.Vector3dVector(
                self.index_to_pcd(np.array(list(self.memory_scene.keys())))
            )
            memory_pcd.colors = o3d.utility.Vector3dVector(
                np.array(list(self.memory_scene.values()))
            )
            transformation = self._calculate_alignment_colored_icp(
                total_pcd, memory_pcd
            )
            for name, obs in observations.items():
                observations[name]["c2w"] = np.dot(
                    transformation, observations[name]["c2w"]
                )
        return observations

    def _calculate_alignment_colored_icp(self, source, target):
        # This function is to calculate the alignment between the source and target point cloud
        # The source and target are both colored point cloud
        # The output is the transformation from source to target
        # The transformation is a 4x4 matrix
        # The function is based on the http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]
        current_transformation = np.identity(4)
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )

            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down,
                target_down,
                radius,
                current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(
                    lambda_geometric=0.9999
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                ),
            )
            current_transformation = result_icp.transformation
        return current_transformation

    def _preprocess_observation(self, obs, filter_mask=None):
        # Process to get the voxel list for current mask first
        # Get the rgb, point cloud, and the camera pose
        color = obs["rgb"]
        position = obs["position"]
        mask = obs["mask"]
        c2w = obs["c2w"]
        # Need to filter the points outside the memory range
        position_world = position @ c2w[:3, :3].T + c2w[:3, 3]
        # Check the valid depth mask
        if self.real_camera:
            # Check the valid depth mask
            depth = obs["depths"]
            valid_depth_mask = depth >= 0.35
            mask = mask * valid_depth_mask
        range_mask = ((position_world - self.lower_bound) >= 0).all(-1) * (
            (self.higher_bound - position_world) >= 0
        ).all(-1)
        mask = mask * range_mask
        if filter_mask is not None:
            mask = mask * ~filter_mask
        # Update the observation mask
        obs["mask"] = mask
        # Store the voxels in the scene and the color for each voxel
        pixel_index_mapping = -np.ones((color.shape[0], color.shape[1]), dtype=np.int32)
        # Get the voxel list for current mask
        pixel_index_mapping[mask] = self.pcd_to_index(position_world[mask])
        return color, pixel_index_mapping, mask

    def _preprocess_observation_attributes(self, observation_attribute, mask):
        # Get the object labels, confidences, masks, features and the robotarm mask used for filtering
        # Get the attributes of the objects in the scene
        pred_phrases = observation_attribute["pred_phrases"]
        pred_masks = observation_attribute["pred_masks"]
        mask_feats = observation_attribute["mask_feats"]
        # Extract the labels and the confidences
        obj_labels = []
        obj_confidences = []
        obj_masks = []
        obj_feats = []
        robotarm_mask = np.zeros_like(mask)
        if pred_phrases is not None:
            for i, phrase in enumerate(pred_phrases):
                label, confidence = phrase.split(":")
                if label == "robot":
                    robotarm_mask = np.logical_or(robotarm_mask, (pred_masks[i] * mask))
                else:
                    obj_labels.append(label)
                    obj_confidences.append(float(confidence))
                    obj_masks.append(pred_masks[i] * mask)
                    obj_feats.append(mask_feats[i])

        return obj_labels, obj_confidences, obj_masks, obj_feats, robotarm_mask

    def _get_scene_from_observation(
        self, color, pixel_index_mapping, mask, robotarm_mask
    ):
        # Get the scene voxels for current observation
        scene = {}
        _scene_avg = {}
        mask = mask * np.logical_not(robotarm_mask)

        # Filter valid voxel indices and colors using the mask
        valid_voxel_indexes = pixel_index_mapping[mask]
        valid_colors = color[mask]

        # Check for invalid voxel indices
        if np.any(valid_voxel_indexes == -1):
            print("Warning: the voxel index is -1. The mask is wrong")

        # Get unique voxel indices, their index in the original array, and counts
        unique_voxel_indices, inverse_indices, counts = np.unique(
            valid_voxel_indexes, return_inverse=True, return_counts=True
        )

        # Sum colors for each unique voxel index
        summed_colors = np.zeros((unique_voxel_indices.size, valid_colors.shape[1]))
        np.add.at(summed_colors, inverse_indices, valid_colors)

        # Calculate average colors
        avg_colors = summed_colors / counts[:, np.newaxis]
        _deleted_keys = []
        for k, v, c in zip(unique_voxel_indices, avg_colors, counts):
            scene[k] = v
            _scene_avg[k] = c
            if c <= 5:
                _deleted_keys.append(k)

        voxel_indexes = np.array(list(_scene_avg.keys()))
        voxel_indexes = voxel_indexes[~np.isin(voxel_indexes, _deleted_keys)]
        neighbours, valid_mask = self._get_voxel_neighbours(voxel_indexes, 1)
        valid_mask *= np.isin(neighbours, list(_scene_avg.keys())) * ~np.isin(
            neighbours, _deleted_keys
        )
        _deleted_keys += voxel_indexes[np.where(valid_mask.sum(1) < 2)].tolist()

        for k in _deleted_keys:
            del _scene_avg[k]
            del scene[k]
            mask[pixel_index_mapping == k] = False
            pixel_index_mapping[pixel_index_mapping == k] = -1

        return scene, mask

    def _get_voxel_neighbours(self, voxel_indexes, size):
        voxels = self.index_to_voxel(voxel_indexes)
        offsets = np.arange(-size, size + 1)
        # Create a 3x3x3 grid of offsets
        di, dj, dk = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        # Flatten the grids and stack them to create a list of offset vectors
        offset_vectors = np.stack((di.ravel(), dj.ravel(), dk.ravel()), axis=-1)
        # Remove the (0,0,0) offset (central voxel)
        offset_vectors = offset_vectors[~np.all(offset_vectors == 0, axis=1)]
        # Apply the offsets to the voxel coordinates using broadcasting
        neighbours = voxels[:, np.newaxis, :] + offset_vectors
        valid_mask = np.all((neighbours >= 0) & (neighbours < self.voxel_num), axis=2)
        return self.voxel_to_index(neighbours), valid_mask

    def _merge_scene(
        self,
        merged_instances,
        merged_scene,
        _merged_scene_avg,
        instances,
        scene,
    ):
        # Merge the scene with the merged_scene
        for voxel_index, color in scene.items():
            if voxel_index not in merged_scene:
                merged_scene[voxel_index] = color
                _merged_scene_avg[voxel_index] = 1
            else:
                # Do the moving average to calculate the color for that voxel
                _merged_scene_avg[voxel_index] += 1
                merged_scene[voxel_index] = (
                    merged_scene[voxel_index]
                    + (color - merged_scene[voxel_index])
                    / _merged_scene_avg[voxel_index]
                )

        # Merge the instances with the merged_instances
        for instance in instances:
            if instance.no_merge:
                continue
            max_iou = 0
            max_iou_instance = None
            for merged_instance in merged_instances:
                if merged_instance.no_merge:
                    continue
                iou = instance.get_iou(merged_instance)
                similarity = instance.get_similarity(merged_instance)
                if (
                    iou > self.iou_thres
                    and (
                        similarity > self.similarity_thres
                        or instance.label == merged_instance.label
                    )
                    and iou > max_iou
                ):
                    max_iou = iou
                    max_iou_instance = merged_instance
            if max_iou_instance is not None:
                max_iou_instance.merge_instance(instance)
            else:
                merged_instances.append(instance)

        return merged_instances, merged_scene

    def _filter_merged_scene(self, merged_instances, merged_scene, object_level_labels):
        if COUNT_TIME:
            start = time.time()
        # Merge the instances in current merged_instances
        label_to_instances = {}
        for instance in merged_instances:
            if instance.label not in label_to_instances:
                label_to_instances[instance.label] = []
            label_to_instances[instance.label].append(instance)
        for label, instances in label_to_instances.items():
            while True:
                # Merge the instances in loop
                _do_merge = False
                for i in range(len(instances) - 1):
                    if instances[i].no_merge:
                        continue
                    for j in range(i + 1, len(instances)):
                        if instances[j].no_merge:
                            continue
                        iou = instances[i].get_iou(instances[j])
                        similarity = instances[i].get_similarity(instances[j])
                        if iou > self.iou_thres and (
                            similarity > self.similarity_thres
                            or instances[i].label == instances[j].label
                        ):
                            instances[i].merge_instance(instances[j])
                            _del_instance = instances[j]
                            instances.remove(_del_instance)
                            merged_instances.remove(_del_instance)
                            _del_instance.deleted = True
                            _do_merge = True
                            break
                    if _do_merge:
                        break
                if not _do_merge:
                    break

        if COUNT_TIME:
            print(f"Filter: instance merging takes {time.time() - start}")
            start = time.time()

        # Make each voxel only belong to one object-level label
        index_to_instances = {}
        for instance in merged_instances:
            if instance.label in object_level_labels:
                for voxel_index in instance.voxel_indexes:
                    if voxel_index not in index_to_instances:
                        index_to_instances[voxel_index] = []
                    index_to_instances[voxel_index].append(instance)
        for voxel_index, instances in index_to_instances.items():
            if len(instances) > 1:
                # Pick the insance with the highest confidence
                max_confidence = 0
                max_confidence_instance = None
                for instance in instances:
                    current_confidence = instance.confidence
                    if instance.label == "table":
                        current_confidence = 0.45
                    if current_confidence > max_confidence:
                        max_confidence = current_confidence
                        max_confidence_instance = instance
                # delete the voxel_index from other instances
                for instance in instances:
                    if instance != max_confidence_instance:
                        instance.voxel_indexes.remove(voxel_index)

        if COUNT_TIME:
            print(f"Filter: voxel assigning takes {time.time() - start}")
            start = time.time()

        # Filter out the voxels in each instance who has no neighbours of the same instance
        for instance in merged_instances:
            voxel_indexes = np.array(list(instance.voxel_indexes))
            neighbours, valid_mask = self._get_voxel_neighbours(voxel_indexes, 1)
            valid_mask *= np.isin(neighbours, list(instance.voxel_indexes))
            _deleted_keys = voxel_indexes[np.where(valid_mask.sum(1) < 2)].tolist()

            for k in _deleted_keys:
                instance.voxel_indexes.remove(k)

        if COUNT_TIME:
            print(f"Filter: voxel in instance filtering takes {time.time() - start}")
            start = time.time()

        indexes_with_label = []
        # Filter out the instances with too small number of voxels
        _deleted_instances = []
        for instance in merged_instances:
            if len(instance.voxel_indexes) < 10:
                _deleted_instances.append(instance)
            else:
                indexes_with_label += instance.voxel_indexes
        for instance in _deleted_instances:
            merged_instances.remove(instance)
            instance.deleted = True
        indexes_with_label = list(set(indexes_with_label))

        if COUNT_TIME:
            print(f"Filter: instance filtering takes {time.time() - start}")
            start = time.time()

        # Filter out the voxels that has no neighbour
        voxel_indexes = np.array(list(merged_scene.keys()))
        voxel_indexes = voxel_indexes[~np.isin(voxel_indexes, indexes_with_label)]
        neighbours, valid_mask = self._get_voxel_neighbours(voxel_indexes, 1)
        valid_mask *= np.isin(neighbours, list(merged_scene.keys()))
        _deleted_keys = voxel_indexes[np.where(valid_mask.sum(1) < 2)].tolist()

        for k in _deleted_keys:
            del merged_scene[k]

        if COUNT_TIME:
            print(f"Filter: voxel with no labels filtering takes {time.time() - start}")

        return merged_instances, merged_scene

    def _update_memory_with_new_observations(
        self, observations, merged_scene, direct_move=None, visualize=False
    ):
        if len(self.memory_scene) == 0:
            return

        if direct_move is not None:
            # When direct move is not None, move the drawer's contents back
            move_vec = direct_move["move_vec"]
            drawer_indexes = direct_move["drawer_indexes"]
            move_indexes = self.pcd_to_index(
                self.index_to_pcd(drawer_indexes) + move_vec
            )
            for i, k in enumerate(drawer_indexes):
                self.memory_scene[move_indexes[i]] = self.memory_scene[k]
                self.memory_scene_avg[move_indexes[i]] = self.memory_scene_avg[k]
                del self.memory_scene[k]
                del self.memory_scene_avg[k]
            for instance in self.memory_instances:
                intersection = set(instance.voxel_indexes).intersection(
                    set(drawer_indexes)
                )
                instance.voxel_indexes = list(
                    set(instance.voxel_indexes) - intersection
                )
                instance.voxel_indexes = list(
                    set(instance.voxel_indexes)
                    | set(move_indexes[np.isin(drawer_indexes, list(intersection))])
                )

        # This function is to remove some voxels that fail to pass the depth test in the new observations
        voxel_indexes = np.array(list(self.memory_scene.keys()))
        deleted_indexes = self._depth_test(
            voxel_indexes,
            observations,
            0.02,
            use_color=True,
            color_depth_thres=0,
            color_thres=0.1,
        )
        # No need to delete the voxels that's also in the new scene, but this may influence the dynamic change, let's see
        deleted_indexes = list(set(deleted_indexes) - set(merged_scene.keys()))

        for i, k in enumerate(deleted_indexes):
            del self.memory_scene[k]
            del self.memory_scene_avg[k]

        _deleted_instances = []
        for instance in self.memory_instances:
            intersection = set(instance.voxel_indexes).intersection(
                set(deleted_indexes)
            )
            instance.voxel_indexes = list(set(instance.voxel_indexes) - intersection)

            if len(instance.voxel_indexes) < 10:
                _deleted_instances.append(instance)

        for instance in _deleted_instances:
            self.memory_instances.remove(instance)
            instance.deleted = True

        if False:
            display = []
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.index_to_pcd(deleted_indexes))
            pcd.paint_uniform_color(np.array([0, 0, 1]))
            display.append(pcd)
            visualize_pc(
                self.index_to_pcd(np.array(list(self.memory_scene.keys()))),
                np.array(list(self.memory_scene.values())),
                # instances=self.memory_instances,
                extra=display,
            )

    def _depth_test(
        self,
        voxel_indexes,
        observations,
        depth_thres=0.02,
        use_color=False,
        color_depth_thres=0,
        color_thres=0.1,
    ):
        # This function is to remove some voxels that fail to pass the depth test in the new observations
        deleted_indexes = []
        pc_memory = self.index_to_pcd(voxel_indexes)
        if use_color:
            pc_color = np.array(list(self.memory_scene.values()))
        for name, obs in observations.items():
            height, width = obs["rgb"].shape[:2]
            w2c = np.linalg.inv(obs["c2w"])

            pc_camera = pc_memory @ w2c[:3, :3].T + w2c[:3, 3]
            # Only consider the voxels in front of the camera
            if not self.real_camera:
                # OpenGL camera coordinate
                mask = pc_camera[:, 2] < 0
                intrinsic = obs["intrinsic"]
                depth = obs["position"][..., 2]
                pc_image = np.zeros_like(pc_camera)
                pc_image[mask, 0] = pc_camera[mask, 1] / pc_camera[mask, 2]
                pc_image[mask, 1] = -pc_camera[mask, 0] / pc_camera[mask, 2]
                pc_image[mask, 2] = 1

                pc_xy = np.zeros((pc_image.shape[0], 2), dtype=np.int64)
                pc_xy[mask] = (
                    np.dot(intrinsic, pc_image[mask].T).T[:, :2].astype(np.int64)
                )
            else:
                # OpenCV camera coordinate
                mask = pc_camera[:, 2] > 0
                depth = obs["depths"]
                intrinsic = obs["intrinsic"]
                dist_coef = obs["dist_coef"]
                pc_xy = np.zeros((pc_camera.shape[0], 2), dtype=np.int64)
                pc_xy[mask] = self._project_point_to_pixel(
                    pc_camera[mask], intrinsic, dist_coef
                )

            # Only consider the voxels in the camera view
            mask *= (
                (pc_xy[:, 0] >= 0)
                * (pc_xy[:, 0] <= height - 1)
                * (pc_xy[:, 1] >= 0)
                * (pc_xy[:, 1] <= width - 1)
            )
            # Consider the voxels whose depth is bigger than the current observation (more close to the camera)
            if not self.real_camera:
                # The OpenGL camera coordinate, distance is negative
                depth_differ = (
                    pc_camera[mask, 2] - depth[pc_xy[mask, 0], pc_xy[mask, 1]]
                )
            else:
                # The OpenCV camera coordinate, distance is positive
                depth_differ = (
                    depth[pc_xy[mask, 0], pc_xy[mask, 1]] - pc_camera[mask, 2]
                )
            valid_mask = depth_differ > depth_thres
            if use_color:
                color_differ = (
                    (
                        (pc_color[mask] - obs["rgb"][pc_xy[mask, 0], pc_xy[mask, 1]])
                        ** 2
                    ).sum(1)
                ) ** 0.5
                valid_mask = np.logical_or(
                    valid_mask,
                    (depth_differ > color_depth_thres) * (color_differ > color_thres),
                )
            deleted_indexes += list(voxel_indexes[mask][valid_mask])

        return deleted_indexes

    def _project_point_to_pixel(self, points, intrinsic, dist_coef):
        # The points here should be in the camera coordinate, n*3
        points = np.array(points)
        pixels = []

        pixels = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            intrinsic,
            dist_coef,
        )[0][:, 0, :]

        return pixels[:, ::-1]

    def diff_memory(self, other_memory_scene, care_position, care_radius):
        # Get the new voxels in current memory compared to the other memory
        # Based on the neighbour voxels to judge the new voxels
        if COUNT_TIME:
            import time

            start = time.time()

        other_voxel_indexes = np.array(list(other_memory_scene.keys()))
        current_voxel_indexes = np.array(list(self.memory_scene.keys()))
        neighbours, valid_mask = self._get_voxel_neighbours(other_voxel_indexes, 1)
        new_voxel_indexes = current_voxel_indexes[
            ~np.isin(current_voxel_indexes, neighbours[valid_mask])
        ]
        # Get mask with pcds in the care region
        new_pcds = self.index_to_pcd(new_voxel_indexes)
        region_mask = np.linalg.norm(new_pcds - care_position, axis=1) < care_radius
        new_voxel_indexes = new_voxel_indexes[region_mask]

        if COUNT_TIME:
            print(f"Memory: Diff: the diff process takes {time.time() - start}")
        if False:
            display = []
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                self.index_to_pcd(new_voxel_indexes)
            )
            pcd.paint_uniform_color(np.array([0, 0, 1]))
            display.append(pcd)
            visualize_pc(
                self.index_to_pcd(np.array(list(self.memory_scene.keys()))),
                np.array(list(self.memory_scene.values())),
                # instances=self.memory_instances,
                extra=display,
            )

        return new_voxel_indexes

    def remove_instance(self, instance):
        # Remove the instance from the memory
        for voxel_index in instance.voxel_indexes:
            del self.memory_scene[voxel_index]
            del self.memory_scene_avg[voxel_index]
        self.memory_instances.remove(instance)
        instance.deleted = True

    def save_memory(self, save_path):
        # Save the memory to the save_path
        pickle.dump(
            {
                "memory_instances": [
                    instance.get_dict() for instance in self.memory_instances
                ],
                "memory_scene": self.memory_scene,
                "memory_scene_avg": self.memory_scene_avg,
                "lower_bound": self.lower_bound,
                "higher_bound": self.higher_bound,
                "voxel_size": self.voxel_size,
                "voxel_num": self.voxel_num,
                "instance_node_mapping": self.instance_node_mapping,
                "action_scene_graph": self.action_scene_graph.to_dict(),
            },
            open(save_path, "wb"),
        )

    def visualize_memory(self):
        # Visualize the memory
        visualize_pc(
            self.index_to_pcd(np.array(list(self.memory_scene.keys()))),
            np.array(list(self.memory_scene.values())),
            instances=self.memory_instances,
        )

    def save_to_ply(self):
        # Save the memory to the save_path
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self.index_to_pcd(np.array(list(self.memory_scene.keys())))
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.array(list(self.memory_scene.values()))
        )
        o3d.io.write_point_cloud("memory.ply", pcd)
