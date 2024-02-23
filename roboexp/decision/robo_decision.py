import numpy as np
from roboexp.utils import visualize_pc, save_image, get_bbx_from_mask
import open3d as o3d
import os
from .models import MyGPTV
import re


class RoboDecision:
    def __init__(self, robo_memory, base_dir, REPLAY_FLAG=False):
        # If replaying, just use the cached answers
        self.REPLAY_FLAG = REPLAY_FLAG
        self.robo_memory = robo_memory
        self.base_dir = base_dir
        # Create the dir for the observations
        dir_name = f"{self.base_dir}/decisions"
        if not os.path.exists(dir_name):
            # Create directory if it doesn't exist
            os.makedirs(dir_name)
        # Intialize the planned action list
        self.action_list = []
        self.save_index = 0

    def update_action_list(self):
        # Used to check the latest scene graph and add potential exploration actions to the action list
        # Update with the latest scene graph and latest observations used to further explore
        self._update_state_from_memory()

    def get_action(self):
        # Get one action to do
        # Need to use LMM to verify if the action is doable or something blocks the action
        if (
            self.action_list[-1][1] == "open_close"
            and self.action_list[-1][0].verified == False
        ):
            node = self.action_list[-1][0]
            node.verified = True
            observations = self.robo_memory.latest_observations
            node_observations = self._preprocess_observation(
                observations, node, is_verify=True
            )
            self._verify_action_gptv(node, node_observations)

        return self.action_list.pop()

    def _verify_action_gptv(self, node, node_observations):
        # Locate the instances around the current node
        node_center = node.handle_center
        if node.joint_type == "revolute":
            joint_origin = node.joint_origin
            joint_axis = node.joint_axis
        around_instances = []
        around_labels = []
        around_distances = []
        for instance in self.robo_memory.memory_instances:
            instance_center = np.mean(
                instance.index_to_pcd(instance.voxel_indexes), axis=0
            )
            if (
                np.linalg.norm(instance_center - node_center) < 0.2
                and node.parent.node_id
                != self.robo_memory.instance_node_mapping[instance.instance_id]
                and instance.label != "table"  # Ignore the root label
                and self.robo_memory.action_scene_graph.object_nodes[
                    self.robo_memory.instance_node_mapping[instance.instance_id]
                ].parent.node_label
                == "table"
            ):
                around_instances.append(instance)
                around_labels.append(instance.label)
                if node.joint_type == "revolute":
                    pcd = instance.index_to_pcd(instance.voxel_indexes)
                    p_ins = pcd - joint_origin
                    dis_ins = min(np.linalg.norm(np.cross(p_ins, joint_axis), axis=1))
                    around_distances.append(str(dis_ins.round(2)))

        if len(around_instances) > 0:
            # Get the part label for the node
            label = node.node_label
            part_label = label.split("_")[0]
            # Use the GPTV to verify if the action is doable based on current observations
            gptv = MyGPTV(
                config_path="LLM/config/roboexp_gptv_verify.json",
                base_path=f"{self.base_dir}/decisions/verify_cache",
                REPLAY_FLAG=self.REPLAY_FLAG,
            )
            if part_label != "door":
                response = gptv(
                    query=f"The process of open {label}. Given object list [{', '.join(around_labels)}]. Please select all objects from the given list that may block the part open. The below images are the observation of the corresponding handle in the scene",
                    query_image_paths=node_observations,
                )
            else:
                side = "left" if joint_origin[1] > node_center[1] else "right"
                # Get the distance between the handle center to the joint axis line
                p_handle = node_center - joint_origin
                dis_handle = str(
                    np.linalg.norm(np.cross(p_handle, joint_axis)).round(2)
                )
                response = gptv(
                    query=f"The process of open {label}. Given object list [{', '.join(around_labels)}], the distances between the objects to the revolute axis of the door are [{', '.join(around_distances)}]. The door is open to {side}. The distance between the door handle to the revolute axis is {dis_handle}. Please select all objects from the given list that may block the part open. The below images are the observation of the corresponding handle in the scene",
                    query_image_paths=node_observations,
                )
            # Get the texts after the final answer
            # Regular expression to extract content after "[Final Answer]"
            match = re.search(r"\[Final Answer\]:\s*(.*)", response)
            # Extracting the content after "[Final Answer]"
            final_answer = match.group(1).lower() if match else None

        final_obstacle_nodes = []
        # Analyze the answer to find all objects that need to be removed
        for i in range(len(around_instances)):
            if around_instances[i].label in final_answer:
                final_obstacle_nodes.append(
                    self.robo_memory.action_scene_graph.object_nodes[
                        self.robo_memory.instance_node_mapping[
                            around_instances[i].instance_id
                        ]
                    ]
                )

        current_action = self.action_list.pop()
        for obstacle_node in final_obstacle_nodes:
            self.action_list.append(
                (
                    obstacle_node,
                    "pick_back",
                )
            )
            node.pre_nodes.append(obstacle_node)
        self.action_list.append(current_action)
        for obstacle_node in final_obstacle_nodes[::-1]:
            self.action_list.append(
                (
                    obstacle_node,
                    "pick_away",
                )
            )

    def is_done(self):
        # Check if the exploration is done
        return len(self.action_list) == 0

    def _update_state_from_memory(self):
        # Update the state from the memory
        action_scene_graph = self.robo_memory.action_scene_graph
        observations = self.robo_memory.latest_observations
        # Locate the node that is not explored and not in the plan list
        unplanned_node_list = self._get_unplanned_node_list(action_scene_graph)
        # Loop the node to check its attributes
        for unplanned_node in unplanned_node_list:
            actions = self._get_action_from_node(unplanned_node, observations)
            self.action_list += actions

        self.save_index += 1

    def _get_action_from_node(self, node, observations):
        # Infer the proper action for the node
        # Prepare the RGB with masks on the instance
        node_observations = self._preprocess_observation(observations, node)
        label = node.node_label

        action = self._judge_action_gptv(label, node_observations)

        if action == "no_action":
            node.explored = True
            return []
        elif action == "open_close":
            node.explored = True
            # Loop the handle to add them into the actions
            actions = []
            for child_node in node.children.values():
                if "handle" in child_node.node_label:
                    actions.append((child_node, "open_close"))
        elif action == "pick":
            node.explored = True
            actions = []
            # First pick away, after doing the observation, then put it back
            actions.append((node, "pick_back"))
            actions.append((node, "pick_away"))
        return actions

    def _judge_action_gptv(self, label, node_observations):
        # Make GPTV to choose the proper skill
        gptv = MyGPTV(
            config_path="LLM/config/roboexp_gptv.json",
            base_path=f"{self.base_dir}/decisions/cache",
            REPLAY_FLAG=self.REPLAY_FLAG,
        )
        response = gptv(
            query=f"Object: {label}. The below images are the observation in the scene",
            query_image_paths=node_observations,
        )
        # Get the texts after the final answer
        # Regular expression to extract content after "[Final Answer]"
        match = re.search(r"\[Final Answer\]:\s*(.*)", response)
        # Extracting the content after "[Final Answer]"
        final_answer = match.group(1).lower() if match else None

        if "doors" in final_answer:
            return "open_close"
        elif "pick" in final_answer:
            return "pick"
        elif "no action" in final_answer:
            return "no_action"
        else:
            raise ValueError(f"Unknown final answer: {final_answer}")

    def _preprocess_observation(self, observations, node, is_verify=False):
        # Prepare the RGB with masks on the instance
        # Get the instance voxels
        instance_voxel_indexes = node.instance.voxel_indexes
        # Get the pcd from the observations
        node_observations = []
        for name, obs in observations.items():
            if name == "wrist_2":
                continue
            color = obs["rgb"]
            position = obs["position"]
            mask = obs["mask"]
            c2w = obs["c2w"]
            position_world = position @ c2w[:3, :3].T + c2w[:3, 3]
            index_world = self.robo_memory.pcd_to_index(position_world)
            # Get the mask for the instance
            instance_mask = np.isin(index_world, instance_voxel_indexes) * mask
            bbox = get_bbx_from_mask(instance_mask)
            if bbox is None:
                continue

            if not is_verify:
                image_path = f"{self.base_dir}/decisions/{self.save_index}_{node.node_id}_{name}.png"
            else:
                image_path = f"{self.base_dir}/decisions/{self.save_index}_{node.node_id}_{name}_verify.png"

            # Store the image for further decision
            save_image(color, boxes=[bbox], save_path=image_path)
            node_observations.append(image_path)

            if False:
                display = []
                pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(self.robo_memory.index_to_pcd(instance_voxel_indexes))
                pcd.points = o3d.utility.Vector3dVector(position_world[mask])
                pcd.paint_uniform_color(np.array([0, 0, 1]))
                display.append(pcd)
                visualize_pc(
                    self.robo_memory.index_to_pcd(
                        np.array(list(self.robo_memory.memory_scene.keys()))
                    ),
                    np.array(list(self.robo_memory.memory_scene.values())),
                    extra=display,
                )
        return node_observations

    def _get_unplanned_node_list(self, action_scene_graph):
        # Get the unplanned node list, the node should not be explored and not be in the plan list
        plan_node_list = [action[0] for action in self.action_list]
        unplanned_node_list = []
        for node in action_scene_graph.object_nodes.values():
            if not node.is_part and not node.explored and node not in plan_node_list:
                unplanned_node_list.append(node)
        return unplanned_node_list
