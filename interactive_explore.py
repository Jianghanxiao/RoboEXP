from roboexp import (
    RobotExplorationReal,
    RoboMemory,
    RoboPercept,
    RoboActReal,
    RoboDecision,
)
from datetime import datetime
import os


def explore(robo_decision, robo_act):
    # Used to test the exploration with the saved observations
    robo_act.get_observations_update_memory(update_scene_graph=True, visualize=False)
    # Test the robo_decision module
    robo_decision.update_action_list()
    while robo_decision.is_done() == False:
        action = robo_decision.get_action()
        print(action)
        if action[1] == "open_close":
            robo_act.skill_open_close(action[0], visualize=False)
        elif "pick" in action[1]:
            robo_act.skill_pick(action[0], action[1], visualize=False)
            if action[1] == "pick_away":
                update_scene_graph = True
                scene_graph_option = {
                    "type": "pick_away",
                    "node": action[0],
                }
                robo_act.get_observations_update_memory(
                    update_scene_graph=update_scene_graph,
                    scene_graph_option=scene_graph_option,
                    visualize=False,
                )
                robo_decision.update_action_list()
            elif action[1] == "pick_back":
                update_scene_graph = True
                scene_graph_option = {
                    "type": "pick_back",
                    "node": action[0],
                }
                robo_act.get_observations_update_memory(
                    update_scene_graph=update_scene_graph,
                    scene_graph_option=scene_graph_option,
                    visualize=False,
                )


def run(base_dir, REPLAY_FLAG=False):
    robo_exp = RobotExplorationReal(gripper_length=0.285, REPLAY_FLAG=REPLAY_FLAG)
    # Initialize the memory module
    robo_memory = RoboMemory(
        lower_bound=[0, -0.8, -1],
        higher_bound=[1, 0.5, 2],
        voxel_size=0.01,
        real_camera=True,
        base_dir=base_dir,
        similarity_thres=0.95,
        iou_thres=0.01,
    )

    # Set the labels
    object_level_labels = [
        "table",
        "refrigerator",
        "cabinet",
        "can",
        "doll",
        "plate",
        "spoon",
        "fork",
        "hamburger",
        "condiment",
    ]
    part_level_labels = ["handle"]

    grounding_dict = (
        " . ".join(object_level_labels) + " . " + " . ".join(part_level_labels)
    )
    # Initialize the perception module
    robo_percept = RoboPercept(grounding_dict=grounding_dict, lazy_loading=False)
    # Initialize the action module
    robo_act = RoboActReal(
        robo_exp,
        robo_percept,
        robo_memory,
        object_level_labels,
        base_dir=base_dir,
        REPLAY_FLAG=REPLAY_FLAG,
    )
    # Initialize the decision module
    robo_decision = RoboDecision(robo_memory, base_dir, REPLAY_FLAG=REPLAY_FLAG)

    explore(robo_decision, robo_act)


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{current_time}"
    REPLAY_FLAG = False
    if not os.path.exists(base_dir):
        # Create directory if it doesn't exist
        os.makedirs(base_dir)
    run(base_dir, REPLAY_FLAG=REPLAY_FLAG)
