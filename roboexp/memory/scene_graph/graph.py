from .node import ObjectNode, ActionNode
import graphviz

SG_index = 0


class ActionSceneGraph:
    # The action-conditioned scene graph
    def __init__(self, node_id, instance_label, instance, base_dir):
        self.root = ObjectNode(None, node_id, instance_label, instance)
        self.object_nodes = {node_id: self.root}
        self.base_dir = base_dir

    def add_object(
        self, parent, node_id, instance_label, instance, parent_relation, is_part=False
    ):
        # Add an object node to the graph
        node = ObjectNode(
            parent,
            node_id,
            instance_label,
            instance,
            parent_relation=parent_relation,
            is_part=is_part,
        )
        parent.add_child(node)
        self.object_nodes[node_id] = node
        return node

    def add_action(self, parent, node_id, node_label, preconditions=[]):
        # Add an action node to the graph
        node = ActionNode(parent, node_id, node_label, preconditions)
        parent.add_action(node)
        return node

    def get_root(self):
        return self.root

    def visualize(self):
        # Visualize the action-conditioned scene graph
        global SG_index
        dag = graphviz.Digraph(
            directory=f"{self.base_dir}/scene_graphs", filename=f"sg_{SG_index}"
        )
        queue = [self.root]

        dag.node(
            self.root.node_id,
            label=self.root.node_label,
            shape="egg",
            color="lightblue2",
            style="filled",
        )
        while len(queue) > 0:
            node = queue.pop(0)
            for child in list(node.children.values()) + list(node.actions.values()):
                if child.is_object():
                    if child.is_part:
                        color = "lightpink"
                    else:
                        color = "lightblue2"
                else:
                    color = "lightsalmon"
                dag.node(
                    child.node_id,
                    label=child.node_label,
                    shape="egg" if child.is_object() else "diamond",
                    color=color,
                    style="filled",
                )
                if child.is_object():
                    dag.edge(node.node_id, child.node_id, label=child.parent_relation)
                else:
                    dag.edge(node.node_id, child.node_id)
                    for precondition in child.preconditions:
                        dag.edge(
                            precondition.node_id,
                            child.node_id,
                        )
                queue.append(child)

        dag.render()
        print(
            [
                (node.node_id, node.instance.instance_id, node.explored)
                for node in self.object_nodes.values()
            ]
        )
        SG_index += 1

    def to_dict(self):
        nodes_dict = {}
        # Convert the scene graph to a dictionary
        for node_id, node in self.object_nodes.items():
            node_dict = {
                "label": node.node_label,
                "type": "object",
            }
            if node.parent is not None:
                node_dict["parent"] = node.parent.node_id
            if "handle" in node.node_label:
                node_dict["is_handle"] = True
                node_dict["handle_center"] = node.handle_center
                node_dict["handle_direction"] = node.handle_direction
                node_dict["open_direction"] = node.open_direction
                node_dict["joint_type"] = node.joint_type
                if node.joint_type == "revolute":
                    node_dict["joint_axis"] = node.joint_axis
                    node_dict["joint_origin"] = node.joint_origin
                    node_dict["side_direction"] = node.side_direction
            nodes_dict[node_id] = node_dict
            # Add all the action node also into this
            for action_node_id, action_node in node.actions.items():
                action_node_dict = {
                    "label": action_node.node_label,
                    "preconditions": [
                        precondition.node_id
                        for precondition in action_node.preconditions
                    ],
                    "parent": node.node_id,
                    "type": "action",
                }
                nodes_dict[action_node_id] = action_node_dict
        return nodes_dict
