class Node:
    def __init__(self, parent, node_id, node_label):
        self.parent = parent
        self.node_id = node_id
        self.node_label = node_label
        self.children = {}
        self.actions = {}

    def is_object(self):
        return isinstance(self, ObjectNode)

    def is_action(self):
        return isinstance(self, ActionNode)

    def add_child(self, child):
        self.children[child.node_id] = child

    def add_action(self, action):
        self.actions[action.node_id] = action

    def get_parent(self):
        return self.parent


class ObjectNode(Node):
    # The node to store the object information
    def __init__(
        self, parent, node_id, node_label, instance, parent_relation=None, is_part=False
    ):
        # Parent relation is the relation between the parent and the object
        assert node_label == instance.label or instance.label in node_label
        super().__init__(parent, node_id, node_label)
        self.instance = instance
        self.voxel_indexes = instance.voxel_indexes
        self.parent_relation = parent_relation
        self.is_part = is_part
        self.explored = False
        self.verified = False
        self.pre_nodes = []

    def update_instance(self, instance):
        assert self.node_label == instance.label or instance.label in self.node_label
        self.instance = instance
        self.voxel_indexes = instance.voxel_indexes

    def delete(self):
        if self.parent:
            del self.parent.children[self.node_id]

    def __str__(self):
        return self.node_label


class ActionNode(Node):
    # The node to store the action information
    def __init__(self, parent, node_id, node_label, preconditions):
        # The preconditions are a list of ActionNode to specify the preconditions
        super().__init__(parent, node_id, node_label)
        self.preconditions = preconditions

    def __str__(self):
        return self.node_label
