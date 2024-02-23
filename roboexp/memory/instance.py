import numpy as np

instance_label_counts = {}


class myInstance:
    def __init__(self, label, confidence, voxel_indexes, feature, index_to_pcd):
        self.label = label
        # Assign a unique instance id
        global instance_label_counts
        if label not in instance_label_counts:
            instance_label_counts[label] = 0
        else:
            instance_label_counts[label] += 1
        self.instance_id = f"{self.label}_{instance_label_counts[label]}_instance"
        self.confidence = confidence
        self.voxel_indexes = voxel_indexes
        self.feature = feature
        self.deleted = False
        self.no_merge = False
        # Get the function to convert the pcd
        self.index_to_pcd = index_to_pcd

    def get_attributes(self):
        pc = self.index_to_pcd(self.voxel_indexes)
        center = np.mean(pc, axis=0)
        size = np.max(pc, axis=0) - np.min(pc, axis=0)
        return {
            "label": self.label,
            "confidence": self.confidence,
            "center": center,
            "size": size,
        }

    def get_iou(self, instance):
        intersection = len(
            set(self.voxel_indexes).intersection(set(instance.voxel_indexes))
        )
        union = len(set(self.voxel_indexes).union(set(instance.voxel_indexes)))
        return intersection / union

    def get_similarity(self, instance):
        similarity = (
            np.dot(self.feature, instance.feature)
            / np.linalg.norm(self.feature)
            / np.linalg.norm(instance.feature)
        )
        return similarity

    def merge_instance(self, instance):
        self.voxel_indexes = list(
            set(self.voxel_indexes).union(set(instance.voxel_indexes))
        )
        weight = self.confidence / (self.confidence + instance.confidence)
        self.feature = weight * self.feature + (1 - weight) * instance.feature
        self.feature /= np.linalg.norm(self.feature)
        if self.confidence < instance.confidence:
            self.label = instance.label
            self.confidence = instance.confidence

    def get_dict(self):
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "confidence": self.confidence,
            "voxel_indexes": self.voxel_indexes,
            "feature": self.feature,
        }
