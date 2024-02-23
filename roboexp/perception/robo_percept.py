from .models import MyGroundingSegment, MyDenseClip
from roboexp.utils import display_image
import torch


class RoboPercept:
    def __init__(self, grounding_dict, lazy_loading=False, device="cuda"):
        # When lazy loading, the model will be loaded when it is used; warning: it will decrease the inference speed
        self.grounding_dict = grounding_dict
        self.lazy_loading = lazy_loading
        self.device = device

        if not self.lazy_loading:
            # Used to get the grounding masks
            self.my_grounding_sam = MyGroundingSegment(device=self.device)
            # Used to get the pixel-wise clip features
            self.my_dense_clip = MyDenseClip(device=self.device)

    def get_attributes_from_observations(self, observations, visualize=False):
        observation_attributes = {}
        for name, obs in observations.items():
            # obs includes "rgb", "position", "mask", "c2w"
            rgb = obs["rgb"]
            img = (rgb * 255).clip(0, 255).astype("uint8")

            pred_boxes, pred_phrases, pred_masks = self.get_grounding_masks(
                img, only_max=True
            )

            # Visualize the segmentation
            if visualize:
                display_image(img, boxes=pred_boxes, labels=pred_phrases, masks=pred_masks)

            mask_feats = self.get_dense_clip(
                img, pred_boxes, pred_masks, pred_phrases, per_mask=True
            )

            observation_attributes[name] = {
                "pred_boxes": pred_boxes,
                "pred_phrases": pred_phrases,
                "pred_masks": pred_masks,
                "mask_feats": mask_feats,
            }
        return observation_attributes

    def get_grounding_masks(self, img, only_max=True):
        # only_max: only return the label with the highest confidence score for each mask
        if not self.lazy_loading:
            my_grounding_sam = self.my_grounding_sam
        else:
            my_grounding_sam = MyGroundingSegment(device=self.device)
        # The example of the text can be "chair. table ."
        pred_boxes, pred_phrases, pred_masks = my_grounding_sam.run(
            img, self.grounding_dict, only_max=only_max
        )
        # Clean the GPU memory
        if self.lazy_loading:
            del my_grounding_sam
            torch.cuda.empty_cache()
        return pred_boxes, pred_phrases, pred_masks

    def get_dense_clip(self, img, boxes, masks, phrases, per_mask=True):
        if not self.lazy_loading:
            my_dense_clip = self.my_dense_clip
        else:
            my_dense_clip = MyDenseClip(device=self.device)
        # The phrases include the confidence score, the example format is "table:0.7834 chair:0.2333" for each mask
        results = my_dense_clip.run(img, boxes, masks, phrases, per_mask=per_mask)
        # Clean the GPU memory
        if self.lazy_loading:
            del my_dense_clip
            torch.cuda.empty_cache()
        if not per_mask:
            dense_feats, dense_confidences = results 
            return dense_feats.numpy(), dense_confidences.numpy()
        else:
            mask_feats = results
            return mask_feats.numpy()
    
    def get_dense_clip_text(self, texts):
        if not self.lazy_loading:
            my_dense_clip = self.my_dense_clip
        else:
            my_dense_clip = MyDenseClip(device=self.device)
        # The phrases include the confidence score, the example format is "table:0.7834 chair:0.2333" for each mask
        text_feats = my_dense_clip.run_text(texts)
        # Clean the GPU memory
        if self.lazy_loading:
            del my_dense_clipobservations
            torch.cuda.empty_cache()
        return text_feats.numpy()
