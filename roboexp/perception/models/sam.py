from segment_anything import (
    build_sam,
    build_sam_hq,
    SamAutomaticMaskGenerator,
    SamPredictor,
)
import torch


# This support both SAM and SAM_HQ
# The code is modified based on https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py
class MySAM:
    def __init__(
        self,
        use_sam_hq,
        sam_checkpoint_path=None,
        sam_hq_checkpoint_path=None,
        device="cuda",
    ):
        self.device = device
        # Build the model
        if use_sam_hq:
            model = build_sam_hq(checkpoint=sam_hq_checkpoint_path)
        else:
            model = build_sam(checkpoint=sam_checkpoint_path)
        model.to(device)
        # Following the hyperparameters setting from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
        # The original ConceptFusion setting focus more on the object level, lack of fine-grained details https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py
        self.mask_generator = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        self.predictor = SamPredictor(model)

    def run(self, img, automatic_mask_flag=False, boxes=None):
        if automatic_mask_flag:
            return self._run_automatic_mask(img)
        else:
            return self._run_predictor(img, boxes)

    def _run_automatic_mask(self, img):
        # img is in numpy array with shape (H, W, 3)
        masks = self.mask_generator.generate(img)
        return masks

    def _run_predictor(self, img, boxes):
        assert boxes is not None
        self.predictor.set_image(img)
        H, W = img.shape[:2]
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] += boxes[i][:2]
        boxes = boxes.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes, img.shape[:2]
        ).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks
