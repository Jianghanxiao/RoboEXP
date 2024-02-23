from .sam import MySAM
from .grounding_dino import MyGroundingDINO


class MyGroundingSegment:
    def __init__(
        self,
        # Parameters for SAM
        use_sam_hq=True,
        sam_checkpoint_path="pretrained_models/sam_vit_h_4b8939.pth",
        sam_hq_checkpoint_path="pretrained_models/sam_hq_vit_h.pth",
        # Parameters for GroundingDINO
        config_path="roboexp/perception/models/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path="pretrained_models/groundingdino_swint_ogc.pth",
        # Other parameters
        device="cuda",
    ):
        self.my_sam = MySAM(
            use_sam_hq=use_sam_hq,
            sam_checkpoint_path=sam_checkpoint_path,
            sam_hq_checkpoint_path=sam_hq_checkpoint_path,
            device=device,
        )
        self.my_grounding_dino = MyGroundingDINO(
            config_path=config_path, checkpoint_path=checkpoint_path, device=device
        )

    def run(self, img, text, only_max=True):
        boxes_filt, pred_phrases = self.my_grounding_dino.get_grounding_output(
            img, text, only_max=only_max
        )

        if boxes_filt.size(0) == 0:
            return None, None, None
        else:
            masks = self.my_sam.run(img, boxes=boxes_filt).squeeze(1)
            return boxes_filt.numpy(), pred_phrases, masks.cpu().numpy()
