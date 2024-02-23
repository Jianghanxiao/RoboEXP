import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import torch
from PIL import Image


# The code is modified based on https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py
class MyGroundingDINO:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        device="cuda",
    ):
        self.device = device
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        # Load the model
        args = SLConfig.fromfile(config_path)
        args.device = device
        self.model = build_model(args)
        # Load the pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval()
        self.model.to(self.device)

    def get_grounding_output(self, img, text, with_logits=True, only_max=True):
        # The example of the text can be "chair. table ."
        _img = self._process_image(img)
        _text = self._process_text(text)
        _img = _img.to(self.device)
        with torch.no_grad():
            outputs = self.model(_img[None], captions=[_text])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(_text)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            # Support multiple confidences for different labels (modify the code based on the original function get_phrases_from_posmap())
            if only_max:
                # Pick the label that the sublabel can have the highest probability
                label_index = torch.argmax(logit)
                # Find the starting point of this label and the ending point of this label
                start_label = int(label_index)
                while True:
                    if (
                        tokenized["input_ids"][start_label - 1] == 101
                        or tokenized["input_ids"][start_label - 1] == 102
                        or tokenized["input_ids"][start_label - 1] == 1012
                    ):
                        break
                    start_label -= 1
                end_label = int(label_index)
                while True:
                    if (
                        tokenized["input_ids"][end_label + 1] == 102
                        or tokenized["input_ids"][end_label + 1] == 101
                        or tokenized["input_ids"][end_label + 1] == 1012
                    ):
                        break
                    end_label += 1
                pred_label = tokenlizer.decode([tokenized["input_ids"][i] for i in range(start_label, end_label + 1)])
                pred_logit = logit[label_index].item()
                pred_phrases.append(f"{pred_label}:{str(pred_logit)[:4]}")
            else:
                posmap = logit > self.text_threshold
                non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
                token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
                pred_labels = tokenlizer.decode(token_ids).split()
                pred_logits = [logit[i].item() for i in non_zero_idx]
                if with_logits:
                    output = [
                        f"{pred_labels[i]}:{str(pred_logits[i])[:4]}"
                        for i in range(len(pred_labels))
                    ]
                    pred_phrases.append(" ".join(output))
                else:
                    pred_phrases.append(" ".join(pred_labels))

        return boxes_filt, pred_phrases

    def _process_image(self, img):
        # Apply the normalization to the image
        img_pil = Image.fromarray(img)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        _img, _ = transform(img_pil, None)
        return _img

    def _process_text(self, text):
        # Do processing to the text
        text = text.lower()
        text = text.strip()
        if not text.endswith("."):
            text = text + "."
        return text
