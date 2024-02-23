import open_clip
from PIL import Image
import torch


# Refer to the example code in ConceptFusion https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py
class MyCLIP:
    def __init__(
        self,
        model_type,
        pretriained_dataset,
        device="cuda",
    ):
        self.model_type = model_type
        self.device = device
        # Construct the models
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_type, pretriained_dataset
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_type)
        self.model.to(self.device)
        self.model.eval()

    def get_image_feature(self, img):
        pil_img = Image.fromarray(img)
        with torch.no_grad() and torch.cuda.amp.autocast():
            _img = self.preprocess(pil_img).unsqueeze(0)
            img_feature = self.model.encode_image(_img.to(self.device)).detach()
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
        return img_feature

    def get_text_feature(self, texts):
        # texts is a list of strings. Refer to https://github.com/mlfoundations/open_clip
        texts = self.tokenizer(texts)
        with torch.no_grad() and torch.cuda.amp.autocast():
            text_features = self.model.encode_text(texts.cuda()).detach()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
