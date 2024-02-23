from .clip import MyCLIP
import torch
import numpy as np


class MyDenseClip:
    def __init__(
        self,
        # Parameters for CLIP
        model_type="ViT-H-14",
        pretriained_dataset="laion2b_s32b_b79k",
        # Other parameters
        device="cuda",
    ):
        # If lazy loading, the text tokenizer will be loaded when it is used; warning: it will decrease the inference speed
        self.my_clip = MyCLIP(
            model_type=model_type,
            pretriained_dataset=pretriained_dataset,
            device=device,
        )
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def run_text(self, texts):
        # Get the text features using the text encoder of CLIP
        text_feats = self.my_clip.get_text_feature(texts).cpu()
        return text_feats

    def run(self, img, boxes, masks, phrases, per_mask=True):
        # The dense clip model is modified on ConceptFusion https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py
        # It further considers the correlation between the part label and the global image feature
        # Parse the phrases to labels with confidence scores and extract the text features
        labels, confidences = self._parse_phrases(phrases)
        text_feats = self.my_clip.get_text_feature(labels)
        # Calculate the global and local features of the masks
        global_img_feat = self.my_clip.get_image_feature(img)
        feat_dim = global_img_feat.shape[-1]
        local_img_feats, local_nonzero_inds = self._extract_local_features(
            img, boxes, masks
        )
        # Calculate the similarities, we don't apply softmax like the ConceptFusion, each mask should be kind of independent
        weighted_similarities = []
        # Make the confidence scores in the range of [0, 0.5], represents the part that the text influences the feature merging process
        weighted_confidences = [confidence / 2 for confidence in confidences]
        for local_img_feat, text_feat, weighted_confidence in zip(
            local_img_feats, text_feats, weighted_confidences
        ):
            # Calculate the similarity between the local image and local text with the global
            _img_local_global = self.cosine_similarity(local_img_feat, global_img_feat)
            _text_local_global = self.cosine_similarity(text_feat, global_img_feat)
            # Average the similarity
            weighted_similarities.append(
                torch.cos(
                    (1 - weighted_confidence) * torch.acos(_img_local_global)
                    + weighted_confidence * torch.acos(_text_local_global)
                )
            )
        # Calculate the dense features, there can be some pixels without any clip feature
        # Because we use the masks from GroundingSAM and may only care about the attributes for the objects we detect
        if not per_mask:
            LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = img.shape[0], img.shape[1]
            dense_feats = torch.zeros(
                LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half
            )
            dense_confidences = torch.zeros(
                LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, dtype=torch.half
            )
            if masks is None:
                # Return the empty stuff
                return dense_feats, dense_confidences
        else:
            if masks is None:
                return torch.zeros(0, feat_dim, dtype=torch.half)
            mask_feats = torch.zeros(len(masks), feat_dim, dtype=torch.half)
        for maskidx in range(len(masks)):
            # Merge the features, use the mean-weeight to apply on the global feature, then merge both local image feature and local text feature (text feature based on confidence)
            _weighted_similarity = weighted_similarities[maskidx]
            _weighted_feat = _weighted_similarity * global_img_feat + (
                1 - _weighted_similarity
            ) * (
                (1 - weighted_confidences[maskidx]) * local_img_feats[maskidx]
                + weighted_confidences[maskidx] * text_feats[maskidx]
            )
            # Normalize the weighted feature
            _weighted_feat = torch.nn.functional.normalize(
                _weighted_feat.float(),
                dim=-1,
            ).half()
            if not per_mask:
                # Merge the feature based on the confidence of current mask
                dense_feats[
                    local_nonzero_inds[maskidx][:, 0], local_nonzero_inds[maskidx][:, 1]
                ] += _weighted_feat[0].cpu()
                # Update the confidence using the maximum confidence
                dense_confidences[
                    local_nonzero_inds[maskidx][:, 0], local_nonzero_inds[maskidx][:, 1]
                ] = torch.maximum(
                    dense_confidences[
                        local_nonzero_inds[maskidx][:, 0],
                        local_nonzero_inds[maskidx][:, 1],
                    ],
                    confidences[maskidx]
                    * torch.ones_like(
                        dense_confidences[
                            local_nonzero_inds[maskidx][:, 0],
                            local_nonzero_inds[maskidx][:, 1],
                        ]
                    ),
                )
            else:
                mask_feats[maskidx] = _weighted_feat[0].cpu()
        if not per_mask:
            # Normalize the feature
            dense_feats = torch.nn.functional.normalize(
                dense_feats.float(),
                dim=-1,
            ).half()
            return dense_feats, dense_confidences
        else:
            return mask_feats

    def _extract_local_features(self, img, boxes, masks):
        if boxes is None:
            return [], []
        # Extract the local features
        local_img_feats = []
        local_nonzero_inds = []
        for box, mask in zip(boxes, masks):
            # box is a numpy array with shape (4,), x0, y0, x1, y1
            # mask is a numpy array with shape (H, W)
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            x0, y0, x1, y1 = tuple(box.astype(np.int32))
            img_roi = img[y0:y1, x0:x1, :]
            roi_feat = self.my_clip.get_image_feature(img_roi)
            local_img_feats.append(roi_feat)
            nonzero_inds = torch.argwhere(torch.from_numpy(mask))
            local_nonzero_inds.append(nonzero_inds)

        return local_img_feats, local_nonzero_inds

    def _parse_phrases(self, phrases):
        if phrases is None:
            return [], []
        labels = []
        confidences = []
        for phrase in phrases:
            # One phrase can potentially contain multiple labels, pick the label with the max probability
            label_confs = phrase.split(" ")
            max_confidence = 0
            for label_conf in label_confs:
                label, confidence = tuple(label_conf.split(":"))
                confidence = float(confidence)
                if confidence > max_confidence:
                    potential_label = label
                    max_confidence = confidence
            labels.append(potential_label)
            confidences.append(max_confidence)
        return labels, confidences
