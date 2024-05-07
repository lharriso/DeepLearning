import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg

### Get visual embedding (Detectron2)
#Reference: https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing#scrollTo=7-5rqN-vtlkq

class VisualEmbedder():
    def __init__(self, cfg_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", min_boxes=10, max_boxes=100):
        self.cfg_path = cfg_path
        #In order to get the box features for the best few proposals and limit the sequence length, we set minimum and maximum boxes and pick those box features.
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Load Config and Model Weights
    # I am using the MaskRCNN ResNet-101 FPN checkpoint, but you can use any checkpoint of your preference. This checkpoint is pre-trained on the COCO dataset. 
    # You can check other checkpoints/configs on the Model Zoo (https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) page.

    def load_config_and_model_weights(self,cfg_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

        # ROI HEADS SCORE THRESHOLD
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # Comment the next line if you're using 'cuda'
        # cfg['MODEL']['DEVICE']='cpu'

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        return cfg

    ## Load the Object Detection Model
    # The `build_model` method can be used to load a model from the configuration, the checkpoints have to be loaded using the `DetetionCheckpointer`.
    def get_model(self,cfg):
        # build model
        model = build_model(cfg)
        model.to(self.device)

        # load weights
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # eval mode
        model.eval()
        return model

    ## Convert Image to Model Input
    # The detectron uses resizing and normalization based on the configuration parameters and the input is to be provided using `ImageList`. 
    # The `model.backbone.size_divisibility` handles the sizes (padding) such that the FPN lateral and output convolutional features have same dimensions.
    def prepare_image_inputs(self,cfg, img_list, model):
        # Resizing the image according to the configuration
        transform_gen = T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
        img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

        # Convert to C,H,W format
        convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

        batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

        # Normalizing the image
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        images = [normalizer(x["image"]) for x in batched_inputs]

        # Convert to ImageList
        images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
        
        return images, batched_inputs

    ## Get ResNet+FPN features
    # The ResNet model in combination with FPN generates five features for an image at different levels of complexity. 
    # For more details, refer to the FPN paper or this [article](https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd). 
    # For this tutorial, just know that `p2`, `p3`, `p4`, `p5`, `p6` are the features needed by the RPN (Region Proposal Network). 
    # The proposals in combination with `p2`, `p3`, `p4`, `p5` are then used by the ROI (Region of Interest) heads to generate box predictions.
    def get_features(self,model, images):
        features = model.backbone(images.tensor.to(self.device))
        return features

    ## Get region proposals from RPN
    # This RPN takes in the features and images and generates the proposals. Based on the configuration we chose, we get 1000 proposals.
    def get_proposals(self,model, images, features):
        proposals, _ = model.proposal_generator(images, features)
        return proposals

    ## Get Box Features for the proposals
    # The proposals and features are then used by the ROI heads to get the predictions. In this case, the partial execution of layers becomes significant. 
    # We want the `box_features` to be the `fc2` outputs of the regions. Hence, I use only the layers that are needed until that step. 
    def get_box_features(self,model, features, proposals):
        features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
        box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head.flatten(box_features)
        box_features = model.roi_heads.box_head.fc1(box_features)
        box_features = model.roi_heads.box_head.fc_relu1(box_features)
        box_features = model.roi_heads.box_head.fc2(box_features)

        box_features = box_features.reshape(self.num_image, 1000, 1024) # depends on your config and batch size
        return box_features, features_list

    ## Get prediction logits and boxes
    # The prediction class logits and the box predictions from the ROI heads, this is used in the next step to get the boxes and scores from the `FastRCNNOutputs`
    def get_prediction_logits(self,model, features_list, proposals):
        cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        cls_features = model.roi_heads.box_head(cls_features)
        pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
        return pred_class_logits, pred_proposal_deltas

    ## Get FastRCNN scores and boxes
    # This results in the softmax scores and the boxes.
    def get_box_scores(self,cfg, pred_class_logits, pred_proposal_deltas,proposals):
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
        )

        boxes = outputs.predict_boxes()
        scores = outputs.predict_probs()
        image_shapes = outputs.image_shapes

        return boxes, scores, image_shapes

    ## Rescale the boxes to original image size
    # We want to rescale the boxes to original size as this is done in the detectron2 library. 
    # This is done for sanity and to keep it similar to the visualbert repository.
    def get_output_boxes(self,boxes, batched_inputs, image_size):
        proposal_boxes = boxes.reshape(-1, 4).clone()
        scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
        output_boxes = Boxes(proposal_boxes)

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(image_size)

        return output_boxes

    ## Select the Boxes using NMS
    # We need two thresholds - NMS threshold for the NMS box section, and score threshold for the score based section.
    # First NMS is performed for all the classes and the max scores of each proposal box and each class is updated.
    # Then the class score threshold is used to select the boxes from those.
    def select_boxes(self,cfg, output_boxes, scores):
        test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        cls_prob = scores.detach()
        cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
        max_conf = torch.zeros((cls_boxes.shape[0]))
        for cls_ind in range(0, cls_prob.shape[1]-1):
            cls_scores = cls_prob[:, cls_ind+1].cpu()
            det_boxes = cls_boxes[:,cls_ind,:].cpu()
            keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
        return keep_boxes, max_conf

    ## Limit the total number of boxes
    # In order to get the box features for the best few proposals and limit the sequence length, we set minimum and maximum boxes and pick those box features.
    def filter_boxes(self,keep_boxes, max_conf, min_boxes, max_boxes):
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
        return keep_boxes

    ## Get the visual embeddings :) 
    # Finally, the boxes are chosen using the `keep_boxes` indices and from the `box_features` tensor.
    def get_visual_embeds(self,box_features, keep_boxes):
        return box_features[keep_boxes.copy()]
    
    def visual_embeds_detectron2(self, img_list):
        self.num_image=len(img_list)
        cfg = self.load_config_and_model_weights(self.cfg_path)
        model = self.get_model(cfg)
        images, batched_inputs = self.prepare_image_inputs(cfg, img_list, model)
        features = self.get_features(model, images)
        # print(features.keys())
        proposals = self.get_proposals(model, images, features)
        box_features, features_list = self.get_box_features(model, features, proposals)
        pred_class_logits, pred_proposal_deltas = self.get_prediction_logits(model, features_list, proposals)
        boxes, scores, image_shapes = self.get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)
        output_boxes = [self.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
        temp = [self.select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
        keep_boxes, max_conf = [],[]
        for keep_box, mx_conf in temp:
            keep_boxes.append(keep_box)
            max_conf.append(mx_conf)
        keep_boxes = [self.filter_boxes(keep_box, mx_conf, self.min_boxes, self.max_boxes) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
        visual_embeds = [self.get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
        torch.cuda.empty_cache()
        return visual_embeds