import os

import numpy as np
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
import cv2
from matplotlib import pyplot as plt

from models import get_mask_rcnn_model
from prepare_data import prepare_image_inputs

cfg, model = get_mask_rcnn_model()


def load_image(image_path):
    # print(image_path)
    img_bgr = cv2.imread(image_path)
    # img = cv2.resize(img, (299, 299))
    # plt.imshow(img_bgr)
    # plt.axis('off')
    # plt.show()
    # Detectron expects BGR images
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img, img_bgr, image_path


def get_features(model, images):
    with torch.no_grad():
        features = model.backbone(images.tensor.cuda())
        torch.cuda.empty_cache()
        return features


def get_proposals(model, images, features):
    with torch.no_grad():
        proposals, _ = model.proposal_generator(images, features)
        return proposals


def get_box_features(model, features, proposals, batch_size):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(batch_size, -1, 1024)  # depends on your config and batch size
    # pad = torch.nn.utils.rnn.pad_sequence([box_features, torch.zeros(1000, 1024)], batch_first=True)
    # box_features = pad[:-1]
    return box_features, features_list


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    pred_class_logits = pred_class_logits.reshape(-1, 81)
    pred_proposal_deltas = pred_proposal_deltas.reshape(-1, 320)
    return pred_class_logits, pred_proposal_deltas


def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals):
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


def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes


def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(-1, 80, 4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1] - 1):
        cls_scores = cls_prob[:, cls_ind + 1].cpu()
        det_boxes = cls_boxes[:, cls_ind, :].cpu()
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf


def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes


def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]


def featureToVem(images, batched_inputs, features):
    proposals = get_proposals(model, images, features)
    box_features, features_list = get_box_features(model, features, proposals, 1)
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)
    boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)
    output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in
                    range(len(proposals))]
    temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
    keep_boxes, max_conf = [], []
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)
    MIN_BOXES = 10
    MAX_BOXES = 100
    keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in
                  zip(keep_boxes, max_conf)]
    visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in
                     zip(box_features, keep_boxes)]
    visual_embeds = visual_embeds[0].cpu().detach().numpy()
    # print(visual_embeds)
    # print(np.shape(visual_embeds))
    embed = torch.tensor(np.array(visual_embeds))
    return embed.cuda()
