import h5py
from tqdm import tqdm
import os.path

from extract_visual_embed import load_image, get_features, get_proposals, get_box_features, get_prediction_logits, \
    get_box_scores, get_output_boxes, select_boxes, filter_boxes, get_visual_embeds
from prepare_data import prepare_image_inputs


def save_visual_embeds(cfg, model, data, fileName):
    if os.path.exists(fileName+".h5"):
        return
    with h5py.File(fileName+".h5", 'w') as vh5:
        for path, caption in tqdm(data.items()):
            imageList = []
            _, img_bgr, _, = load_image(path)
            imageList.append(img_bgr)
            images, batched_inputs = prepare_image_inputs(cfg, model, imageList)
            features = get_features(model, images)
            proposals = get_proposals(model, images, features)
            box_features, features_list = get_box_features(model, features, proposals, 1)
            pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)
            boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)
            output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
            temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
            keep_boxes, max_conf = [],[]
            for keep_box, mx_conf in temp:
                keep_boxes.append(keep_box)
                max_conf.append(mx_conf)
            MIN_BOXES=10
            MAX_BOXES=100
            keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
            visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
            vh5[path] = visual_embeds[0].cpu().detach().numpy()

