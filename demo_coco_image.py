# Test few images from coco dataset
# Note: There is a big chance that you will use the training image to test.
import os

import matplotlib.pyplot as plt
import nltk
import torch
import json
import collections
from torch.utils.data import DataLoader
import models
from data_loader import collate_fn
from extract_visual_embed import load_image
from mask_and_decode_func import greedy_decode
from prepare_data import download_images, sample_from_dict, CocoCaptionDataset
from save_visual_embds import save_visual_embeds


def main():
    if os.path.exists("visual_embeds_test.h5"):
        os.remove("visual_embeds_test.h5")

    train_PATH, val_PATH = download_images()
    cfg, model = models.get_mask_rcnn_model()
    with open('annotations/captions_val2014.json') as f:
        a = json.load(f)
    listOfAnn = a['annotations'][0:1000]
    image_path_to_caption = {}
    for val in listOfAnn:
        caption = val['caption']
        image_path = val_PATH + 'COCO_val2014_' + '%012d.jpg' % (val['image_id'])
        # print(image_path)
        image_path_to_caption[image_path] = caption
    fileName = "visual_embeds_test"
    save_visual_embeds(cfg, model, image_path_to_caption, fileName)

    path_caption_dict = collections.defaultdict(list)
    for val in listOfAnn:
        caption = val['caption']
        image_path = val_PATH + 'COCO_val2014_' + '%012d.jpg' % (val['image_id'])
        path_caption_dict[image_path].append(caption)

    train_path_caption_dict, val_path_caption_dict = sample_from_dict(path_caption_dict, 10, 20)

    nltk.download('punkt')
    vocab = models.build_vocab(json='annotations/captions_train2014.json', threshold=5)
    dataset_val = CocoCaptionDataset(val_path_caption_dict, vocab, fileName)
    bertload = models.VisualBertTransformer(num_decoder_layers=6,
                                            emb_size=768, nhead=8,
                                            tgt_vocab_size=(len(vocab)) + 3).cuda()
    bertload.load_state_dict(torch.load('model/bert_full_13.ckpt'))
    bertload.eval()
    val_loader = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=True, collate_fn=collate_fn)

    for i, (vembed, captions, path) in enumerate(val_loader):
        with torch.no_grad():
            vembed = vembed.cuda()
            num_tokens = vembed.shape[0]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
            tgt_tokens = greedy_decode(
                bertload, vembed, src_mask, max_len=num_tokens + 5, start_symbol=1).flatten()
            img, img_bgr, _, = load_image(path[0])

            predtokens = [vocab.idx2word[idx] for idx in tgt_tokens.cpu().numpy() if idx > 4]
            gts = path_caption_dict[path[0]]
            truetokens = [' '.join(nltk.tokenize.word_tokenize(str(s).lower())) for s in gts]

            preds = 'pred: ' + " ".join(predtokens)

            plt.imshow(img)

            plt.text(0, -1, preds, bbox=dict(fill=False, edgecolor='red', linewidth=2))
            plt.axis('off')
            plt.show()
            print('pred: ' + " ".join(predtokens))
            print('true: ' + "\n".join(truetokens))
            if i == 5:
                break


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    main()
