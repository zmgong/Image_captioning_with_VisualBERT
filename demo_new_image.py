import os
import matplotlib.pyplot as plt
import nltk
import torch

import models
from extract_visual_embed import load_image, get_features, featureToVem
from mask_and_decode_func import greedy_decode
from prepare_data import prepare_image_inputs, download_images


def main():
    image_folder_path = download_images()
    nltk.download('punkt')
    vocab = models.build_vocab(json='annotations/captions_train2014.json', threshold=5)
    bertload = models.VisualBertTransformer(num_decoder_layers=6,
                                            emb_size=768, nhead=8,
                                            tgt_vocab_size=(len(vocab) + 3)).cuda()
    bertload.load_state_dict(torch.load('model/bert_full_13.ckpt'))
    bertload.eval()

    cfg, model = models.get_mask_rcnn_model()
    for file in os.listdir('{}/image_for_test'.format(os.getcwd())):
        filename = os.fsdecode(file)
        imagePath = os.path.join('{}/image_for_test'.format(os.getcwd()), filename)
        imageList = []
        img, img_bgr, _, = load_image(imagePath)

        imageList.append(img_bgr)
        images, batched_inputs = prepare_image_inputs(cfg, model, imageList)
        features = get_features(model, images)

        vembed = featureToVem(images, batched_inputs, features)
        vembed = torch.reshape(vembed, (100, 1, 1024))
        num_tokens = vembed.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        tgt_tokens = greedy_decode(bertload, vembed, src_mask, max_len=num_tokens + 5, start_symbol=1).flatten()
        predtokens = [vocab.idx2word[idx] for idx in tgt_tokens.cpu().numpy() if idx > 4]

        preds = 'pred: ' + " ".join(predtokens)

        plt.imshow(img)

        plt.text(0, -1, preds, bbox=dict(fill=False, edgecolor='red', linewidth=2))
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    main()
