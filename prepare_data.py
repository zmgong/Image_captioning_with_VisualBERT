import os

import h5py
import nltk
import numpy as np
import tensorflow as tf
import torch
import random
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from torch.utils.data import Dataset, DataLoader

class CocoCaptionDataset(Dataset):
    def __init__(self, path_caption_dict, vocab, h5FileName):
        self.data = list(path_caption_dict.items())
        self.vocab = vocab
        self.h5 = h5py.File(h5FileName + ".h5", 'r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        (path, caption) = self.data[idx]

        embed = torch.tensor(np.array(self.h5[path]))
        # set_list = list(h5f[set_dir])
        # embed = self.visual_embed(path)

         # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption[0]).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        return embed, target, path

def sample_from_dict(d, train_size, val_size):
    sample = train_size + val_size
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    tkeys, tvals = keys[:train_size], values[:train_size]
    vkeys, vvals = keys[train_size:], values[train_size:]
    return dict(zip(tkeys, tvals)), dict(zip(vkeys, vvals))

def download_images():
    annotation_folder = '/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)

    image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder
    return PATH

def prepare_image_inputs(cfg, model, img_list):
    with torch.no_grad():
        # Resizing the image according to the configuration
        transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

        # Convert to C,H,W format
        convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

        batched_inputs = [{"image": convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in
                          img_list]

        # Normalizing the image
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        images = [normalizer(x["image"]) for x in batched_inputs]

        # Convert to ImageList
        images = ImageList.from_tensors(images, model.backbone.size_divisibility)

        return images, batched_inputs
