from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from transformers import BertTokenizer, VisualBertModel
import torch.nn as nn
from torch import Tensor
import math
import torch
from pycocotools.coco import COCO
from collections import Counter
import nltk
from torch.nn import Transformer


# get mask_rcnn model
def get_mask_rcnn_model():
    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    # cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return cfg, model


# transformer model
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# bert model
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class VisualBertTransformer(nn.Module):
    def __init__(self,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dropout: float = 0.1):
        super(VisualBertTransformer, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre').cuda()
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                tgt_mask: Tensor,
                tgt_padding_mask: Tensor):
        # visualbert batch first
        src = src.transpose(1, 0)

        questions = [''] * src.shape[0]
        tokens = self.tokenizer(questions, padding='max_length', max_length=0)
        self.input_ids = torch.tensor(tokens["input_ids"]).cuda()
        self.attention_mask = torch.tensor(tokens["attention_mask"]).cuda()
        self.token_type_ids = torch.tensor(tokens["token_type_ids"]).cuda()

        visual_attention_mask = torch.ones(src.shape[:-1], dtype=torch.long).cuda()
        visual_token_type_ids = torch.ones(src.shape[:-1], dtype=torch.long).cuda()
        outputs = self.bert(input_ids=self.input_ids, attention_mask=self.attention_mask,
                            token_type_ids=self.token_type_ids, visual_embeds=src,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids)
        memory = outputs.last_hidden_state.transpose(1, 0)

        tgt = self.tgt_tok_emb(trg)
        tgt_emb = self.positional_encoding(tgt)

        outs = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_padding_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        src = src.transpose(1, 0)

        questions = [''] * src.shape[0]
        tokens = self.tokenizer(questions, padding='max_length', max_length=0)
        self.input_ids = torch.tensor(tokens["input_ids"]).cuda()
        self.attention_mask = torch.tensor(tokens["attention_mask"]).cuda()
        self.token_type_ids = torch.tensor(tokens["token_type_ids"]).cuda()

        visual_attention_mask = torch.ones(src.shape[:-1], dtype=torch.long).cuda()
        visual_token_type_ids = torch.ones(src.shape[:-1], dtype=torch.long).cuda()
        outputs = self.bert(input_ids=self.input_ids, attention_mask=self.attention_mask,
                            token_type_ids=self.token_type_ids, visual_embeds=src,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids)
        return outputs.last_hidden_state.transpose(1, 0)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt = self.tgt_tok_emb(tgt)
        tgt_emb = self.positional_encoding(tgt)

        return self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." % (i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


