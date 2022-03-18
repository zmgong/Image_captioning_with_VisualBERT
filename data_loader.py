import torch


def collate_fn(batch):
    src_batch, tgt_batch, path_batch = [], [], []
    for src_sample, tgt_sample, path in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        path_batch.append(path)

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0).type(torch.long)
    return src_batch, tgt_batch, path_batch