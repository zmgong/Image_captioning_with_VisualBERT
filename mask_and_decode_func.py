import torch

def generate_square_subsequent_mask(sz):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    batch_size = src.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    # src_padding_mask = (src == 0).transpose(0, 1)
    src_padding_mask = torch.zeros(batch_size, src_seq_len).type(torch.bool).cuda()
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        # _, next_word = torch.topk(prob, 2, dim=1)
        # next_word = next_word[0, 1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        # print(next_word)
        ys = torch.cat([ys,
                        torch.ones(1, 1).fill_(next_word).type(torch.long).cuda()], dim=0)
        # print(ys)
        if next_word == 2:
            break
    return ys