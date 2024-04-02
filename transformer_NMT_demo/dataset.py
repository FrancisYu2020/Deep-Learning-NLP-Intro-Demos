import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def yield_tokens(data_iter, tokenizer):
    for _, src, tgt in data_iter:
        yield tokenizer(src)
        yield tokenizer(tgt)

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Load the Multi30k dataset
train_iter, valid_iter, test_iter = Multi30k(split=('train', 'valid', 'test'))

# Build vocabularies
src_tokenizer = get_tokenizer(tokenize_de)
tgt_tokenizer = get_tokenizer(tokenize_en)

src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, src_tokenizer), specials=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])
tgt_vocab = build_vocab_from_iterator(yield_tokens(train_iter, tgt_tokenizer), specials=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])

src_vocab.set_default_index(src_vocab["<UNK>"])
tgt_vocab.set_default_index(tgt_vocab["<UNK>"])

def data_process(raw_data_iter):
    data = []
    for (_, src, tgt) in raw_data_iter:
        src_tensor = torch.tensor([src_vocab[token] for token in src_tokenizer(src)], dtype=torch.long)
        tgt_tensor = torch.tensor([tgt_vocab[token] for token in tgt_tokenizer(tgt)], dtype=torch.long)
        data.append((src_tensor, tgt_tensor))
    return data

train_data = data_process(train_iter)
valid_data = data_process(valid_iter)
test_data = data_process(test_iter)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(torch.cat([torch.tensor([src_vocab["<SOS>"]]), src_item, torch.tensor([src_vocab["<EOS>"]])], dim=0))
        tgt_batch.append(torch.cat([torch.tensor([tgt_vocab["<SOS>"]]), tgt_item, torch.tensor([tgt_vocab["<EOS>"]])], dim=0))
    src_batch = pad_sequence(src_batch, padding_value=src_vocab["<PAD>"])
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab["<PAD>"])
    return src_batch, tgt_batch

def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
