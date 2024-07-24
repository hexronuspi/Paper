import pandas as pd

df = pd.read_csv('/kaggle/input/hindi-english-parallel-corpus/hindi_english_parallel.csv')

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from transformers import AutoTokenizer
from torchtext.utils import download_from_url, extract_archive
import io

import re

def clear_text(sentences):
    remove_special_chars = re.compile(r'[^A-Za-z\sअ-हक-ह]')
    remove_extra_spaces = re.compile(r'\s+')

    cleaned_sentences = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            sentence = str(sentence)
        cleaned_sentence = remove_special_chars.sub('', sentence)
        cleaned_sentence = remove_extra_spaces.sub(' ', cleaned_sentence).strip()
        cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences

hindi_sentences = df['hindi']
english_sentences = df['english']

hindi_sentences = (clear_text(hindi_sentences.tolist()))
english_sentences = (clear_text(english_sentences.tolist()))

print(hindi_sentences[:100], english_sentences[:100])

hi_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def tokenize_hindi_sentences(sentences, tokenizer):
    return [tokenizer.tokenize(sentence) for sentence in sentences]

def tokenize_english_sentences(sentences, tokenizer):
    return [tokenizer(sentence) for sentence in sentences]

tokenized_hindi = tokenize_hindi_sentences(hindi_sentences, hi_tokenizer)
tokenized_english = tokenize_english_sentences(english_sentences, en_tokenizer)

pd.DataFrame({
    'tokenized_hindi': tokenized_hindi,
    'tokenized_english': tokenized_english
}).to_csv('tokenized_sentences.csv', index=False)

print(f"Hindi: {len(tokenized_hindi)}, English: {len(tokenized_english)}")

num_tok = 5
print(f"Hindi: {tokenized_hindi[:num_tok]}, English: {tokenized_english[:num_tok]}")

from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

import pandas as pd
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from transformers import AutoTokenizer
from torchtext.utils import download_from_url, extract_archive
import io

df_token = pd.read_csv('/kaggle/input/token-english-hindi/tokenized_sentences.csv')

tokenized_hindi = df_token['tokenized_hindi'].apply(eval).tolist()
tokenized_english = df_token['tokenized_english'].apply(eval).tolist()

specials=['<unk>', '<pad>', '<bos>', '<eos>']

def yield_tokens(tokens_list):
    for tokens in tokens_list:
        yield tokens

def build_vocab_from_tokens(tokens_list, specials):
    counter = Counter(token for sentence in tokens_list for token in sentence)
    vocab = build_vocab_from_iterator(yield_tokens(tokens_list), specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

hi_vocab = build_vocab_from_tokens(tokenized_hindi, specials)
en_vocab = build_vocab_from_tokens(tokenized_english, specials)

print(f"Hindi Vocabulary Size: {len(hi_vocab)}")
print(f"English Vocabulary Size: {len(en_vocab)}")

import torch

def tokens_to_tensor(tokens, vocab):
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

def data_to_tensors(hindi_tokens_list, english_tokens_list, hi_vocab, en_vocab):
    data = []
    for hi_tokens, en_tokens in zip(hindi_tokens_list, english_tokens_list):
        hi_tensor = tokens_to_tensor(hi_tokens, hi_vocab)
        en_tensor = tokens_to_tensor(en_tokens, en_vocab)
        data.append((hi_tensor, en_tensor))
    return data

data = data_to_tensors(tokenized_hindi, tokenized_english, hi_vocab, en_vocab)

from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
PAD_IDX = hi_vocab['<pad>']
BOS_IDX = hi_vocab['<bos>']
EOS_IDX = hi_vocab['<eos>']

def generate_batch(data_batch):
    hi_batch, en_batch = [], []
    for (hi_item, en_item) in data_batch:
        hi_sequence = torch.cat([torch.tensor([BOS_IDX]), hi_item, torch.tensor([EOS_IDX])], dim=0)
        en_sequence = torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0)

        hi_batch.append(hi_sequence)
        en_batch.append(en_sequence)

    hi_batch = pad_sequence(hi_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)

    return hi_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=generate_batch)

test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=generate_batch)

for hi_batch, en_batch in train_iter:
    print(f"Train batch - Hindi batch shape: {hi_batch.shape}, Eenglish batch shape: {en_batch.shape}")
    break

import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))
        return output, decoder_hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        output = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
        return outputs

INPUT_DIM = len(hi_vocab)
OUTPUT_DIM = len(en_vocab)
ENC_EMB_DIM = 4
DEC_EMB_DIM = 4
ENC_HID_DIM = 8
DEC_HID_DIM = 8
ATTN_DIM = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

import math
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(tqdm(iterator, desc='Training')):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(tqdm(iterator, desc='Evaluating')):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
