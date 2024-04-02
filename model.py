import torch
import torch.nn as nn
from dataset import *
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout=0.1):
        super().__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, dropout)
        self.decoder = DecoderRNN(hidden_size, output_size, device)
    
    def forward(self, input_tensor, target_tensor=None):
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden, target_tensor)
        return decoder_outputs, decoder_hidden, decoder_attn
        

class Transformer(nn.Module):
    def __init__(
            self, 
            src_vocab_size, 
            tgt_vocab_size, 
            embedding_size, 
            num_heads, 
            num_encoder_layers, 
            num_decoder_layers, 
            max_length,
            forward_expansion,
            dropout,
            src_pad_idx,
            device
    ):
        super().__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_positional_embedding = nn.Embedding(max_length, embedding_size)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.tgt_positional_embedding = nn.Embedding(max_length, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
    
    def make_src_mask(self, src):
        '''
        Parameters:
        src: (src_len, N)
        '''
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask
    
    def forward(self, src, tgt):
        src_seq_length, N = src.shape
        tgt_seq_length, N = tgt.shape

        src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        tgt_positions = torch.arange(0, tgt_seq_length).unsqueeze(1).expand(tgt_seq_length, N).to(self.device)

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_positional_embedding(src_positions))
        embed_tgt = self.dropout(self.tgt_positional_embedding(tgt) + self.tgt_positional_embedding(tgt_positions))

        src_padding_mask = self.make_src_mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(self.device)

        out = self.transformer(embed_src, embed_tgt, src_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask)

        return out