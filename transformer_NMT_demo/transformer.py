# import torch
# import torch.nn as nn

# torch.manual_seed(1234)
# class SelfAttention(nn.Module):
#     def __init__(self, emb_size, n_heads):
#         super().__init__()
#         assert emb_size % n_heads == 0, "embedding size should be multiple of number of heads"
#         self.head_dim = emb_size // n_heads
#         self.n_heads = n_heads
#         self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(emb_size, emb_size)
#         self.softmax = nn.Softmax(dim=3)
#         self.sqrt_dk = self.head_dim ** (1/2)
    
#     def forward(self, query, key, value, mask):
#         '''
#         query: N x L_src x n_heads x dk
#         key: N x L_src x n_heads x dk
#         value: N x L_tgt x n_heads x dk
#         '''
#         N, L_src, _ = query.size()
#         N, L_tgt, _ = value.size()
#         query = query.reshape(N, L_src, self.n_heads, self.head_dim)
#         key = key.reshape(N, L_tgt, self.n_heads, self.head_dim)
#         value = value.reshape(N, L_tgt, self.n_heads, self.head_dim)
#         q = self.W_q(query)
#         k = self.W_k(key)
#         v = self.W_v(value) # nvhk
#         energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
#         # energy = q.transpose(1, 2) @ k.transpose(1, 2).transpose(2, 3)
        
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float('-1e20'))
        
#         attention = self.softmax(energy / self.sqrt_dk) # nhqk
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, L_src, self.n_heads * self.head_dim)
#         # out = (attention @ v.transpose(1, 2)).transpose(1, 2).reshape(N, L_src, self.n_heads * self.head_dim)
        
#         return self.fc_out(out)

# class TransformerBlock(nn.Module):
#     def __init__(self, emb_size, n_heads, dropout, forward_expansion):
#         super().__init__()
#         self.attention = SelfAttention(emb_size, n_heads)
#         self.ffn = nn.Sequential(
#             nn.Linear(emb_size, emb_size * forward_expansion),
#             nn.ReLU(),
#             nn.Linear(emb_size * forward_expansion, emb_size)
#         )
#         self.norm1 = nn.LayerNorm(emb_size)
#         self.norm2 = nn.LayerNorm(emb_size)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, query, key, value, mask):
#         attn = self.attention(query, key, value, mask)
#         x = self.dropout(self.norm1(query + attn))
#         return self.dropout(self.norm2(x + self.ffn(x)))
    
# class Encoder(nn.Module):
#     def __init__(self, vocab_size, emb_size, num_layers, n_heads, forward_expansion, dropout, max_length, device):
#         super().__init__()
#         self.word_embedding = nn.Embedding(vocab_size, emb_size)
#         self.positional_embedding = nn.Embedding(max_length, emb_size)
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList([TransformerBlock(emb_size, n_heads, dropout, forward_expansion) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(dropout)
#         self.device = device

#     def forward(self, x, mask):
#         N, seq_length = x.size()
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
#         for layer in self.layers:
#             out = layer(out, out, out, mask)
#         return out

# class DecoderBlock(nn.Module):
#     def __init__(self, emb_size, n_heads, dropout, forward_expansion):
#         super().__init__()
#         self.masked_attention = SelfAttention(emb_size, n_heads)
#         self.encoder_block = TransformerBlock(emb_size, n_heads, dropout, forward_expansion)
#         self.norm = nn.LayerNorm(emb_size)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x, key, value, src_mask, tgt_mask):
#         masked_attn = self.masked_attention(x, x, x, tgt_mask)
#         query = self.dropout(self.norm(x + masked_attn))
#         return self.encoder_block(query, key, value, src_mask)
        
    
# class Decoder(nn.Module):
#     def __init__(self, tgt_vocab_size, emb_size, num_layers, n_heads, forward_expansion, dropout, max_length, device):
#         super().__init__()
#         self.word_embedding = nn.Embedding(tgt_vocab_size, emb_size)
#         self.positional_embedding = nn.Embedding(max_length, emb_size)
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList([DecoderBlock(emb_size, n_heads, dropout, forward_expansion) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(emb_size, tgt_vocab_size)
#         self.device = device

#     def forward(self, x, encoder_out, src_mask, tgt_mask):
#         N, seq_length = x.size()
#         positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
#         x = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
#         for layer in self.layers:
#             x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)
#         return self.fc(x)

# class Transformer(nn.Module):
#     def __init__(
#             self, 
#             src_vocab_size, 
#             tgt_vocab_size, 
#             src_pad_idx, 
#             tgt_pad_idx, 
#             emb_size=128, 
#             num_layers=2, 
#             n_heads=8, 
#             forward_expansion=4,
#             dropout=0,
#             device='cpu',
#             max_length=100        
#         ):
#         super().__init__()
#         self.encoder = Encoder(src_vocab_size, emb_size, num_layers, n_heads, forward_expansion, dropout, max_length, device)
#         self.decoder = Decoder(tgt_vocab_size, emb_size, num_layers, n_heads, forward_expansion, dropout, max_length, device)
#         self.src_pad_idx = src_pad_idx
#         self.tgt_pad_idx = tgt_pad_idx
#         self.device = device
    
#     def make_src_mask(self, src):
#         src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
#         return src_mask.to(self.device)
    
#     def make_tgt_mask(self, tgt):
#         N, tgt_len = tgt.size()
#         tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).expand(N, 1, tgt_len, tgt_len)
#         return tgt_mask.to(self.device)

#     def forward(self, src, tgt):
#         src_mask = self.make_src_mask(src)
#         tgt_mask = self.make_tgt_mask(tgt)
#         encoded_src = self.encoder(src, src_mask)
#         return self.decoder(tgt, encoded_src, src_mask, tgt_mask)

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
#     tgt = torch.tensor([[1,7,3,4,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)

#     src_pad_idx = 0
#     tgt_pad_idx = 0
#     src_vocab_size = 10
#     tgt_vocab_size = 10
#     model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx).to(device)
#     out = model(x, tgt[:, :-1])
#     print(out.norm())

import torch
import torch.nn as nn

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