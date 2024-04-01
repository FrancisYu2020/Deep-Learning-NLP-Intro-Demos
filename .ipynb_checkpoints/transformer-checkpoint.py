import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, emb_size, n_heads):
        super().__init__()
        assert emb_size % n_heads == 0, "embedding size should be multiple of number of heads"
        self.head_dim = emb_size // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax()
    
    def forward(self, query, key, value, mask):
        '''
        query: N x L_src x n_heads x dk
        key: N x L_src x n_heads x dk
        value: N x L_tgt x n_heads x dk
        '''
        N, L_src, _ = query.size()
        N, L_tgt, _ = value.size()
        query = query.reshape(N, L_src, self.n_heads, self.head_dim)
        key = key.reshape(N, L_tgt, self.n_heads, self.head_dim)
        value = value.reshape(N, L_tgt, self.n_heads, self.head_dim)
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value) # nvhk
        # energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        energy = q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = self.softmax(energy / torch.sqrt(self.head_dim), dim=3) # nhqk
        # out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, L_src, self.n_heads * self.emb_size)
        out = (attention @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(L_src, self.n_heads * self.emb_size)
        
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, n_heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(emb_size, n_heads)
        