import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# TODO comment, use paper as reference

'''
Get the device which will run the model
'''
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Clone a module n times
'''
def clone(module, n):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(n)])

'''
Narrow Multihead attention. 
'''
class SelfAttention(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.1, relative_pos=True):
        super().__init__()
        '''
        if d_model % heads != 0:
            raise AttentionError("Number of heads does not divide model dimension")
        '''
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = 1024

        self.relative_pos = relative_pos

        self.Er = None # Default
        if relative_pos:
            self.Er = torch.rand([heads, self.max_length, s], device=_device)

    # TODO implement from paper
    def forward(self, x, mask):
        '''
        if self.relative_pos:
            Er self.Er[:, embedding_start:, :].unsqueeze(0)
            QEr = torch.matmul(queries, Er.transpose(-1,-2))
            QEr = self.mask(QEr)
            SRel = self.skew(QEr).contiguous().view(b * h, t , t)
        else:
            SRel = torch.zeros([b * h, t, t], device=_device)
        '''
        return

    # Mask the upper triangle of the matrix
    def mask(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=_device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    # Skew QEr to correct logit positions
    def skew(self, qe):
        # Pad
        padded_qe = F.pad(qe, [1,0])
        # Reshape
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        # Slice
        return padded_qe[:,:,1:,:]

'''
Embedding scaled by the sqrt of the models hidden state size
'''
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

'''
Implements FNN Equation:
FNN(x) = max(0, xW1 + b1)W2 + b2
'''
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

'''
Decoder layer.
Input receives masked attention, then sent through attention and ff sublayers
'''
class DecoderLayer(nn.Module):
    def __init__(self, size, n_heads, d_ff, dropout, relative_pos):
        super().__init__()
        self.size = size
        self.self_attn = SelfAttention(size, n_heads, dropout, relative_pos)
        self.feed_forward = PositionwiseFeedForward(size, d_ff, dropout)
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn = self.self_attn(x, mask)
        x = x + self.dropout1(attn)
        x = self.norm1(x)

        ff = self.feed_forward(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)

        return x

'''
Music generation model
'''
class MusicTransformer(nn.Module):
    # Initialize model
    def __init__(
        self,
        n_tokens, # number of commands in encoded musical sequence
        seq_length = None, # length of padded input/target sequences
        d_model = 64, # dimensions of embedded sequences
        n_heads = 4, # number of attention heads
        depth = 2, # number of stacked transformer layers
        d_ff = 512, # dimensionality of dense sublayer
        dropout = 0.1, # probability of dropout in dropout sublayer
        pos_encoding = False, # if true, use a positional encoding layer
        relative_pos = True # if true, use relative positional embeddings
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.embed = Embeddings(n_tokens, d_model)
        self.pos_encoding = pos_encoding
        # For now, this assumes no positional encoding
        # TODO check if seq_length is None
        self.pos = nn.Embedding(seq_length, d_model)
        self.to_scores = nn.Linear(d_model, n_tokens)
        self.layers = clone(DecoderLayer(d_model, n_heads, d_ff, dropout, relative_pos), depth)
        self.norm = nn.LayerNorm(d_model)

    # TODO implement
    def forward(self, x, mask=None):
        x = self.embed(x)