import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def d(tensor=None):
    """ Return best available device """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def mask(matrices, mask_val, mask_diagonal=True):
    """ Mask all values in place of given batch of matrices. Upper triangle becomes mask_val """
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = mask_val

class SelfAttention(nn.Module):
    """ Multi-headed, scaled dot-product self attention """
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        # q, k, v
        # Map query against set of keys to values
        self.to_queries = nn.Linear(emb, emb * heads, bias=False)
        self.to_keys = nn.Linear(emb, emb * heads, bias=False)
        self.to_values = nn.Linear(emb, emb * heads, bias=False)
        # Unify output of heads to a single emb-vector
        self.unify_heads = nn.Linear(heads * emb, emb)
    
    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        # Init
        queries = self.to_queries(x).view(b, t, h, e)
        keys = self.to_keys(x).view(b, t, h, e)
        values = self.to_values(x).view(b, t, h, e)
        # Fold head into batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        # Get dot product
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # Dot contains b * h, t x t matrices with raw self attention logits
        dot = dot / math.sqrt(e)
        # Mask lower half of the dot matrix incl diagonal
        if self.mask:
            mask(dot, maskval=float('-inf'), mask_diagonal=False)
        # Dot contains row-wise self-attention probaiblities
        dot = F.softmax(dot, dim=2)
        # Remove NaN
        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1 :] = 0.0
        # Apply self attention to values
        out = torch.bmm(dot, values).view(b, h, t, e)
        # Swap h, t
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        # Unify heads
        return self.unify_heads(out)

class TransformerBlock(nn.Module):
    """ Transformer block: attn + norm -> ff + norm """
    def __init__(self, emb, n_heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        # Self attention layer
        self.attention = SelfAttention(emb, n_heads=n_heads)
        # Layer normalization
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        # Feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention
        attended = self.attention(x)
        # First norm layer
        x = self.norm1(attended + x)
        # Dropout
        x = self.dropout(x)
        # Feed-forward layer
        fed_forward = self.ff(x)
        # Second norm layer
        x = self.norm2(fed_forward + x)
        # Dropout
        x = self.dropout(x)
        # Complete
        return x

class GenTransformer(nn.Module):
    """ Autoregressive transformer model """
    def __init__(self, emb, heads, depth, seq_length, n_tokens):
        super().__init__()
        # number of tokens
        self.n_tokens = n_tokens
        # Token Embedding
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=n_tokens)
        # Positional embedding
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        # Linearly transform embeddings to unify
        self.unify_embeddings = nn.Linear(2 * emb, emb)
        # Depth of transformer
        t_blocks = []
        for i in range(depth):
            t_blocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=seq_length,
                    mask=True
                )
            )
        self.t_blocks = nn.Sequential(*t_blocks)
        # To probabilities
        self.to_probs = nn.Linear(emb, n_tokens)

    def forward(self, x):
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        # Get positional embeddings of the batch
        positions = self.pos_embedding(torch.arrange(t, device=d()))[None, :, :].expand(b, t, e)
        # Unify embeddings
        x = self.unify_embeddings(torch.cat((tokens, positions), dim=2).view(-1, 2 * e)).view(b, t, e)
        # Run the batch through transformer blocks
        x = self.t_blocks(x)
        x = self.to_probs(x.view(b * t, e)).view(b, t, self.n_tokens)
        # Predicted log probability for each token based on preceding tokens
        return F.log_softmax(x, dim=2)

