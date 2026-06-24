import mlx.core as mx
import mlx.nn as nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen: int, vocab_size:int, embed_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)

class TransformerBlock(nn.Module):
    def __init__(self, emed_dim: int, num_heads:int, ff_dim: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(emed_dim, num_heads)
    def __call__(self, x, mask=None):
        attn_out = self.attention(x, x, x, mask=mask)
        x = x + attn_out
        return x 


class NanoLLM(nn.Module):
    def __init__(self, maxlen: int, vocab_size:int, embed_dim:int, num_heads:int, feed_forward_dim:int , num_transformer_blocks: int):
        super().__init__()
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim)
            for _ in range(num_transformer_blocks)
        ]
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        logits = self.output_layer(x)
        return logits