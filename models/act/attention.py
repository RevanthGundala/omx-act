import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        use_cross_attention=False,
    ):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(dim, num_heads, dropout)
        self.cross_attn = Attention(dim, num_heads, dropout, use_cross_attention=True) if use_cross_attention else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim) if use_cross_attention else None
        self.mlp_norm = nn.LayerNorm(dim)
        self.self_residual_dropout = nn.Dropout(dropout)
        self.cross_residual_dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out=None, key_padding_mask=None, pos=None, query_pos=None):
        self_pos = query_pos if self.cross_attn is not None else pos
        skip = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask, pos=self_pos)
        x = self.self_norm(skip + self.self_residual_dropout(x))

        if self.cross_attn is not None and encoder_out is not None:
            skip = x
            x = self.cross_attn(x, encoder_out=encoder_out, pos=pos, query_pos=query_pos)
            x = self.cross_norm(skip + self.cross_residual_dropout(x))

        return self.mlp_norm(x + self.mlp(x))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, use_cross_attention=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_cross_attention = use_cross_attention
        self.last_attn_weights = None
        if use_cross_attention:
            self.cross_q_proj = nn.Linear(dim, dim)
            self.cross_k_proj = nn.Linear(dim, dim)
            self.cross_v_proj = nn.Linear(dim, dim)

    def forward(self, x, encoder_out=None, key_padding_mask=None, pos=None, query_pos=None):
        if self.use_cross_attention and encoder_out is not None:
            q_in = x if query_pos is None else x + query_pos
            k_in = encoder_out if pos is None else encoder_out + pos
            cross_q = rearrange(
                self.cross_q_proj(q_in),
                "b i (h d) -> b h i d",
                h=self.num_heads,
            )
            cross_k = rearrange(
                self.cross_k_proj(k_in),
                "b j (h d) -> b h j d",
                h=self.num_heads,
            )
            cross_v = rearrange(
                self.cross_v_proj(encoder_out),
                "b j (h d) -> b h j d",
                h=self.num_heads,
            )
            attn_weights = einsum(
                cross_q,
                cross_k,
                "b h i d, b h j d -> b h i j",
            ) / (self.head_dim ** 0.5)
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(key_padding_mask == 0, torch.finfo(attn_weights.dtype).min)
            attn_weights = F.softmax(attn_weights, dim=-1)
            self.last_attn_weights = attn_weights.detach()

            attn_weights = self.dropout(attn_weights)
            attn_output = einsum(
                attn_weights,
                cross_v,
                "b h i j, b h j d -> b h i d",
            )
            attn_output = rearrange(
                attn_output,
                "b h i d -> b i (h d)",
            )

            output = self.out_proj(attn_output)
            return output

        qkv = rearrange(
            self.qkv_proj(x),
            "b n (qkv h d) -> qkv b h n d",
            qkv=3,
            h=self.num_heads,
        )
        if pos is None:
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qk = rearrange(
                self.qkv_proj(x + pos),
                "b n (qkv h d) -> qkv b h n d",
                qkv=3,
                h=self.num_heads,
            )
            q, k, v = qk[0], qk[1], qkv[2]

        attn_weights = einsum(
            q,
            k,
            "b h i d, b h j d -> b h i j",
        ) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask == 0, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        self.last_attn_weights = attn_weights.detach()

        attn_weights = self.dropout(attn_weights)
        attn_output = einsum(
            attn_weights,
            v,
            "b h i j, b h j d -> b h i d",
        )
        attn_output = rearrange(
            attn_output,
            "b h i d -> b i (h d)",
        )

        output = self.out_proj(attn_output)
        return output
