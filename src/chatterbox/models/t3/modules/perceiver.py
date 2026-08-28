# Copyright (c) 2025 Resemble AI
# Author: Manmay Nakhashi
# MIT License
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class RelativePositionBias(nn.Module):
    """This class implements a module for relative position bias in neural network models, facilitating self-attention mechanisms by considering positional relationships between elements. It supports causal and non-causal configurations and uses embedding to represent these biases efficiently."""
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        """Initialize relative positional encoding.
        Args:
        scale (float): Scaling factor.
        causal (bool, optional): Whether to use causal masking. Default is False.
        num_buckets (int, optional): Number of buckets for relative positions. Default is 32.
        max_distance (int, optional): Maximum distance between tokens. Default is 128.
        heads (int, optional): Number of attention heads. Default is 8.
        Returns:
        None
        """
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        """Computes the relative position bucket for a given relative position.
        Args:
        relative_position (torch.Tensor): The relative positions to be bucketized.
        causal (bool, optional): If True, treats the sequence as causal. Defaults to True.
        num_buckets (int, optional): The number of buckets to use. Defaults to 32.
        max_distance (int, optional): The maximum distance for which to compute buckets. Defaults to 128.
        Returns:
        torch.Tensor: The bucketized relative positions.
        """
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        """Applies relative positional encoding to query-key dot products.
        Args:
        qk_dots (Tensor): The query-key dot product tensor of shape (*, i, j).
        Returns:
        Tensor: The transformed qk_dots tensor with added relative positional bias.
        """
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)


class AttentionQKV(nn.Module):
    """Class representing a Multi-Head Attention mechanism using Query-Key-Value (QKV) inputs.
    Manages multi-head attention calculations with support for scaled dot-product attention and optional dropout. Can utilize optimized flash attention if enabled.
    """
    def __init__(self, n_heads, head_dim, dropout_rate=0.1, scale=None, flash=False):
        """Initializes a new instance of the class with specified parameters.
        Args:
        n_heads (int): The number of attention heads.
        head_dim (int): The dimensionality of each attention head.
        dropout_rate (float, optional): Dropout rate for the layer. Default is 0.1.
        scale (float, optional): Scaling factor for attention scores. If None, defaults to head_dim ** -0.5.
        flash (bool, optional): Whether to enable flash attention. Default is False.
        Returns:
        None
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.flash = flash
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.flash_config = self.setup_flash_config() if flash else None

    def setup_flash_config(self):
        """Set up flash attention configuration and perform forward pass.
        Args:
        q (tensor): Query tensor.
        k (tensor): Key tensor.
        v (tensor): Value tensor.
        mask (tensor, optional): Attention mask.
        Returns:
        tensor: Output tensor from either flash or scaled dot-product attention.
        """
        # Setup flash attention configuration
        flash_config = {
            'enable_flash': True,
            'enable_math': True,
            'enable_mem_efficient': True
        }
        return flash_config

    def forward(self, q, k, v, mask=None):
        """Forward pass for a multi-head attention mechanism.
        Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        mask (torch.Tensor, optional): Attention mask.
        Returns:
        torch.Tensor: Output tensor after applying scaled dot-product attention.
        """
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        if self.flash:
            out = self.flash_attention(q, k, v, mask=mask)
        else:
            out = self.scaled_dot_product_attention(q, k, v, mask=mask)

        return self.combine_heads(out)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Computes scaled dot-product attention between query (q), key (k), and value (v).
        Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        mask: Optional mask to apply during attention computation.
        Returns:
        Tensor representing the output of scaled dot-product attention.
        """
        sim = torch.einsum("bhlt,bhls->bhts", q, k) * self.scale
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        return torch.einsum("bhts,bhls->bhlt", attn, v)

    def flash_attention(self, q, k, v, mask=None):
        """Applies scaled dot-product attention to queries, keys, and values.
        Args:
        q: Query tensor of shape (bs, length, embed_dim).
        k: Key tensor of shape (bs, length, embed_dim).
        v: Value tensor of shape (bs, length, embed_dim).
        mask: Attention mask tensor of shape (bs, 1, length, length), optional.
        Returns:
        Output tensor after applying attention mechanism.
        """
        config = self.flash_config if self.flash_config else {}
        with torch.backends.cuda.sdp_kernel(**config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.
            )
        return out

    def split_heads(self, x):
        """This class implements an attention block that facilitates spatial positions attending to each other through attention mechanisms. It includes methods for splitting input into multiple heads and combining the outputs of those heads.
        Args:
        n_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        Methods:
        split_heads(x): Splits the input tensor into multiple attention heads.
        combine_heads(x): Combines the outputs of multiple attention heads.
        """
        bs, length, _ = x.shape
        x = x.view(bs, length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        """Combines the heads of a tensor after transposing and flattening.
        Args:
        x (Tensor): Input tensor of shape (bs, _, length, _).
        Returns:
        Tensor: Output tensor of shape (bs, length, -1).
        """
        bs, _, length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bs, length, -1)


class AttentionBlock2(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other,
    using AttentionQKV and separate linear transformations for Q, K, and V.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        relative_pos_embeddings=False,
        flash_attention=True,
        dropout_rate=0.2,
        scale=None
    ):
        """Initializes a new instance with specified parameters.
        Args:
        channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        num_head_channels (int, optional): Number of channels per head. Defaults to -1.
        relative_pos_embeddings (bool, optional): Whether to use relative positional embeddings. Defaults to False.
        flash_attention (bool, optional): Whether to use flash attention algorithm. Defaults to True.
        dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.2.
        scale (float, optional): Scaling factor. Defaults to None.
        Returns:
        None
        """
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, and V
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKV(self.num_heads, channels // self.num_heads, dropout_rate=dropout_rate, flash=flash_attention, scale=scale)

        self.proj_out = nn.Linear(channels, channels)

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x1, x2, mask=None):
        """Applies a multi-head attention mechanism to two input tensors and returns their weighted sum.
        Args:
        x1: The first input tensor.
        x2: The second input tensor.
        mask: An optional mask for the attention mechanism.
        Returns:
        A tensor resulting from adding the input tensors with their weighted sum.
        """
        b1, c1, *spatial1 = x1.shape
        b2, c2, *spatial2 = x2.shape

        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return (x1 + h).reshape(b1, c1, *spatial1)


class Perceiver(nn.Module):
    """Inspired by https://arxiv.org/abs/2103.03206"""
    def __init__(self, pre_attention_query_token=32, pre_attention_query_size=1024, embedding_dim=1024, num_attn_heads=4):
        """
        Initialize the perceiver module.

        :param pre_attention_query_token: Number of query tokens for pre-attention
        :param pre_attention_query_size: Size of each query token
        :param embedding_dim: Dimension of the embedding space
        :param num_attn_heads: Number of attention heads
        """
        super().__init__()

        # Initialize the pre-attention query parameter
        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, pre_attention_query_token, pre_attention_query_size)
        )

        # Calculate the variance for uniform initialization
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (pre_attention_query_token + pre_attention_query_token))

        # Initialize the pre-attention query with uniform distribution
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)

        # Initialize the attention block
        self.attn = AttentionBlock2(embedding_dim, num_attn_heads)

    def forward(self, h):
        """
        Forward pass of the perceiver module.
        :param h: Input tensor
        :return: Output after applying attention mechanisms
        """
        # Expand the pre-attention query to match the batch size of the input
        query_ = self.pre_attention_query.expand(h.shape[0], -1, -1)
        # Apply the first attention mechanism (cross-attention)
        pre_att = self.attn(query_, h)
        # Apply the second attention mechanism (self-attention)
        attn = self.attn(pre_att, pre_att)
        return attn
