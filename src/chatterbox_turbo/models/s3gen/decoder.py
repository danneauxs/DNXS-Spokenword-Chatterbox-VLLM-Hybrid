# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from .utils.mask import add_optional_chunk_mask
from .matcha.decoder import SinusoidalPosEmb, Block1D, ResnetBlock1D, Downsample1D, \
    TimestepEmbedding, Upsample1D
from .matcha.transformer import BasicTransformerBlock
from .utils.intmeanflow import get_intmeanflow_time_mixer


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Converts a boolean mask to a bias tensor.
    Args:
    mask (torch.Tensor): Input boolean mask.
    dtype (torch.dtype): Target data type for the output tensor.
    Returns:
    torch.Tensor: A bias tensor derived from the input mask.
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask



class Transpose(torch.nn.Module):
    """A module for transposing dimensions of a tensor.
    :param dim0: The first dimension to transpose.
    :param dim1: The second dimension to transpose.
    Represents a causal block in 1D.
    :param dim: Input dimension.
    :param dim_out: Output dimension.
    """
    def __init__(self, dim0: int, dim1: int):
        """Initializes a causal block for 1D data. Args: dim (int): The number of input features. dim_out (int): The number of output features. Returns: None"""
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        """Transposes two dimensions of a tensor.
        Args:
        x (torch.Tensor): Input tensor.
        dim0 (int): First dimension to transpose.
        dim1 (int): Second dimension to transpose.
        Returns:
        torch.Tensor: Tensor with transposed dimensions.
        """
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalBlock1D(Block1D):
    """A causal block for 1D data processing that includes a causal convolution, layer normalization, and Mish activation function, applying a mask to inputs and outputs."""
    def __init__(self, dim: int, dim_out: int):
        """Initializes a CausalBlock1D instance.
        Args:
        dim (int): Dimension of the input and output.
        dim_out (int): Output dimension of the block.
        Returns:
        torch.Tensor: The forward pass output.
        """
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Applies a causal convolution along the time dimension.
        Args:
        x (torch.Tensor): Input tensor.
        mask (torch.Tensor): Mask tensor to apply for causality.
        Returns:
        torch.Tensor: Output tensor after applying the causal convolution and masking.
        """
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    """A causal convolutional block for 1D data.
    Attributes:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (int): Size of the convolving kernel.
    stride (int, optional): Stride of the convolution. Default is 1.
    dilation (int, optional): Spacing between kernel elements. Default is 1.
    """
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        """Initialize a CausalResnetBlock1D instance.
        Args:
        dim (int): Input dimension.
        dim_out (int): Output dimension.
        time_emb_dim (int): Dimension of time embeddings.
        groups (int, optional): Number of groups for grouped convolution. Default is 8.
        Returns:
        None
        """
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class CausalConv1d(torch.nn.Conv1d):
    """A 1D convolutional layer that respects causal ordering by disabling padding on one side.
    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int): Size of the convolving kernel.
    - stride (int, optional): Stride of the convolution. Default: 1
    - dilation (int, optional): Spacing between kernel elements. Default: 1
    - groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    - bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: True
    - padding_mode (str, optional): Type of padding. Default: 'zeros
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        """Initializes a causal 1D convolutional layer.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int, optional): Stride of the convolution. Default: 1
        dilation (int, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        padding_mode (str, optional): Type of padding algorithm to use. Default: 'zeros'
        Returns:
        None
        """
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        """Applies causal convolution to input tensor.
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).
        Returns:
        torch.Tensor: Output tensor after applying causal convolution.
        """
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x
class ConditionalDecoder(nn.Module):
    """A PyTorch module implementing a conditional decoder with various customizable parameters for neural network architectures."""
    def __init__(
        self,
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        meanflow=False,
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        channels = tuple(channels)
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # NOTE jrm: `static_chunk_size` is missing?
        self.static_chunk_size = 0

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim) if self.causal else \
                ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else
                CausalConv1d(output_channel, output_channel, 3) if self.causal else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim) if self.causal else \
                ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            ) if self.causal else ResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3) if self.causal else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = CausalBlock1D(channels[-1], channels[-1]) if self.causal else Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()
        self.time_embed_mixer = None
        if self.meanflow:
            self.time_embed_mixer = get_intmeanflow_time_mixer(time_embed_dim)


    @property
    def dtype(self):
        """Returns the data type of the weight tensor in `final_proj`.
        Args:
        None
        """
        return self.final_proj.weight.dtype

    def initialize_weights(self):
        """Initializes weights for convolutional and linear layers using Kaiming normal distribution and sets biases to zero. Also initializes weights of group normalization layers to ones and biases to zeros.
        Args:
        x: Input tensor.
        mask: Mask tensor.
        mu: Mean parameter.
        t: Time step.
        spks: Spike tensor (optional).
        cond: Conditional input (optional).
        r: Rate parameter (optional).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x: (B, 80, T)
            mask (_type_)
            t (_type_): shape (batch_size)
            spks (_type_, optional) Defaults to None.
            cond (_type_, optional)
            r: end time for meanflow mode (shape (1,) tensor)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        if self.meanflow:
            r = self.time_embeddings(r).to(t.dtype)
            r = self.time_mlp(r)
            concat_embed = torch.cat([t, r], dim=1)
            t = self.time_embed_mixer(concat_embed)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            # attn_mask = torch.matmul(mask_down.transpose(1, 2).contiguous(), mask_down)
            attn_mask = add_optional_chunk_mask(x, mask_down.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            # attn_mask = torch.matmul(mask_mid.transpose(1, 2).contiguous(), mask_mid)
            attn_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            # attn_mask = torch.matmul(mask_up.transpose(1, 2).contiguous(), mask_up)
            attn_mask = add_optional_chunk_mask(x, mask_up.bool(), False, False, 0, self.static_chunk_size, -1)
            attn_mask = mask_to_bias(attn_mask == 1, x.dtype)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_mask,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
