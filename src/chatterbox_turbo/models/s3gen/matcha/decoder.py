import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat

from .transformer import BasicTransformerBlock


class SinusoidalPosEmb(torch.nn.Module):
    """A PyTorch module for generating sinusoidal positional embeddings."""
    def __init__(self, dim):
        """Initializes the sinusoidal positional embedding with a given dimension.
        Args:
        dim (int): The dimension of the positional embedding, must be even.
        Returns:
        torch.Tensor: The resulting positional embedding tensor.
        """
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        """Computes a positional embedding for input tensor `x`.
        Args:
        x (torch.Tensor): Input tensor.
        scale (int, optional): Scale factor for the embedding. Default is 1000.
        Returns:
        torch.Tensor: Positional embedding tensor with shape `[N, C]`.
        """
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    """A 1D convolutional block followed by group normalization and Mish activation, designed for sequential processing tasks."""
    def __init__(self, dim, dim_out, groups=8):
        """Initializes a 1D residual block with convolutional layers and group normalization.
        Args:
        dim (int): Input dimension.
        dim_out (int): Output dimension.
        groups (int): Number of groups for group normalization.
        Returns:
        torch.nn.Module: The initialized ResnetBlock1D module.
        """
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        """Applies a forward pass through a ResnetBlock1D layer.
        Args:
        x (Tensor): Input tensor of shape (batch_size, channels, sequence_length).
        mask (Tensor): Binary mask for attention mechanism, same shape as x.
        Returns:
        Tensor: Output tensor after applying the block and masking.
        """
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """A ResnetBlock1D class representing a 1D residual block for processing time-series data with optional grouping and attention mechanisms."""
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        """Initializes a model with specified dimensions and parameters.
        Args:
        dim (int): Input dimension.
        dim_out (int): Output dimension.
        time_emb_dim (int): Dimension of time embedding.
        groups (int, optional): Number of groups for group normalization. Default is 8.
        Returns:
        Tensor: The output tensor after processing the input through multiple blocks and a residual convolution.
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        """Applies a downsampling operation to a 1D input tensor.
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, length).
        Returns:
        torch.Tensor: Downsampled tensor of shape (batch_size, channels, length // 2).
        """
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    """Class representing a 1D downsampling module using a convolutional layer.
    Class for generating time embeddings by projecting input channels to a higher dimension.
    """
    def __init__(self, dim):
        """Initializes a TimestepEmbedding module.
        Args:
        in_channels (int): Number of input channels.
        time_embed_dim (int): Dimensionality of time embedding.
        act_fn (str, optional): Activation function name (default: "silu").
        out_dim (int, optional): Output dimension (default: None).
        Returns:
        TimestepEmbedding module.
        """
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        """Applies a convolutional layer to the input.
        Args:
        x (Tensor): Input tensor.
        Returns:
        Tensor: Output tensor after applying the convolutional layer.
        """
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    """A class for creating a timestep embedding layer, which transforms input channels into a higher-dimensional space based on the timestep information."""
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        """Initializes a neural network layer.
        Args:
        in_channels (int): Number of input channels.
        time_embed_dim (int): Dimensionality of time embeddings.
        act_fn (str, optional): Activation function to use. Default is "silu".
        out_dim (int, optional): Output dimension. If None, defaults to `in_channels`.
        post_act_fn (str, optional): Post-activation function. Default is None.
        cond_proj_dim (int, optional): Dimensionality of conditional projection.
        Returns:
        None
        """
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        """Applies a sequence of linear transformations and activations to the input sample.
        Args:
        sample (Tensor): Input tensor.
        condition (Tensor, optional): Optional condition tensor for conditional projection.
        Returns:
        Tensor: Transformed output tensor.
        """
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        """Initializes a convolutional or transposed convolutional layer based on given parameters.
        Args:
        channels (int): Number of input and output channels.
        use_conv (bool): If True, uses standard convolution; otherwise, uses transposed convolution.
        use_conv_transpose (bool): If True, uses transposed convolution; if False and use_conv is also False, no convolution layer is used.
        out_channels (int): Number of output channels. Defaults to None, in which case it equals the number of input channels.
        name (str): Name of the layer.
        Returns:
        torch.Tensor: Output tensor after applying the chosen convolutional operation.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        """Computes forward pass for a Conformer wrapper.
        Args:
        inputs (Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
        Tensor: Output tensor after applying convolutional and interpolation operations if specified.
        """
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class ConformerWrapper(ConformerBlock):
    """A subclass of ConformerBlock that provides a wrapper for conformer architecture components."""
    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        """Initializes a transformer model with specified parameters.
        Args:
        dim (int): The dimensionality of the input and output.
        dim_head (int): The dimensionality of each attention head.
        heads (int): The number of attention heads.
        ff_mult (int): Multiplier for feedforward network hidden size.
        conv_expansion_factor (int): Expansion factor for convolutional layers.
        conv_kernel_size (int): Kernel size for convolutional layers.
        attn_dropout (float): Dropout rate for attention mechanism.
        ff_dropout (float): Dropout rate for feedforward network.
        conv_dropout (float): Dropout rate for convolutional layers.
        conv_causal (bool): Whether to use causal convolution.
        """
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        """```python
        Calls the forward method of the superclass with the provided hidden states and attention mask.
        Args:
        hidden_states (Tensor): Input tensor for the decoder.
        attention_mask (Tensor): Mask to apply during attention mechanism.
        encoder_hidden_states (Tensor, optional): Hidden states from the encoder. Defaults to None.
        encoder_attention_mask (Tensor, optional): Attention mask from the encoder. Defaults to None.
        timestep (int, optional): Current time step in the sequence. Defaults to None.
        Returns:
        Tensor: Output tensor after processing through the decoder.
        ```
        """
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(nn.Module):
    """A PyTorch module implementing a decoder architecture for image processing tasks, utilizing transformer-based blocks for feature transformation and upsampling."""
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
    ):
        """Initializes a transformer-based model architecture.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple): Tuple defining the number of channels in each block.
        dropout (float): Dropout rate.
        attention_head_dim (int): Dimension of attention heads.
        n_blocks (int): Number of blocks.
        num_mid_blocks (int): Number of middle blocks.
        num_heads (int): Number of attention heads per transformer block.
        act_fn (str): Activation function type.
        down_block_type (str): Type of down block.
        mid_block_type (str): Type of middle block.
        up_block_type (str): Type of up block.
        Returns:
        None
        """
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            out_channels = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        """Constructs a neural network block based on the specified type.
        Args:
        block_type (str): Type of block to create ('conformer' or 'transformer').
        dim (int): Dimensionality of the input.
        attention_head_dim (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        act_fn (str): Activation function.
        Returns:
        nn.Module: Constructed block.
        """
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        """Initializes weights of the model's layers using Kaiming normal distribution for Conv1d and Linear layers, and constant initialization for GroupNorm layers.
        Args:
        self: The instance of the class containing the layers to be initialized.
        Returns:
        None
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

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask
