#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)


from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchaudio.compliance.kaldi as Kaldi


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def extract_feature(audio):
    """Extracts features from audio data using Kaldi's fbank algorithm.
    Args:
    audio (list): List of audio tensors.
    Returns:
    tuple: Padded features, lengths of each feature, and times of each audio segment.
    """
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # padding for batch inference
    features_padded = pad_list(features, pad_value=0)
    # features = torch.cat(features)
    return features_padded, feature_lengths, feature_times


class BasicResBlock(torch.nn.Module):
    """A basic residual block for use in convolutional neural networks. Defines a sequence of convolutions and batch normalization layers, with an optional shortcut connection if necessary."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Initializes a BasicResBlock for use in a residual network.
        Args:
        in_planes (int): Number of input planes.
        planes (int): Number of output planes.
        stride (int): Stride value for the first convolution layer (default is 1).
        Returns:
        None
        """
        super(BasicResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Applies a forward pass through a basic residual block.
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
        torch.Tensor: Output tensor after applying the residual block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(torch.nn.Module):
    """A PyTorch module implementing a Fully Convolutional Module (FCM) with residual blocks for feature extraction."""
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        """Initializes an FCM model with specified block type, number of blocks per layer, initial number of channels, and feature dimension. Args: block (BasicResBlock): The type of residual block to use. num_blocks (list): Number of blocks in each of the two main layers. m_channels (int): Initial number of channels in convolutional layers. feat_dim (int): Feature dimension for the final output. Returns: None"""
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = torch.nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = torch.nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Creates a layer of blocks.
        Args:
        block (type): Block type.
        planes (int): Number of output channels for each convolution in the block.
        num_blocks (int): Number of blocks in the layer.
        stride (int): Stride for the first block's convolution.
        Returns:
        torch.nn.Sequential: Sequential layer containing the specified number of blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        """Applies a series of convolutional and batch normalization layers to input tensor `x`.
        Args:
        x (torch.Tensor): Input tensor.
        Returns:
        torch.Tensor: Processed tensor after applying convolutional and batch normalization layers.
        """
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str, channels):
    """Constructs a sequential neural network layer based on a configuration string.
    Args:
    config_str: A string specifying the layers in the format "relu-prelu-batchnorm".
    channels: The number of input channels for each layer.
    Returns:
    A torch.nn.Sequential object containing the specified layers.
    """
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    """Computes mean and standard deviation along a specified dimension and concatenates them.
    Args:
    x (torch.Tensor): Input tensor.
    dim (int, optional): Dimension to apply pooling over. Defaults to -1.
    keepdim (bool, optional): If True, keeps the reduced dimensions with size 1. Defaults to False.
    unbiased (bool, optional): Whether to use Bessel's correction for sample standard deviation. Defaults to True.
    eps (float, optional): A small value added to the denominator to prevent division by zero. Defaults to 1e-2.
    Returns:
    torch.Tensor: Concatenated tensor of mean and standard deviation.
    """
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(torch.nn.Module):
    """A module for performing statistics pooling on input data.
    TDNNLayer applies a series of convolutional operations to input data.
    """
    def forward(self, x):
        """Applies statistics pooling to input tensor `x`.
        Args:
        x (Tensor): Input tensor.
        Returns:
        Tensor: Result of statistics pooling.
        """
        return statistics_pooling(x)


class TDNNLayer(torch.nn.Module):
    """A TDNNLayer class implementing a Time-Delay Neural Network layer for processing sequential data. Supports batch normalization and ReLU activation by default."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        """Initialize a TDNNLayer with specified parameters.
        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Size of the convolutional kernel.
        - stride (int, optional): Stride for the convolution. Default is 1.
        - padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        - dilation (int, optional): Spacing between kernel elements. Default is 1.
        - bias (bool, optional): If True, adds a learnable bias to the output. Default is False.
        - config_str (str, optional): Configuration string for additional layer settings. Default is "batchnorm-relu".
        Returns:
        None
        """
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        """Applies a CAM (Channel Attention Module) layer.
        Args:
        bn_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        bias (bool): If ``True``, adds a learnable bias to the output.
        reduction (int, optional): Reduction factor for attention mechanism. Default: 2.
        Returns:
        torch.Tensor: The output tensor after applying the CAM layer.
        """
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(torch.nn.Module):
    """A convolutional attention module layer. Inherits from `torch.nn.Module`."""
    def __init__(
        self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2
    ):
        """Initialize a CAMLayer for channel attention mechanism.
        Args:
        bn_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernels.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        bias (bool): If True, adds a learnable bias to the output.
        reduction (int): Reduction factor for the number of channels.
        Returns:
        None
        """
        super(CAMLayer, self).__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Applies a forward pass through a neural network layer, combining linear transformations and element-wise multiplication.
        Args:
        x (Tensor): Input tensor.
        Returns:
        Tensor: Output tensor after applying the transformation.
        """
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        """Applies segment pooling to input tensor `x` using average or max pooling based on `stype`.
        Args:
        x (Tensor): Input tensor.
        seg_len (int, optional): Segment length for pooling. Default is 100.
        stype (str, optional): Pooling type, either "avg" or "max". Default is "avg".
        Returns:
        Tensor: Pooled tensor with the same shape as `x` but reduced by `seg_len`.
        """
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(torch.nn.Module):
    """A dense time-delay neural network layer with batch normalization and optional ReLU activation."""
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        """Initialize a CAMDenseTDNNLayer instance.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_channels (int): Number of channels for batch normalization.
        kernel_size (int): Size of the convolutional kernel, must be odd.
        stride (int, optional): Stride of the convolution. Default is 1.
        dilation (int, optional): Dilation rate of the convolution. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is False.
        config_str (str, optional): Configuration string for batch normalization and activation. Default is "batchnorm-relu".
        memory_efficient (bool, optional): If True, uses a more memory-efficient implementation. Default is False.
        Raises:
        AssertionError: If kernel_size is even.
        """
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(
            kernel_size
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        """Conducts batch normalization followed by a nonlinear transformation.
        Args:
        x: Input tensor.
        Returns:
        Transformed output tensor through batch normalization and nonlinear layers.
        """
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        """Conducts forward pass through a block of layers.
        Args:
        x (Tensor): Input tensor.
        Returns:
        Tensor: Output tensor after applying batch normalization, non-linear transformation, and channel attention module.
        """
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    """A block for a CAM-Dense TDNN architecture, inheriting from torch.nn.ModuleList. Defines a series of convolutional layers with batch normalization and activation functions."""
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        """Initialize a CAMDenseTDNNBlock.
        Args:
        num_layers (int): Number of layers in the block.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn_channels (int): Number of channels for batch normalization.
        kernel_size (tuple or list of int): Kernel size(s) for each layer.
        stride (int, optional): Stride for each layer. Default is 1.
        dilation (int, optional): Dilation rate for each layer. Default is 1.
        bias (bool, optional): Whether to use bias in convolutional layers. Default is False.
        config_str (str, optional): Configuration string specifying batch normalization and activation functions. Default is "batchnorm-relu".
        memory_efficient (bool, optional): Whether to use memory-efficient operations. Default is False.
        Returns:
        None
        """
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        """Applies a series of TransitLayer modules to input tensor x by concatenating each layer's output along dimension 1.
        Args:
        x (Tensor): Input tensor of shape (batch_size, in_channels, length).
        Returns:
        Tensor: Output tensor after applying all layers, shape (batch_size, out_channels * num_layers, length).
        """
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(torch.nn.Module):
    """Class representing a dense layer with optional batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        """Initializes a dense layer with a linear transformation and optional non-linear activation.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is False.
        config_str (str, optional): Configuration string for the non-linear activation. Default is "batchnorm-relu".
        Returns:
        None
        """
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        """Applies a dense layer transformation to the input tensor.
        Args:
        x (torch.Tensor): The input tensor of shape (batch_size, in_channels).
        Returns:
        torch.Tensor: The output tensor after applying linear and nonlinear transformations.
        """
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(torch.nn.Module):
    """A dense layer that applies a linear transformation followed by a non-linear activation function."""
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        """Initializes a dense layer for 1D convolution and non-linear activation.
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is False.
        config_str (str, optional): Configuration string for the non-linear activation function. Default is "batchnorm-relu".
        Returns:
        torch.Tensor: The output tensor after applying linear transformation and non-linear activation.
        """
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        """Applies a linear transformation followed by a non-linear activation to the input tensor.
        Args:
        x (torch.Tensor): Input tensor.
        Returns:
        torch.Tensor: Transformed output tensor.
        """
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x

# @tables.register("model_classes", "CAMPPlus")
class CAMPPlus(torch.nn.Module):
    """A PyTorch module for CAMPPlus, a neural network architecture designed for efficient and effective feature extraction and processing in audio tasks."""
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        """Initialize a neural network layer.
        Args:
        feat_dim (int): Feature dimension.
        embedding_size (int): Embedding size.
        growth_rate (int): Growth rate for each dense block.
        bn_size (int): Bottleneck size.
        init_channels (int): Initial number of channels.
        config_str (str): Configuration string.
        memory_efficient (bool): Whether to use memory-efficient implementation.
        output_level (str): Output level.
        Returns:
        None
        """
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )
        else:
            assert (
                self.output_level == "frame"
            ), "`output_level` should be set to 'segment' or 'frame'. "

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Performs forward pass through the model to process input data.
        Args:
        x: Input tensor of shape (B,T,F) where B is batch size, T is sequence length, and F is feature dimension.
        Returns:
        Processed output tensor depending on output level.
        Inference method for processing a list of audio files.
        Args:
        audio_list: List of audio file paths.
        Returns:
        Model results after forward pass.
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    def inference(self, audio_list):
        """Processes a list of audio files to extract features and infer results.
        Args:
        audio_list (list): List of audio file paths.
        Returns:
        list: Inference results for each audio file.
        """
        speech, speech_lengths, speech_times = extract_feature(audio_list)
        results = self.forward(speech.to(torch.float32))
        return results
