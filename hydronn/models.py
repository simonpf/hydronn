"""
==============
hydronn.models
==============

Defines the neural-network models that are used for the Hydronn retrieval.
"""
import torch
from torch import nn

from quantnn.models.pytorch.xception import (DownsamplingBlock,
                                             UpsamplingBlock,
                                             XceptionBlock,
                                             SymmetricPadding)


class MLPHead(nn.Module):
    """
    MLP-type head for convolutional network.
    """
    def __init__(self,
                 n_inputs,
                 n_hidden,
                 n_outputs,
                 n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(n_inputs, n_hidden, 1),
                nn.GroupNorm(n_hidden, n_hidden),
                nn.GELU()
            ))
            n_inputs = n_hidden
        self.layers.append(nn.Sequential(
            nn.Conv2d(n_hidden, n_outputs, 1),
        ))

    def forward(self, x):
        "Propagate input through head."
        for l in self.layers[:-1]:
            y = l(x)
            n = min(x.shape[1], y.shape[1])
            y[:, :n] += x[:, :n]
            x = y
        return self.layers[-1](y)


class Hydronn2(nn.Module):
    """
    Feature pyramid network (FPN) with 5 stages based on xception
    architecture.
    """
    def __init__(self,
                 n_outputs,
                 n_blocks,
                 n_features_body,
                 n_layers_head,
                 n_features_head):
        """
        Args:
            n_outputs: The number of output channels,
            n_blocks: The number of blocks in each stage of the encoder.
            n_features_body: The number of features/channels in the network
                body.
            n_layers_head: The number of layers in each network head.
            n_features_head: The number of features in each layer of each head.
            ancillary: Whether or not to make use of ancillary data.
            target: List of target variables.
        """
        super().__init__()
        self.n_outputs = n_outputs

        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * 7

        n_features_hi_res = n_features_body // 16
        n_features_med_res = n_features_body // 4

        self.hi_res_in = nn.Sequential(
            XceptionBlock(
                1,
                n_features_hi_res,
                downsample=True
            ),
            *([XceptionBlock(
                n_features_hi_res,
                n_features_hi_res
            )] * (n_blocks[0] - 1))
        )

        self.med_res_in = nn.Sequential(
            XceptionBlock(
                n_features_hi_res + 3,
                n_features_med_res,
                downsample=True
            ),
            *([XceptionBlock(
                n_features_med_res,
                n_features_med_res,
            )] * (n_blocks[1] - 1))
        )

        self.low_res_in = nn.Sequential(
            XceptionBlock(n_features_med_res + 12, n_features_body, downsample=False),
        )

        self.down_block_2 = DownsamplingBlock(n_features_body, n_blocks[0])
        self.down_block_4 = DownsamplingBlock(n_features_body, n_blocks[1])
        self.down_block_8 = DownsamplingBlock(n_features_body, n_blocks[2])
        self.down_block_16 = DownsamplingBlock(n_features_body, n_blocks[3])
        self.down_block_32 = DownsamplingBlock(n_features_body, n_blocks[4])

        self.up_block_16 = UpsamplingBlock(n_features_body)
        self.up_block_8 = UpsamplingBlock(n_features_body)
        self.up_block_4 = UpsamplingBlock(n_features_body)
        self.up_block_2 = UpsamplingBlock(n_features_body)
        self.up_block = UpsamplingBlock(n_features_body)

        n_inputs = n_features_body + n_features_med_res + 12
        self.head = MLPHead(n_inputs,
                            n_features_head,
                            n_outputs,
                            n_layers_head)

    def forward(self, x):
        """
        Propagate input through block.
        """
        low_res, med_res, hi_res = x

        x_hi = self.hi_res_in(hi_res)

        x_med = torch.cat([x_hi, med_res], 1)
        x_med = self.med_res_in(x_med)

        x_low = torch.cat([x_med, low_res], 1)
        x_low = self.low_res_in(x_low)

        x_2 = self.down_block_2(x_low)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_16(x_32, x_16)
        x_8_u = self.up_block_8(x_16_u, x_8)
        x_4_u = self.up_block_4(x_8_u, x_4)
        x_2_u = self.up_block_2(x_4_u, x_2)
        x_u = self.up_block(x_2_u, x_low)

        x = torch.cat([x_u, x_med, x[0]], 1)
        return self.head(x)


Hydronn = Hydronn2


class Hydronn4(nn.Module):
    """
    Feature pyramid network (FPN) with 5 stages based on xception
    architecture.
    """
    def __init__(self,
                 n_outputs,
                 n_blocks,
                 n_features_body,
                 n_layers_head,
                 n_features_head):
        """
        Args:
            n_outputs: The number of output channels,
            n_blocks: The number of blocks in each stage of the encoder.
            n_features_body: The number of features/channels in the network
                body.
            n_layers_head: The number of layers in each network head.
            n_features_head: The number of features in each layer of each head.
            ancillary: Whether or not to make use of ancillary data.
            target: List of target variables.
        """
        super().__init__()
        self.n_outputs = n_outputs

        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * 7

        n_features_hi_res = n_features_body // 16
        n_features_med_res = n_features_body // 4

        self.hi_res_in = nn.AvgPool2d(8, 8)
        self.med_res_in = nn.AvgPool2d(4, 4)
        self.low_res_in = nn.AvgPool2d(2, 2)

        self.block_in = XceptionBlock(
            16, n_features_body, downsample=False
        )

        self.down_block_2 = DownsamplingBlock(n_features_body, n_blocks[0])
        self.down_block_4 = DownsamplingBlock(n_features_body, n_blocks[1])
        self.down_block_8 = DownsamplingBlock(n_features_body, n_blocks[2])
        self.down_block_16 = DownsamplingBlock(n_features_body, n_blocks[3])
        self.down_block_32 = DownsamplingBlock(n_features_body, n_blocks[4])

        self.up_block_16 = UpsamplingBlock(n_features_body)
        self.up_block_8 = UpsamplingBlock(n_features_body)
        self.up_block_4 = UpsamplingBlock(n_features_body)
        self.up_block_2 = UpsamplingBlock(n_features_body)
        self.up_block = UpsamplingBlock(n_features_body)

        self.head = MLPHead(n_features_body + 16,
                            n_features_head,
                            n_outputs,
                            n_layers_head)

    def forward(self, x):
        """
        Propagate input through block.
        """
        low_res, med_res, hi_res = x

        x_hi = self.hi_res_in(hi_res)
        x_med = self.med_res_in(med_res)
        x_low = self.low_res_in(low_res)

        x_in = torch.cat([x_low, x_med, x_hi], axis=1)
        x = self.block_in(x_in)

        x_2 = self.down_block_2(x)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_16(x_32, x_16)
        x_8_u = self.up_block_8(x_16_u, x_8)
        x_4_u = self.up_block_4(x_8_u, x_4)
        x_2_u = self.up_block_2(x_4_u, x_2)
        x_u = self.up_block(x_2_u, x)

        return self.head(torch.cat([x_u, x_in], axis=1))


class Hydronn4IR(nn.Module):
    """
    Feature pyramid network (FPN) with 5 stages based on xception
    architecture.
    """
    def __init__(self,
                 n_outputs,
                 n_blocks,
                 n_features_body,
                 n_layers_head,
                 n_features_head):
        """
        Args:
            n_outputs: The number of output channels,
            n_blocks: The number of blocks in each stage of the encoder.
            n_features_body: The number of features/channels in the network
                body.
            n_layers_head: The number of layers in each network head.
            n_features_head: The number of features in each layer of each head.
            ancillary: Whether or not to make use of ancillary data.
            target: List of target variables.
        """
        super().__init__()
        self.n_outputs = n_outputs

        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * 7

        self.avg_in = nn.AvgPool2d(2, 2)
        self.block_in = XceptionBlock(
            1, n_features_body, downsample=False
        )

        self.down_block_2 = DownsamplingBlock(n_features_body, n_blocks[0])
        self.down_block_4 = DownsamplingBlock(n_features_body, n_blocks[1])
        self.down_block_8 = DownsamplingBlock(n_features_body, n_blocks[2])
        self.down_block_16 = DownsamplingBlock(n_features_body, n_blocks[3])
        self.down_block_32 = DownsamplingBlock(n_features_body, n_blocks[4])

        self.up_block_16 = UpsamplingBlock(n_features_body)
        self.up_block_8 = UpsamplingBlock(n_features_body)
        self.up_block_4 = UpsamplingBlock(n_features_body)
        self.up_block_2 = UpsamplingBlock(n_features_body)
        self.up_block = UpsamplingBlock(n_features_body)

        self.head = MLPHead(n_features_body + 1,
                            n_features_head,
                            n_outputs,
                            n_layers_head)

    def forward(self, x):
        """
        Propagate input through block.
        """
        low_res, med_res, hi_res = x

        x_in = self.avg_in(low_res[:, [-4]])
        x = self.block_in(x_in)

        x_2 = self.down_block_2(x)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_16(x_32, x_16)
        x_8_u = self.up_block_8(x_16_u, x_8)
        x_4_u = self.up_block_4(x_8_u, x_4)
        x_2_u = self.up_block_2(x_4_u, x_2)
        x_u = self.up_block(x_2_u, x)

        return self.head(torch.cat([x_u, x_in], axis=1))
