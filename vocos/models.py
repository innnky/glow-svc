from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.heads import ISTFTHead
from vocos.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, bandwidth_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x

config = {
    'backbone': {
        'class_path': 'vocos.models.VocosBackbone',
        'init_args': {
            'input_channels': 100,
            'dim': 512,
            'intermediate_dim': 1536,
            'num_layers': 8
        }
    },
    'head': {
        'class_path': 'vocos.heads.ISTFTHead',
        'init_args': {
            'dim': 512,
            'n_fft': 1024,
            'hop_length': 256,
            'padding': 'same'
        }
    }
}

class Generator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = VocosBackbone(
            config['backbone']["init_args"]['input_channels'],
            config['backbone']["init_args"]['dim'],
            config['backbone']["init_args"]['intermediate_dim'],
            config['backbone']["init_args"]['num_layers']
        )
        self.head = ISTFTHead(
            config['head']["init_args"]['dim'],
            config['head']["init_args"]['n_fft'],
            config['head']["init_args"]['hop_length'],
            config['head']["init_args"]['padding']
        )

    def forward(self, features, **kwargs):
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output


class Discriminators(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
    def forward(self, y, y_hat):
        real_score_mp, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
            y=y, y_hat=y_hat
        )
        real_score_mrd, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
            y=y, y_hat=y_hat
        )

        return real_score_mp, gen_score_mp, fmap_rs_mp, fmap_gs_mp, real_score_mrd, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd