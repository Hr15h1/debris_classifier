import torch
from torch import nn
import torchvision
from .position_encoding import build_position_encoding
from util.misc import NestedTensor
import torch.nn.functional as F
from typing import Dict, List

class SqueezeNetBackbone(nn.Module):
    def __init__(self, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # Get pretrained SqueezeNet and remove classifier
        backbone = torchvision.models.squeezenet1_0(pretrained=True)
        self.body = backbone.features  # Only feature extractor part
        self.num_channels = 512  # SqueezeNet's final feature map channels

        # Optionally freeze backbone
        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        out = {"0": NestedTensor(xs, mask)}
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if getattr(args, "backbone", None) == "squeezenet":
        backbone = SqueezeNetBackbone(train_backbone, return_interm_layers)
    else:
        # default ResNet backbone (existing code)
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model