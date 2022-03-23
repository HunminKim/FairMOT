import os

import torch
from torch import nn
# from .model_util import Branch


class Branch(nn.Module):
    def __init__(self, in_channel, out_channel, repeat=3):
        super(Branch, self).__init__()
        blocks = []
        temp = in_channel
        for i in range(repeat):
            if repeat -1 == i:
                temp = out_channel
            block = nn.Sequential(
                nn.Conv2d(in_channel, temp, kernel_size=3,
                                    stride=1, padding=1, bias=False),
                nn.BatchNorm2d(temp),
                nn.LeakyReLU()
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class FairMOT(nn.Module):
    def __init__(self, class_num, id_num, 
                backbone_module_path = 'core', 
                backbone_module_name = 'backbone', 
                backbone='dla_x_60_c', emb_size=256):
        super(FairMOT, self).__init__()
        
        
        build_model = self._import_backbone(backbone_module_path, 
                                            backbone_module_name, backbone)
        self.backbone = build_model()

        feature_map_ch = self.backbone.output_ch
        self.heatmap_conv = Branch(feature_map_ch, class_num)
        self.offset_conv = Branch(feature_map_ch, 2)
        self.wh_conv = Branch(feature_map_ch, 2)
        self.emb_conv = Branch(feature_map_ch, emb_size)
        self.id_class_conv = nn.Conv2d(emb_size, id_num, 1, 1, 0)

    def _import_backbone(self, backbone_module_path, backbone_module_name, backbone_name):
        try:
            core = __import__('.'.join([backbone_module_path, backbone_module_name]))
            backbone = getattr(core, backbone_module_name)
            backbone_list = []
            for item in dir(backbone):
                if 'dla' not in item:
                    continue
                backbone_list.append(item)
            build_model = getattr(backbone, backbone_name)
        except AttributeError as e:
            txt = 'No Attribute Backbone : "{}"\n select backbone this list\n{}'.format(backbone_name, backbone_list)
            raise AttributeError(txt)
        return build_model

    def forward(self, img):
        x = self.backbone(img)
        heatmap = self.heatmap_conv(x)
        offset = self.offset_conv(x)
        wh = self.wh_conv(x)
        emb = self.emb_conv(x)
        if self.training:
            id_class = self.id_class_conv(emb)
            return heatmap, offset, wh, id_class
        return heatmap, offset, wh, emb