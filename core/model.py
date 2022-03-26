
import math
import torch
from torch import nn


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

    def forward(self, x, activation=True):
        for block in self.blocks:
            x = block(x)
        if activation:
            x = nn.functional.softmax(x, 1)
        return x


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            num_classes: size of each output sample
            m: margin
    """
    
    def __init__(self, id_num, emb_size=256, scale_factor=64.0, margin=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.emb_size = emb_size
        self.id_num = id_num
        self.scale_factor = scale_factor
        self.margin = margin
        # self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))#.to(device)
        self.cos_conv = nn.Conv2d(emb_size, id_num + 1, 1, 1, 0, bias=False)
        nn.init.xavier_uniform_(self.cos_conv.weight)
        self.cos_conv = nn.utils.weight_norm(self.cos_conv)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = self.cos_conv(input)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.argmax(label, 1, keepdim=True)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale_factor
        output = nn.functional.softmax(output, 1)
        return output


class FairMOT(nn.Module):
    def __init__(self, class_num, id_num, 
                backbone_module_path = 'core', 
                backbone_module_name = 'backbone', 
                backbone='dla_34', emb_size=256):
        super(FairMOT, self).__init__()
        
        
        build_model = self._import_backbone(backbone_module_path, 
                                            backbone_module_name, backbone)
        self.backbone = build_model()

        feature_map_ch = self.backbone.output_ch
        self.heatmap_conv = Branch(feature_map_ch, class_num + 1)
        self.offset_conv = Branch(feature_map_ch, 2)
        self.wh_conv = Branch(feature_map_ch, 2)
        self.emb_conv = Branch(feature_map_ch, emb_size)
        self.arcmargin = ArcMarginProduct(id_num, emb_size)

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
        except AttributeError:
            txt = 'No Attribute Backbone : "{}"\n select backbone this list\n{}'.format(backbone_name, backbone_list)
            raise AttributeError(txt)
        return build_model

    def forward(self, img, label=None):
        x = self.backbone(img)
        heatmap = self.heatmap_conv(x)
        offset = self.offset_conv(x, activation=False)
        wh = self.wh_conv(x, activation=False)
        emb = self.emb_conv(x, activation=False)
        emb =  nn.functional.normalize(emb, p=2.0, eps=1e-12)
        if self.training:
            id_class = self.arcmargin(emb, label)
            return heatmap, offset, wh, id_class
        return heatmap, offset, wh, emb