import torch
from torch import nn


class UPDLAUnit(nn.Module):
    def __init__(self, repeat, in_channel, out_channel):
        super(UPDLAUnit, self).__init__()
        convs = []
        for _ in range(repeat):
            convs.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, 
                                    stride=2, padding=1, output_padding=1)
            )
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        x = self.convs(x)
        return x

        
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x, skip=None):
        if skip is None:
            skip = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += skip
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        bottle_channel = out_channel // 2
        self.conv1 = nn.Conv2d(in_channel, bottle_channel,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_channel)
        self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, kernel_size=3,
                               stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottle_channel)
        self.conv3 = nn.Conv2d(bottle_channel, out_channel,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU()

    def forward(self, x, skip=None):
        if skip is None:
            skip = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += skip
        out = self.activation(out)
        return out


class BottleneckX(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU()

    def forward(self, x, skip=None):
        if skip is None:
            skip = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += skip
        out = self.activation(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()
        self.skip = skip

    def forward(self, *x):
        children = list(x)
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.skip:
            x += children[0]
        x = self.activation(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_in_channels=0, root_kernel_size=1, root_skip_connection=False):
        super(Tree, self).__init__()
        self.downsample = None
        self.channel_control_conv = None
        self.level_root = level_root
        self.levels = levels
        
        if root_in_channels == 0:
            root_in_channels = 2 * out_channels

        if level_root:
            root_in_channels += in_channels

        if levels == 1:
            self.root = Root(root_in_channels, out_channels, root_kernel_size, root_skip_connection)
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_in_channels=0,
                              root_kernel_size=root_kernel_size, root_skip_connection=root_skip_connection)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_in_channels=root_in_channels + out_channels,
                              root_kernel_size=root_kernel_size, root_skip_connection=root_skip_connection)
            
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        if in_channels != out_channels:
            self.channel_control_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, skip=None, children=None):
        if children is None:
             children = []

        if isinstance(self.downsample, nn.MaxPool2d):
            bottom = self.downsample(x)
        else: 
            bottom = x.clone()

        if isinstance(self.channel_control_conv, nn.Sequential):
            skip = self.channel_control_conv(bottom)
        else:
            skip = bottom.clone()

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, skip)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class UPDLA(nn.Module):
    def __init__(self, channels):
        super(UPDLA, self).__init__()
        self.channels = channels
        self.channels.reverse()
        self.repeat_full = len(self.channels) - 1
        up_unit = []
        for i in range(len(self.channels) - 1):
            in_channel = self.channels[i]
            out_channel = self.channels[i+1]
            repeat = self.repeat_full - i
            up_unit.append(UPDLAUnit(repeat, in_channel, out_channel))
        self.up_unit = nn.ModuleList(up_unit)

    def forward(self, xs):
        xs.reverse()
        x_temp = None
        for i, unit in enumerate(self.up_unit):
            if x_temp is not None:
                x = x + x_temp
            x = unit(xs[i])
            x_temp = x.clone()
        x = x + x_temp
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, block=BasicBlock, root_skip_connection=False):
        super(DLA, self).__init__()
        self.input_block = self._get_input_block(channels[0])

        stage_0 = self._get_simple_conv_block(
            channels[0], channels[0])
        stage_1 = self._get_simple_conv_block(
            channels[0], channels[1], stride=2)

        stage_2 = Tree(levels[0], block, channels[1], channels[2], 2,
                           level_root=False, root_skip_connection=root_skip_connection)
        stage_3 = Tree(levels[1], block, channels[2], channels[3], 2,
                           level_root=True, root_skip_connection=root_skip_connection)
        stage_4 = Tree(levels[2], block, channels[3], channels[4], 2,
                           level_root=True, root_skip_connection=root_skip_connection)
        stage_5 = Tree(levels[3], block, channels[4], channels[5], 2,
                           level_root=True, root_skip_connection=root_skip_connection)

        stages = [stage_0, stage_1, stage_2, stage_3, stage_4, stage_5]
        self.stages = nn.ModuleList(stages)
        self.last = UPDLA(channels[-4:])
        self.output_ch = channels[-4]

    def _get_input_block(self, out_channel):
            input_block = nn.Sequential(
                nn.Conv2d(3, out_channel, kernel_size=7, stride=1,
                            padding=3, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU()
            )
            return input_block

    def _get_simple_conv_block(self, in_channel, out_channel, stride=1):
        modules = []
        modules = [
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                        stride=stride,
                        padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
            ]
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.input_block(x)
        for stage in self.stages:
            x = stage(x)
            y.append(x)
        x = self.last(y)
        return x



def dla_34():  # DLA-34
    model = DLA([1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                block=BasicBlock, root_skip_connection=False)
    return model


def dla_46_c():  # DLA-46-C
    model = DLA([1, 2, 2, 1], [16, 32, 64, 64, 128, 256],
                block=Bottleneck, root_skip_connection=False)
    return model


def dla_x_46_c():  # DLA-X-46-C
    model = DLA([1, 2, 2, 1], [16, 32, 64, 64, 128, 256],
                block=BottleneckX, root_skip_connection=False)
    return model


def dla_x_60_c():  # DLA-X-60-C
    model = DLA([1, 2, 3, 1], [16, 32, 64, 64, 128, 256],
                block=BottleneckX, root_skip_connection=False)
    return model


def dla_60():  # DLA-60
    model = DLA([1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, root_skip_connection=False)
    return model


def dla_x_60():  # DLA-X-60
    model = DLA([1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, root_skip_connection=False)
    return model


def dla_102():  # DLA-102
    model = DLA([1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, root_skip_connection=True)
    return model


def dla_x_102():  # DLA-X-102
    model = DLA([1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, root_skip_connection=True)
    return model


def dla_169():  # DLA-169
    model = DLA([2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, root_skip_connection=True)
    return model