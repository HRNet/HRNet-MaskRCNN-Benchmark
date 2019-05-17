import torch
import torch.nn as nn
import torch.nn.functional as F


class HRFPN(nn.Module):

    def __init__(self, cfg):
        super(HRFPN, self).__init__()

        config = cfg.MODEL.NECK
        self.pooling_type = config.POOLING
        self.num_outs = config.NUM_OUTS
        self.in_channels = config.IN_CHANNELS
        self.out_channels = config.OUT_CHANNELS
        self.num_ins = len(self.in_channels)
        assert isinstance(self.in_channels, (list, tuple))

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(self.in_channels),
                      out_channels=self.out_channels,
                      kernel_size=1),
        )
        self.fpn_conv = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1
            ))
        if self.pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))
        return tuple(outputs)
