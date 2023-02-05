from turtle import forward
import torch
from torch import nn
from model.SIREN import SirenNet
from model.MLP import MLPNet
from model.ProbNet import PSNet

class Mixnet(nn.Module):
    def __init__(self, base_conf, fine_conf, inter_conf):
        super().__init__()
        self.Siren_conf = fine_conf
        self.MLP_conf = base_conf
        self.Prob_conf = inter_conf

        self.fine = SirenNet(**self.Siren_conf)
        self.base = MLPNet(**self.MLP_conf)
        self.inter = PSNet(**self.Prob_conf)

        print(self)

    def forward(self, x):
        y1 = self.base(x)
        y2 = self.fine(x)
        p = self.inter(x)
        #p = y2 / (y2 - y1)

        y = p * y1 + (1-p) * y2

        return {'base': y1,
                'fine': y2,
                'inter': p,
                'mix': y}

class Mixnet_General(nn.Module):
    def __init__(self, base_conf, fine_conf, inter_conf):
        super().__init__()
        self.fine_conf = fine_conf
        self.base_conf = base_conf
        self.P_conf = inter_conf

        self.fine = SirenNet(**self.fine_conf)
        self.base = SirenNet(**self.base_conf)
        self.inter = PSNet(**self.P_conf)

        print(self)

    def forward(self, x):
        y1 = self.base(x)
        y2 = self.fine(x)
        p = self.inter(x)
        #p = y2 / (y2 - y1)

        y = p * y1 + (1-p) * y2

        return {'base': y1,
                'fine': y2,
                'inter': p,
                'mix': y}