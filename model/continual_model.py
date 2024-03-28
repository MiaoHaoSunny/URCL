import torch
import torch.nn as nn
import numpy as np

from datasets.utils.utils import get_device
from GraphWaveNet.model import gwnet
from GraphWaveNet.GWN_AE import gwnet_encoder, gwnet_decoder, gwnet_AE
from model.backbone import get_backbone
from model.simsiam_ST import SimSiam_ST
from model.STBuffer import Buffer


class ContinualModel(nn.Module):
    def __init__(self, args):
        super(ContinualModel, self).__init__()
        self.shared_encoder, self.net_decoder = get_backbone(args=args)

        # self.ST_SimSiam = STSimSiam(self.shared_encoder)

        self.ST_SimSiam = SimSiam_ST(self.shared_encoder, args=args)

        # self.buffer = Buffer(buffer_size=args.buffer_size, device=torch.device(args.device))

        # self.gwnet_encoder = gwnet_encoder()
    
    def forward(self, x, aug1=None, aug2=None, lamda=None, buffer_x=None):
        if buffer_x is None:
            orginal_mixup = x
        else:
            if buffer_x.shape[0] != x.shape[0]:
                idx = x.shape[0]
                buffer_x = buffer_x[:idx]
            orginal_mixup = lamda * x + (1 - lamda) * buffer_x
        # aug_x1_mixed = lamda * aug_x1 + (1 - lamda) * buffer_x
        # aug_x2_mixed = lamda * aug_x2 + (1 - lamda) * buffer_x

        orginal_mixup_shared = self.shared_encoder(x)
        
        net_out = self.net_decoder(orginal_mixup_shared)

        p1, p2, z1, z2 = self.ST_SimSiam(orginal_mixup, aug1, aug2)
        # p1, p2, z1, z2 = None, None, None, None

        return net_out, p1, p2, z1, z2
