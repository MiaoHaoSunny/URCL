from this import s
from turtle import back, forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphWaveNet.GWN_AE import gwnet_encoder, gwnet_decoder


class projection_MLP_flat(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=256):
        super(projection_MLP_flat, self).__init__()

        # layer1
        self.linear_layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn_layer1 = nn.BatchNorm2d(hidden_dim)
        self.relu_layer1 = nn.ReLU(inplace=True)
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )

        # layer2
        self.linear_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_layer2 = nn.BatchNorm2d(hidden_dim)
        self.relu_layer2 = nn.ReLU(inplace=True)
        # self.layer2 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )

        # layer3
        self.linear_layer3 = nn.Linear(hidden_dim, out_dim)
        self.bn_layer3 = nn.BatchNorm2d(out_dim)
        # self.layer3 = nn.Sequential(
        #     nn.Linear(hidden_dim, out_dim),
        #     nn.BatchNorm2d(hidden_dim)
        # )
        self.num_layers = 3
    
    def set_layers(self, num_layers):
        self.num_layers = num_layers
    
    def forward(self, x):
        if self.num_layers == 3:
            # x = self.layer1(x)
            # x = self.layer2(x)
            # x = self.layer3(x)
            x = self.linear_layer1(x)
            x = x.transpose(1, 3)
            x = self.bn_layer1(x)
            x = self.relu_layer1(x)
            x = x.transpose(1, 3)

            x = self.linear_layer2(x)
            x = x.transpose(1, 3)
            x = self.bn_layer2(x)
            x = self.relu_layer2(x)
            x = x.transpose(1, 3)

            x = self.linear_layer3(x)
            # print(x.shape)
            x = x.transpose(1, 3)
            # print('.', x.shape)
            x = self.bn_layer3(x)
            x = x.transpose(1, 3)
        elif self.num_layers == 2:
            # x = self.layer1(x)
            # x = self.layer3(x)
            x = self.linear_layer1(x)
            x = x.transpose(1, 3)
            x = self.bn_layer1(x)
            x = self.relu_layer1(x)
            x = x.transpose(1, 3)

            x = self.linear_layer3(x)
            x = x.transpose(1, 3)
            x = self.bn_layer3(x)
        else:
            raise Exception
        return x


# class projection_MLP(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=128, out_dim=256):
#         super(projection_MLP, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Linear(hidden_dim, out_dim),
#             nn.BatchNorm2d(hidden_dim)
#         )
#         self.num_layers = 3
    
#     def set_layers(self, num_layers):
#         self.num_layers = num_layers
    
#     def forward(self, x):
#         if self.num_layers == 3:
#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#         elif self.num_layers == 2:
#             x = self.layer1(x)
#             x = self.layer3(x)
#         else:
#             raise Exception
#         return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=256):
        super(prediction_MLP, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU(inplace=True)
        # )
        self.linear_layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn_layer1 = nn.BatchNorm2d(hidden_dim)
        self.relu_layer1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        # x = self.layer1(x)
        x = self.linear_layer1(x)
        x = x.transpose(1, 3)
        x = self.bn_layer1(x)
        x = self.relu_layer1(x)

        x = x.transpose(1, 3)

        x = self.layer2(x)
        # x = x.transpose(1, 3)
        return x


# class STSimSiam(nn.Module):
#     def __init__(self, backbone):
#         super(STSimSiam).__init__()
#         self.backbone = backbone()
#         self.projector = projection_MLP(backbone.skip_channels)

#         self.STEncoder = nn.Sequential(
#             self.backbone,
#             self.projector
#         )

#         self.predictor = prediction_MLP()
    
#     def forward(self, x1, x2):
#         f, h = self.STEncoder, self.predictor

#         z1, z2 = f(x1), f(x2)

#         p1, p2 = h(z1), h(z2)
#         return p1, p2, z1.detach(), z2.detach()


class SimSiam_ST(nn.Module):
    def __init__(self, backbone, args):
        super(SimSiam_ST, self).__init__()
        self.backbone = backbone
        self.projector = projection_MLP_flat(args.nhid*8)

        # self.STEncoder = nn.Sequential(
        #     self.backbone,
        #     self.projector
        # )

        self.predictor = prediction_MLP()
    
    def forward(self, x, aug1=None, aug2=None):
        # f, h = self.STEncoder, self.predictor
        # print('x shape', x.shape)
        x1 = self.backbone(x, aug=aug1)
        # print(x1.shape)
        # x1 = x1.permute(1, 2, 0)
        # x1 = x1.transpose
        # x1 = x1.view(64, -1, 207, 2)
        # print('x1 shape', x1.shape)
        
        x1 = x1.transpose(1, 3)
        # print('x1 shape: {}'.format(x1.shape))
        z1 = self.projector(x1)

        x2 = self.backbone(x, aug=aug2)

        # x2 = x2.permute(1, 2, 0)
        # x2 = x2.view(64, -1, 207, 2)

        x2 = x2.transpose(1, 3)
        z2 = self.projector(x2)

        h = self.predictor

        # z1, z2 = f(x, aug1), f(x, aug2)

        p1, p2 = h(z1), h(z2)
        return p1, p2, z1.detach(), z2.detach()