from ast import arg
import os
import importlib
from model.simsiam_ST import *
import torch
import torch.nn as nn
from GraphWaveNet.model import gwnet
from GraphWaveNet import util
from datasets.utils.utils import get_device
from datasets.utils.load_PEMS08D4 import get_adjacency_matrix

def get_backbone(args):
    # device = get_device()
    if args.backbone == 'GraphWaveNet':
        from GraphWaveNet import util
        from GraphWaveNet.GWN_AE import gwnet_encoder, gwnet_decoder
        device = torch.device(args.device)
        # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        adj_mx = get_adjacency_matrix(args.adj_path, args.num_nodes)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        if args.randomadj:
            adjinit = None
        else:
            adjinit = supports[0]
        
        if args.aptonly:
            supports = None

        # net = gwnet(device=device, num_nodes=args.num_nodes, dropout=args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid*8, end_channels=args.nhid*16)

        shared_encoder = gwnet_encoder(device, num_nodes=args.num_nodes, dropout=args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid*8)

        # shared_encoder = gwnet_encoder(device, num_nodes=args.num_nodes, dropout=args.dropout, supports=supports, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid*8)

        net_decoder = gwnet_decoder(out_dim=args.seq_length, skip_channels=args.nhid*8, end_channels=args.nhid*16)
    else:
        print('Backbone name error!!!')
    
    return shared_encoder, net_decoder