import os
import shutil
import numpy as np
import torch
import pdb

from copy import deepcopy


def drop_nodes(data, edge_index, ratio=0.1):
    time_step, node_num, _ = data.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num * ratio)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    # data = data[:, idx_nondrop]

    return data, adj, edge_index


def permute_edges(data, edge_index, ratio=0.1):
    time_step, node_num, _ = data.size()
    _, edge_num = edge_index.size()
    permute_num = int(edge_num*ratio)

    edge_index = edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))

    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data, edge_index


def add_edges(data, adj, ratio=0.2):
    # data_new = data
    adj_new = deepcopy(adj)
    batch_size, channels, num_nodes, time_step = data.size()
    data_new = data.transpose(0, 2).reshape(num_nodes, -1)
    similarity = torch.mm(data_new, data_new.t())

    # print(similarity.shape)

    _, indices = similarity.topk(int(num_nodes*ratio), dim=1, largest=True, sorted=True)
    # print(indices)
    # print(adj)
    for i in range(len(adj)):
        # print(indices[i].cpu().numpy())
        adj_new[0][i, indices[i]] = adj[0][i, indices[i]] + similarity[i, indices[i]]
        if len(adj_new) > 1:
            adj_new[1][i, indices[i]] = adj[1][i, indices[i]] + similarity[i, indices[i]]
    
    # print(adj[0].shape, adj[1].shape)

    return adj_new


def mask_nodes(data, ratio=0.1):
    time_step, node_num, feat_dim = data.size()
    mask_num = int(node_num * ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    data[:, idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data


def sub_graph(data, edge_index, ratio=0.1):
    time_step, node_num, _ = data.size()
    _, edge_num = edge_index.size()
    sub_num = int(node_num * ratio)

    edge_index = edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
    # data = data[:, idx_nondrop]
    edge_index = edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    return data, adj, edge_index


def time_inverse(data):

    data = torch.flip(data, dims=[-1])

    return data


def get_batch_augment_data(original_x, original_adj, type=None):
    if type == 'edge':
        aug_x = original_x
        aug_adj = add_edges(original_x, original_adj)
    elif type == 'time':
        aug_x = time_inverse(original_x)
        aug_adj = original_adj
    return aug_x, aug_adj