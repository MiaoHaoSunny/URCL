from tkinter.messagebox import NO
# import numpy as np
import copy

import torch
import torch.nn.functional as F


def overwrite_grad(pp, new_grad, grad_dims):
    cnt = 0
    for param in pp():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt+1])
        # print(beg, en, param.data.size(), new_grad, grad_dims)
        # this_grad = new_grad[beg:en]
        this_grad = new_grad[beg:en].contiguous().view(param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1


def get_grad_vector(pp, grad_dims, args):
    grads = torch.Tensor(sum(grad_dims))
    if torch.cuda.is_available():
        device = torch.device(args.device)
        grads = grads.to(device)
    
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt+1])
            grads[beg:en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    new_net = copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters, grad_vector, grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                # print(lr)
                param.data = param.data - lr * param.grad.data
    return new_net


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def contrastive_loss(p, z):
    cos_sim = F.cosine_similarity(p, z, dim=-1)
    exp_cos_sim = torch.exp(cos_sim)
    log_exp_sim = torch.log(exp_cos_sim)
    loss = cos_sim - log_exp_sim
    return -loss.mean()


def simple_contrastive_loss(p, z):
    # print('p shape: {}, z shape: {}'.format(p.shape, z.shape))
    return -F.cosine_similarity(p, z, dim=-1).mean()