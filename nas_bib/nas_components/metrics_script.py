import copy
import time
import torch
import torch.nn as nn
import logging
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop the training when the monitored quantity has stopped improving.
    """

    def __init__(self, patience=5, monitor='val_loss', mode='min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, val_loss):
        """
        Call method to update early stopping criteria.
        """
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def stop(self):
        """
        Check if early stopping criteria is met.
        """
        return self.early_stop


# Latency

def rm_bn_from_net(net):
    """
    Remove BatchNorm layers from the network.
    """
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


def get_net_device(net):
    """
    Get the device of the network.
    """
    return net.parameters().__next__().device


def measure_net_latency(net, l_type='cpu', fast=True, input_shape=(3, 224, 224), clean=False):
    """
    Measure the latency of the network.
    """
    if isinstance(net, nn.DataParallel):
        net = net.module

    # Remove BN from the graph
    rm_bn_from_net(net)

    if 'gpu' in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == 'cpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device('cpu'):
            if not clean:
                print('Move net to CPU for measuring CPU latency')
            net = copy.deepcopy(net).cpu()
    elif l_type == 'gpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError

    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {'warmup': [], 'sample': []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency['warmup'].append(used_time)
            #TO DO
            #if not clean:
                #print('Warmup %d: %.3f' % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency['sample'].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


"""
from https://github.com/abdelfattah-lab/nasflat_latency
"""


def calculate_model_size(model):
    """
    Calculate the size of the model.
    """
    # Calculate the total size of the model parameters
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
