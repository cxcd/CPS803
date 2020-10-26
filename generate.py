import tf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import tqdm

def here(subpath=None):
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))


def get_data(data_path):
    """ Incredibly naive approach to making data sets,
        only for proof of concept usage. """
    data = np.loadtxt(data_path, dtype=chr)
    train, half_data2 = np.split(data, 2)
    valid, test = np.split(half_data2, 2)
    return train, valid, test

def train(n_heads, depth, seq_length, n_tokens, emb_size, n_batches, batch_size, data, lr, warmup, seed):
    # Seed the network
    if (seed < 0):
        seed = random.randint(0, 1000000)
        print("Using seed: ", seed)
    else:
        torch.manual_seed(seed)
    # Load training data
    data_train, data_valid, data_test = get_data(here("alice.txt"))
    # Create the model
    model = tf.GenTransformer(
                emb=emb_size, 
                heads=n_heads,
                depth=depth,
                seq_length=seq_length,
                n_tokens=n_tokens
            )
    # Optimizer
    opt = torch.optim.SGD(model.parameters(), lr)
    # Train over batches of random sequences
    for i in tqdm.trange(n_batches): # tqdm is a nice progress bar
        # Warming up learning rate by linearly increasing to the provided learning rate
        if lr > 0 and i < warmup:
            lr = max((lr / warmup) * i, 1e-10)
            opt.lr = lr
        # Prevent gradient accumulation
        opt.zero_grad()
        # Sample batch of random subsequences

