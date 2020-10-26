import transformer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def train(n_batches, batch_size, data, lr, emb_size, seed):
    # Seed the network
    if (seed < 0):
        seed = random.randint(0, 1000000)
        print("Using seed: ", seed)
    else:
        torch.manual_seed(seed)
    # Load training data
    

