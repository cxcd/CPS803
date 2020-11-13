from torch import uint8
import tf
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist
import random
import numpy as np
import tqdm, math
import util

def split_padded(a,n):
    """
    Pad splits to make them even
    """
    padding = (-len(a))%n
    return np.split(np.concatenate((a,np.zeros(padding))),n)

def get_data(data_path):
    """ Incredibly naive approach to making data sets,
        only for proof of concept usage. """
    with open(data_path, encoding="utf8") as f:
        content = f.read()
    # print (list(content))
    data = np.array(list(content))
    data = data.view(np.uint8)
    # data = np.loadtxt(data_path, dtype=chr)
    train, half_data2 = split_padded(data, 2)
    valid, test = split_padded(half_data2, 2)
    return torch.from_numpy(train), torch.from_numpy(valid), torch.from_numpy(test)

def sample(lnprobs, temperature=0.0):
    """
    Sample an element from the model. Temp of 1 follows the distribution, 0 follows the maximum probability.
    """
    # Return max value if there is no temperature
    if temperature == 0.0:
        return lnprobs.argmax()
    # Otherwise return the value according to the temperature
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()


def gen(model, input, size=100, temp=0.5):
    """
    Generate data from the model
    """
    with torch.no_grad():
        # Use cuda
        if torch.cuda.is_available():
            input = input.cuda()
        #
        input = Variable(input)
        #  Print the input
        print('[', end='', flush=True)
        for i in input:
            # TODO change this to appending to a midi array
            print(str(chr(i)), end='', flush=True)
        print(']', end='', flush=True)
        # Get generated data
        for i in range(size):
            output = model(input[None, :])
            c = sample(output[0, -1, :], temp)
            # print(" SAMPLE: ", c)
            print(str(chr(max(32, c))), end='', flush=True)
            # Make next prediction informed by this prediction
            input = torch.cat([input[1:], c[None]], dim=0)
        #
        print()


def train(n_heads=8, depth=4, seq_length=32, n_tokens=256, emb_size=128, n_batches=500, batch_size=64, test_every=50, lr=0.0001, warmup=100, seed=-1, data=None, output_path="genmodel.pt"):
    """
    Train the model and save it to output_path
    """
    # Seed the network
    if (seed < 0):
        seed = random.randint(0, 1000000)
        print("Using seed: ", seed)
    else:
        torch.manual_seed(seed)

    # Load training data
    if data == None: return
    data_train, data_valid, data_test = get_data(util.here(data))
    # Create the model
    model = tf.GenTransformer(
                emb=emb_size, 
                n_heads=n_heads,
                depth=depth,
                seq_length=seq_length,
                n_tokens=n_tokens
            )
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr)
    # Train over batches of random sequences
    for i in tqdm.trange(n_batches - 1): # tqdm is a nice progress bar
        # Warming up learning rate by linearly increasing to the provided learning rate
        if lr > 0 and i < warmup:
            lr = max((lr / warmup) * i, 1e-10)
            opt.lr = lr
        # Prevent gradient accumulation
        opt.zero_grad()
        # Sample batch of random subsequences
        starts = torch.randint(size=(batch_size, ), low=0, high=data_train.size(0) - seq_length - 1)
        seqs_source = [data_train[start : start + seq_length] for start in starts]
        # The target is the same as the source sequence except one character ahead
        seqs_target = [data_train[start + 1 : start + seq_length + 1] for start in starts]
        source = torch.cat([s[None, :] for s in seqs_source], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
        # Get cuda
        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()
        source, target = Variable(source), Variable(target)
        # Initialize the output
        output = model(source)
        # Get the loss
        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        loss.backward()
        # Clip the gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        # Perform optimization step
        opt.step()
        # Validate every so often, compute compression then generate
        if i != 0 and (i % test_every == 0 or i == n_batches - 1):
            # TODO sort of arbitrary, make this rigorous
            upto = data_test.size(0) if i == n_batches - 1 else 32
            data_sub = data_test[:upto]
            # 
            with torch.no_grad():
                bits, tot = 0.0, 0
                # When this buffer is full we run it through the model
                batch = []
                for current in range (data_sub.size(0)):
                    fr = max(0, current - seq_length)
                    to = current + 1
                    context = data_sub[fr:to].to(torch.long)
                    # If the data doesnt fit the sequence length pad it
                    if context.size(0) < seq_length + 1:
                        pad = torch.zeros(size=(seq_length + 1 - context.size(0), ), dtype=torch.long)
                        context = torch.cat([pad, context], dim=0)
                        assert context.size(0) == seq_length + 1
                    # Get cuda
                    if torch.cuda.is_available():
                        context = context.cuda()
                    # Fill the batch
                    batch.append(context[None, :])
                    # Check if the batch is full
                    if len(batch) == batch_size or current == data_sub.size(0) - 1:
                        # Run through model
                        b = len(batch)
                        all = torch.cat(batch, dim=0)
                        source = all[:, :-1] # Input
                        target = all[:, -1] # Target values
                        #
                        output = model(source)
                        # Get probabilities and convert to bits
                        lnprobs = output[torch.arange(b, device=util.device()), -1, target]
                        log2probs = lnprobs * math.log2(math.e)
                        # For logging
                        bits += log2probs.sum()
                        # Empty batch buffer
                        batch = []
                # Print validation performance
                bits_per_byte = abs(bits / data_sub.size(0))
                print(f' epoch {i}: {bits_per_byte:.4} bits per byte')
                print(" Loss:", loss.item())
                # Monitor progress by generating data based on the test data
                seedfr = random.randint(0, data_test.size(0) - seq_length)
                input = data_test[seedfr:seedfr + seq_length].to(torch.long)
                gen(model, input)

    # Save the model when we're done training it
    util.save_model(model, output_path)
    # 
    print("Finished training. Model saved to", output_path)