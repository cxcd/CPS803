# ML
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Util
import time

# Hold information to inject about the relative or absolute position of the tokens in the sequence
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    # Initialize
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()
    
    # Create empty square attention mask
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # Initialize weights
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # If there is no mask, make a mask
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask
        # Encode, decode
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class Generator():
    '''
    Initialize the model.
    ntokens = the size of the vocabulary
    emsize = embedding dimensions
    nhid = dimensions of feeedforward network model in nn.TransformerEncoder
    nlayers = number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = number of heads in the multheadattention models
    dropout = the dropout value
    '''
    def __init__(self, train_data, valid_data, test_data, ntokens, emsize=200, nhid=200, nlayers=2, nhead=2, dropout=0.2, bptt=35): 
        self.bptt = bptt
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.ntokens = ntokens
        # Find the device to run the model on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Construct the model
        self.model = TransformerModel(self.ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
        # Get the loss
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0 # learning rate
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    '''
    Generate input and target sequence for the transformer model.
    Subdivide source data into chunks of length 'bptt'
    '''
    def get_batch(self, i, source):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1 : i + 1 + seq_len].view(-1)
        return data, target
    
    '''
    Train the model.
    CrossEntropyLoss is applied to trakc the loss and SGD is the optimizer. 
    The initial learning rate is .50 and StepLR is applied to adjust the learing rate through epochs.
    clip_grad_norm scales all the gradients to prevent exploding.
    '''
    def train(self, epoch, log_interval=200):
        # Turn on the train mode of the model
        self.model.train()
        # Init values
        total_loss = 0
        start_time = time.time()
        # For the data
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            # Get a batch
            data, targets = get_batch(i, self.train_data)
            # Reset the optimizer to stop accumulating gradient
            optimizer.zero_grad()
            # Get output
            output = model(data)
            # Get loss
            loss = criterion(output.view(-1, self.ntokens), targets)
            loss.backward()
            # Clip to prevent exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # Step optimizer
            optimizer.step()
            # Accumulate loss
            total_loss += loss.item()
            # When an interval of batches is complete, log the progress
            if batch % log_interval == 0 and batch > 0:
                curr_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    curr_loss, math.exp(curr_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(self, data_source):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range (0, data_source.size(0) - 1, self.bptt):
                data, targets = get_batch(data_source, i)
                output = model(data)
                output_flat = output.view(-1, self.ntokens)
                total_loss += len(data) * self.criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def run(self, best_val_loss=float("inf"), epochs=3, best_model=None):
        # Loop over epochs, save the model if the validation loss is the best we've seen, adjust learning rate after each epoch
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            val_loss = evaluate(model, self.valid_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        scheduler.step()
        
        # Evaluate model with the test dataset, apply the best model to check the results with the dest dataset
        test_loss = evaluate(best_model, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        # Save model
        # torch.save(best_model.state_dict(), "model")