import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import time
import numpy as np
from torch import optim
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook
import os


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

def get_batch(source, i):
    if USING_SYNTHETIC_DATA:
        seq_len = args.bptt
    else:
        seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class SyntheticDataLoader:
    def __init__(self, batch_size, max_iter, local_rank):
        self.cur_iter = 0
        self.max_iter = max_iter
        data = np.random.randint(0, SYNTHETIC_NTOKEN, [batch_size, ])
        # data = np.random.randint(0, 1, [batch_size, ])
        self.data = torch.from_numpy(data).to(local_rank)
        self.data_cache = dict()
    
    def __getitem__(self, key):
        assert isinstance(key, slice), "synthetic data must be a slice!"
        assert key.step is None, "synthetic data do not use slice step!"
        seq_len = key.stop - key.start
        if seq_len not in self.data_cache:
            data_seq = []
            for _ in range(seq_len):
                data_seq.append(self.data)
            data_seq = torch.stack(data_seq)
            self.data_cache[seq_len] = data_seq
            return data_seq
        else:
            return self.data_cache[seq_len]


USING_SYNTHETIC_DATA = True
# https://gist.github.com/Smerity/34e57f258cea48dba4c93d2261fc9330 shows the token number of wikitext-2
SYNTHETIC_NTOKEN = 33278

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=80,
                    help='training batch size per device (default: 80)')
parser.add_argument('--num-epochs', type=int, default=1)
parser.add_argument('--bptt', type=int, default=70)
parser.add_argument('--embedding-dim', type=int, default=1500)
parser.add_argument('--hidden-dim', type=int, default=1500)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--clip', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.65)
parser.add_argument('--log-interval', type=int, default=20)
parser.add_argument('--num-iterations', type=int, default=200)
parser.add_argument('--powersgd', type=int, default=0)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)

dist.init_process_group('nccl')
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

batch_size = args.batch_size
num_epochs = args.num_epochs
log_interval = args.log_interval
epoch_size = 200

model = RNNModel('LSTM', SYNTHETIC_NTOKEN, args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout)
model = model.to(local_rank)
raw_model = model
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

# PowerSGD configs here
low_rank = 1
state = PowerSGDState(
    process_group=dist.distributed_c10d.group.WORLD,
    matrix_approximation_rank=low_rank,
    use_error_feedback=True,
    warm_start=False,
    start_powerSGD_iter=10)
if args.powersgd:
    model.register_comm_hook(state, powerSGD_hook)

loss_fn = nn.NLLLoss()

opt = optim.SGD(model.parameters(), lr=0.001)

train_data = SyntheticDataLoader(args.batch_size, args.num_iterations, local_rank)

btic = time.time()
for epoch in range(num_epochs):
    # for nbatch, batch in enumerate(train_data, start=1):
    model.train()
    hidden = raw_model.init_hidden(args.batch_size)
    i = 0
    for nbatch in range(1, args.num_iterations + 1):
        data, label = get_batch(train_data, i)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss_fn(output, label).backward()
        if args.clip: 
            opt.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            with opt.skip_synchronize():
                opt.step()
        else:
            opt.step()
        if nbatch % log_interval == 0:
            if rank == 0:
                batch_speed = world_size * batch_size * log_interval / (time.time() - btic)
                logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                epoch, nbatch, batch_speed)
            btic = time.time()
        i += args.bptt