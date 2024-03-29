from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import tensorboardX
import torch.autograd.profiler as profiler
import os
import math
from tqdm import tqdm
import numpy as np
import logging


class SyntheticDataIter():
    def __init__(self, num_classes, data_shape, max_iter, rank):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape).astype(np.float32)
        self.data = torch.from_numpy(data).to(rank)
        self.label = torch.from_numpy(label).to(rank)

    def __iter__(self):
        return self

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return self.data, self.label
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0
    
    def __len__(self):
        return self.max_iter


torch.cuda.current_device()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--num-iterations', type=int, default=10,
                    help='number of iterations trained per epoch')
parser.add_argument('--print-intervals', type=int, default=20,
                    help = "print average speed per interval epochs")
parser.add_argument('--threshold', type=int, default=262144,
                    help = 'minimal(excluded) number of floats to compress')

parser.add_argument('--partition-threshold', type=int, default=16777216,
                    help = 'minimal(excluded) number of floats to partition')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--model', type=str, default='vgg19',
                    help='random seed')

parser.add_argument('--algorithm', type=str, default='tbq',
                    help='random seed')

logging.basicConfig(level=logging.INFO)

args = parser.parse_args()
logging.info(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

allreduce_batch_size = args.batch_size * args.batches_per_allreduce

hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None


kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

synthetic_training_data = True

if not synthetic_training_data:
    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=1, rank=0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)

    val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                            sampler=val_sampler, **kwargs)
else:
    data_shape = (args.batch_size,) + (3, 224, 224)
    train_loader = SyntheticDataIter(1000, data_shape, args.num_iterations, hvd.local_rank())
    val_loader = None




# Set up standard ResNet-50 model.
#model = models.resnet50()
model = models.vgg19() if args.model == 'vgg19' else models.resnet50()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce * hvd.size()),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.

if args.algorithm == 'tbq':
    algorithm_params = {"threshold" : 2}
elif args.algorithm == 'terngrad':
    algorithm_params = {"enable_random" : 0, 'bitwidth' : 2}
elif args.algorithm == 'graddrop':
    algorithm_params = {"sample_rate" : 0.001, "drop_ratio" : 0.999}
elif args.algorithm == 'powersgd':
    algorithm_params = {"matrix_approximation_rank": 1}
else:
    raise ValueError
    
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_allreduce,
    threshold=args.threshold,
    partition_threshold=args.partition_threshold,
    algorithm_name = args.algorithm,
    algorithm_params = algorithm_params
    )

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

import time
last_time = time.time()

logging.info("iterations:", args.num_iterations)

def train(epoch):
    model.train()
    if not synthetic_training_data:
        train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    if True:
        btic = time.time()
        #cpu api begin end
        with profiler.profile(enabled=False) as prof:
            with profiler.record_function("model_inference"):
                for batch_idx, (data, target) in enumerate(train_loader):
                    adjust_learning_rate(epoch, batch_idx)
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    optimizer.zero_grad()
                    # Split data into sub-batches of size batch_size
                    for i in range(0, len(data), args.batch_size):
                        data_batch = data[i:i + args.batch_size]
                        target_batch = target[i:i + args.batch_size]
                        with profiler.record_function("forward"):
                            output = model(data_batch)
                            #train_accuracy.update(accuracy(output, target_batch))
                            loss = F.cross_entropy(output, target_batch)
                        #train_loss.update(loss)
                        # Average gradients among sub-batches
                            loss.div_(math.ceil(float(len(data)) / args.batch_size))
                        with profiler.record_function("backward"):
                            loss.backward()
                    # Gradient is applied across all ranks
                    with profiler.record_function("update"):
                        optimizer.step()
                    if (batch_idx + 1) % args.print_intervals == 0:
                        average_speed = hvd.size() * args.batch_size * args.print_intervals / (time.time() - btic)
                        btic = time.time()
                        if hvd.rank() == 0:
                            logging.info("Epoch[{:d}] Batch[{:d}]\tSpeed: {:.2f} samples/sec".format(epoch, batch_idx + 1, average_speed))
                    if (batch_idx >= args.num_iterations):
                        break

            #prof.export_chrome_trace("vgg19" + str(hvd.rank()) + ".json")
    #import lltm_cuda
    #lltm_cuda.end()
    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    # validate(epoch)
    # save_checkpoint(epoch)
import hp_cuda
hp_cuda.end()
