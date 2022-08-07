import torch
import argparse
import logging
import time
import numpy as np
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook
import os


class SyntheticDataIter():
    def __init__(self, num_classes, data_shape, max_iter, rank):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        # self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape).astype(np.float32)
        self.data = torch.from_numpy(data).to(rank)
        self.label = torch.from_numpy(label).to(rank)

    def __iter__(self):
        return self

    # @property
    # def provide_data(self):
    #     return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    # @property
    # def provide_label(self):
    #     return [mx.io.DataDesc('softmax_label',
    #                            (self.batch_size,), self.dtype)]

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return self.data, self.label
        else:
            raise StopIteration
        #     return DataBatch(data=(self.data,),
        #                      label=(self.label,),
        #                      pad=0,
        #                      index=None,
        #                      provide_data=self.provide_data,
        #                      provide_label=self.provide_label)
        # else:
        #     raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (default: 128)')
    parser.add_argument('--model', type=str, default='vgg19',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--num-epochs', type=int, default=90,
                        help='number of training epochs (default: 90)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of NN output classes, default is 1000')
    parser.add_argument('--log-interval', type=int, default=20,
                    help='number of batches to wait before logging (default: 20)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    dist.init_process_group('nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    log_interval = args.log_interval
    epoch_size = 200
    image_shapes = {
        'vgg19': (3, 224, 224)
    }
    image_shape = image_shapes[args.model]
    data_shape = (batch_size,) + image_shape

    models = {
        'vgg19': torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    }
    model = models[args.model].to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # PowerSGD configs here
    low_rank = 1
    state = PowerSGDState(
        process_group=dist.distributed_c10d.group.WORLD,
        matrix_approximation_rank=low_rank,
        use_error_feedback=True,
        warm_start=False,
        batch_tensors_with_same_shape=False)
    model.register_comm_hook(state, powerSGD_hook)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(), lr=0.001)

    train_data = SyntheticDataIter(num_classes, data_shape, epoch_size, local_rank)

    btic = time.time()
    for epoch in range(num_epochs):
        for nbatch, batch in enumerate(train_data, start=1):
            data, label = batch
            output = model(data)
            loss_fn(output, label).backward()
            opt.step()
            if nbatch % log_interval == 0:
                if rank == 0:
                    batch_speed = world_size * batch_size * log_interval / (time.time() - btic)
                    logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                    epoch, nbatch, batch_speed)
                btic = time.time()

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
