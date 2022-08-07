import torch
from torch import nn
import torch.distributed as c10d
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
from itertools import product
import tempfile
import time


file_name = tempfile.NamedTemporaryFile(delete=False).name
world_size = 1
rank = 0

class MyGradBucket:
    def __init__(self, grad_size, rank=0):
        self._grad = (rank + 1) * torch.ones(grad_size, device=torch.device('cuda'))
        # for i in range(rank):
        #     self._grad += torch.ones(grad_size, device=torch.device('cuda'))
        # torch.normal(mean=torch.arange(1., grad_size), std=torch.arange(1, 0, -0.1), out=self._grad)
    def index(self) -> int: 
        return 0
    def buffer(self): 
        return self._grad
    def gradients(self): 
        return [self._grad]
    def is_last(self) -> bool: 
        return False

def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.cuda.device_count()))
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank

class Task(nn.Module):
    def __init__(self, grad_size):
        super().__init__()
        self.p = nn.Parameter(torch.ones(grad_size))

    def forward(self, x):
        return self.p + x

class ModuleForDdpCommHook(nn.Module):
    def __init__(self, grad_size):
        super().__init__()
        self.t0 = Task(grad_size)

    def forward(self, x, rank):
        return self.t0(x)

# def _gpu_model_with_ddp_comm_hook(
#         process_group,
#         hook=None,
#         gradient_as_bucket_view=False,
#         state=None,
#         static_graph=False,
#         grad_size=262144
#     ):
#         device_id = gpus_for_rank(world_size)[rank][0]
#         gpu_model = DistributedDataParallel(
#             ModuleForDdpCommHook(grad_size).to(device_id),
#             device_ids=[device_id],
#             process_group=process_group,
#             gradient_as_bucket_view=gradient_as_bucket_view,
#             static_graph=static_graph,
#         )

#         # Register a DDP communication hook if any.
#         if hook is not None:
#             gpu_model.register_comm_hook(state, hook)

#         return gpu_model

# def _run_hook(model, input):
#     # Run forward
#     output = model(input, rank)

#     # Run backward
#     output.mean().backward()

def sync_powersgd_hook(state, bucket):
    hook = powerSGD.powerSGD_hook
    # t1 = time.time()
    grad = hook(state, bucket).wait()
    # print(grad.size())
    # t2 = time.time()
    # print(1000 * (t2 - t1))
    return hook(state, bucket)

class Benchmark:
    def __init__(self, low_rank):
        self.store = c10d.FileStore(file_name, world_size)
        self.process_group = c10d.ProcessGroupNCCL(self.store, 0, world_size)
        self.low_rank = low_rank
        self.use_error_feedback = True
        self.warm_start = False
        self.batch_tensors_with_same_shape = False

        self.state = powerSGD.PowerSGDState(
            process_group=self.process_group,
            matrix_approximation_rank=self.low_rank,
            use_error_feedback=self.use_error_feedback,
            warm_start=self.warm_start,
            batch_tensors_with_same_shape=self.batch_tensors_with_same_shape,
        )
    
    def hook(self, grad_bucket):
        powerSGD.powerSGD_hook(self.state, grad_bucket)

def benchmark_xpu(grad_size, iters, low_rank):
    benchmark = Benchmark(low_rank)
    e2e_times = []
    for i in range(iters):
        bucket = MyGradBucket(grad_size, i)
        # 1. synchornize
        # 2. cuda event
        # 3. nsight
        torch.cuda.synchronize()
        t1 = time.time()
        benchmark.hook(bucket)
        torch.cuda.synchronize()
        t2 = time.time()
        e2e_times.append(1000 * (t2 - t1))
        # sleep(0.01)
    head = int(iters * 0.25)
    tail = int(iters * 0.75)
    e2e_times.sort()
    e2e_times = e2e_times[head: tail]
    avg_e2e_time = sum(e2e_times) / len(e2e_times)
    return avg_e2e_time

def _test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=False):

    grad_sizes = []
    grad_sizes += ['0.01MB']
    grad_sizes += ['0.1MB']
    grad_sizes += ['1MB']
    grad_sizes += ['10MB']
    grad_sizes += ['100MB']
    grad_sizes += ['500MB']
    # grad_sizes += ['2000MB']
    iters = 200
    low_rank = 1
    print("rank = {}".format(low_rank))

    

    # hook = sync_powersgd_hook
    

    for grad_size_str in grad_sizes:
        grad_size = int(float(grad_size_str[:grad_size_str.find('MB')]) * 262144)
        e2e_times = []
        avg_e2e_time = benchmark_xpu(grad_size, iters, low_rank)
        print("size: {}, total: {:.2f}".format(grad_size_str, avg_e2e_time))
        # gpu_model = _gpu_model_with_ddp_comm_hook(
        #     process_group, hook, gradient_as_bucket_view, state, grad_size=grad_size
        # )
        # _run_hook(gpu_model, torch.ones(grad_size))

if __name__ == '__main__':
    _test_powerSGD_ddp_comm_hook_nccl()