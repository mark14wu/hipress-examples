gpus = [18, 17, 16, 14, 13, 11, 10, 9, 15]
master_gpu = 11
gpus.remove(master_gpu)

N = int(input("input node number:"))
worker_per_node = 1
master_addr = "192.168.1.51"
master_port = "43210"
batch_size = 16

assert N >= 2, "1N1C not supported!"
assert N <= len(gpus), "not enough GPUs!"

hosts = ["192.168.2.%d:1" % (40 + master_gpu)]
for i in range(N - 1):
    hosts.append("192.168.2.%d:1" % (40 + gpus[i]))
hosts_str = ','.join(hosts)

run_torchddp_command = """mpirun --allow-run-as-root -n %d -H %s -x NCCL_SOCKET_IFNAME=ens14f1 -x NCCL_DEBUG=INFO -x NCCL_TREE_THRESHOLD=0 --mca btl tcp,self --mca btl_tcp_if_include ens14f1 -bind-to none -map-by slot torchrun --nnodes=%d --nproc_per_node=%d --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="%s:%s" /workspace/torchddp_vgg.py --batch-size 16 --num-epochs 1 --log-interval 20 --model vgg19""" \
% (N, hosts_str, N, worker_per_node, master_addr, master_port)

# run_hipress_mxnet_command = """MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=15 MXNET_NUM_OF_PARTICIPANTS=%d NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens14f1 \
# horovodrun -np %d -H %s python /workspace/hipress_mxnet.py --batch-size 16 --num-epochs 1 --log-interval 20 --model vgg19""" \
#     % (N, N, hosts_str)

run_hipress_mxnet_command = """NCCL_SOCKET_IFNAME=ens14f1 horovodrun -np %d -H %s python /workspace/hipress_mxnet.py --batch-size %d --num-epochs 1 --log-interval 20 --model vgg19""" \
    % (N, hosts_str, batch_size)

log_command = "2>&1 | tee %dN%dC.log" % (N, N)

# print(run_torchddp_command)
print(run_hipress_mxnet_command + ' ' + log_command)