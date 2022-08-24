# lstm pytorch ddp
mpirun --allow-run-as-root -bind-to none -map-by slot -n 2 -H 10.0.0.58:1,10.0.0.41:1 -x NCCL_SOCKET_IFNAME=ib0 -x NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="192.168.1.41:43210" ./torchddp/torchddp_lstm.py --batch-size 80 --num-epochs 1 --log-interval 20

NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=0 NCCL_IB_HCA=mlx4_0 horovodrun --start-timeout 60 --network-interface ib0 -np 2 -H 10.0.0.58:1,10.0.0.41:1 torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="10.0.0.58:43210" ./torchddp/torchddp_lstm.py --batch-size 80 --num-epochs 1 --log-interval 20

# lstm pytorch powersgd
NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=0 NCCL_IB_HCA=mlx4_0 horovodrun --start-timeout 60 --network-interface ib0 -np 2 -H 10.0.0.58:1,10.0.0.41:1 torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="10.0.0.58:43210" ./torchddp/torchddp_lstm.py --batch-size 80 --num-epochs 1 --log-interval 20 --powersgd 1

# lstm hipress
NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO NCCL_IB_HCA=mlx4_0 horovodrun -np 2 -H 10.0.0.58:1,10.0.0.41:1 --start-timeout 60 --network-interface ib0 python ./hipress_pytorch/hipress_pytorch_lstm.py --batch_size 80 --epochs 1 --threshold 0 --num-iterations 200 --partition-threshold 4194304 --algorithm powersgd


# ugatit pytorch ddp
NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO NCCL_IB_HCA=mlx4_0 horovodrun -np 2 -H 10.0.0.58:1,10.0.0.41:1 --start-timeout 60 --network-interface ib0 torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="10.0.0.58:43210" ./torchddp/torchddp_ugatit/main.py --light True --batch_size 1

# ugatit pytorch powersgd
NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO NCCL_IB_HCA=mlx4_0 horovodrun -np 2 -H 10.0.0.58:1,10.0.0.41:1 --start-timeout 60 --network-interface ib0 torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="10.0.0.58:43210" ./torchddp/torchddp_ugatit/main.py --light True --batch_size 1 --powersgd 1

# ugatit hipress
DISTRIBUTED_FRAMEWORK=horovod NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO NCCL_IB_HCA=mlx4_0 horovodrun -np 2 -H 10.0.0.58:1,10.0.0.41:1 --start-timeout 60 --network-interface ib0 python ./hipress_pytorch/hipress_pytorch_ugatit/main.py --dataset selfie2anime --dataset_dir /data/trainData/ --light True --batch_size 1 --threshold 0