mpirun --allow-run-as-root -n 6 -H 192.168.2.58:1,192.168.2.57:1,192.168.2.56:1,192.168.2.54:1,192.168.2.53:1,192.168.2.50:1 -x NCCL_SOCKET_IFNAME=ens14f1 -x NCCL_DEBUG=INFO torchrun --nnodes=6 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="192.168.1.58:43210" /workspace/torchddp_ugatit/main.py --dataset selfie2anime --dataset_dir /data/trainData/ --light True --batch_size 1