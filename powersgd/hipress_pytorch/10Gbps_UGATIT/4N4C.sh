DISTRIBUTED_FRAMEWORK=horovod NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_SOCKET_IFNAME=ens14f1 horovodrun -np 4 -H 192.168.2.58:1,192.168.2.57:1,192.168.2.56:1,192.168.2.54:1 --start-timeout 600 python /workspace/hipress_pytorch_ugatit/main.py --dataset selfie2anime --dataset_dir /data/trainData/ --light True --batch_size 1 --threshold 0