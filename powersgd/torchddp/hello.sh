mpirun --allow-run-as-root -n 2 -H 10.0.0.50:1,10.0.0.51:1 torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="192.168.1.51:43210" /workspace/hello.py

mpirun --allow-run-as-root -n 1 -H 10.0.0.51:1 torchrun --stand-a--nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="10.0.0.51:43210" /workspace/hello.py