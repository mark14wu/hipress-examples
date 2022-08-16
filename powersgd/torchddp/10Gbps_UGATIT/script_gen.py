gpus = [18, 17, 16, 14, 13, 10, 9]
# gpus = [10, 9, 17, 16, 18, 14, 13]
gpus_per_node = 1
# gpus_per_node = 2
gpu_ips = ["192.168.2.%d" % (40 + x) for x in gpus]
master_ip = '192.168.1.58'
# master_ip = '192.168.1.50'
master_port = '43210'
master_address = ":".join((master_ip, master_port))

GENERATE_SCRIPT = False

if GENERATE_SCRIPT:
    assert input("will generate script, please confirm(yes/no)") == 'yes', "script generation cancelled."
else:
    N = int(input("input node number\n"))

for i in range(2, len(gpus) + 1):
    if not GENERATE_SCRIPT and i != N:
        continue
    script_filename = "%dN%dC.sh" % (i, i)
    selected_gpu_ip_name = ','.join(gpu_ips[:i])

    if GENERATE_SCRIPT:
        script_file = open(script_filename, 'w')
    else:
        print(script_filename)

    script = "mpirun --allow-run-as-root -n %d -H %s -x NCCL_SOCKET_IFNAME=ens14f1 -x NCCL_DEBUG=INFO -x NCCL_TREE_THRESHOLD=0 --mca btl tcp,self --mca btl_tcp_if_include ens14f1 -bind-to none -map-by slot torchrun --nnodes=%d --nproc_per_node=%d --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=\"%s\" /workspace/torchddp_ugatit/main.py --light True --batch_size 1" \
        % (i, selected_gpu_ip_name, i, gpus_per_node, master_address)

    if GENERATE_SCRIPT:
        script_file.write(script)
        script_file.close()
    else:
        print(script)