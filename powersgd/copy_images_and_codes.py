import os


gpus = [18, 17, 16, 15, 14, 13, 11, 10, 9, 7, 2]
# gpus = [12]
for gpu_id in gpus:
    copy_id_command = "ssh-copy-id 192.168.1.%d" % (40 + gpu_id)
    ssh_command = "ssh 192.168.1.%d" % (40 + gpu_id)
    mkdir_command = 'ssh 192.168.1.%d "mkdir ~/sparse_adam"' % (40 + gpu_id)
    image_command = "scp -r ~/wuhao_hipress_upgrade_image.tar 192.168.1.%d:~/wuhao_hipress_upgrade_image.tar" % (40 + gpu_id)
    code_hipress_mxnet_command = "scp ~/sparse_adam/hipress-examples/benchmark/powersgd/vgg/hipress_mxnet/hipress_mxnet.py 192.168.1.%d:~/sparse_adam/hipress_mxnet.py" % (40 + gpu_id)
    code_hipress_pytorch_command = "scp ~/sparse_adam/hipress-examples/benchmark/powersgd/vgg/hipress_pytorch/hipress_pytorch.py 192.168.1.%d:~/sparse_adam/hipress_pytorch.py" % (40 + gpu_id)
    code_torchddp_command = "scp ~/sparse_adam/hipress-examples/benchmark/powersgd/vgg/torchddp/torchddp_vgg.py 192.168.1.%d:~/sparse_adam/torchddp_vgg.py" % (40 + gpu_id)
    copy_hipress_pytorch_lstm = "scp ./hipress_pytorch/hipress_pytorch_lstm.py 192.168.1.%d:~/sparse_adam/hipress_pytorch_lstm.py" % (40 + gpu_id)

    os.system(copy_hipress_pytorch_lstm)

    