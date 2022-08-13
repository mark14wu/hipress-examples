# 最新安装方法

1. 配置docker环境(可选)
   1. docker pull nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
   2. docker run --name wuhao_hipress_upgrade_test --gpus all --network=host --ipc=host --security-opt seccomp=unconfined -v /home/mark/sparse_adam:/workspace -v /data:/data -it nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 /bin/bash
2. 安装基本软件
   1. apt update
   2. apt install python3 python3-pip cmake openmpi-bin
      1. 上述安装过程中选择时区，需要分别输入6和70
3. ln -s /usr/bin/python3 /usr/bin/python
4. 安装horovod依赖：
   1. pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
5. 下载hipress代码
   1. git clone https://github.com/mark14wu/hipress.git
   2. git submodule init deps/torch-hipress-extension src/CCO && git submodule update
   3. cd src/CCO
   4. git submodule init && git submodule update
6. 编译hipress-torch扩展
   1. cd hipress/deps/torch-hipress-extension
   2. bash install.sh
7. 编译hipress本体
   1. cd hipress/src/CCO
   2. bash install.sh
8. 到hipress/src/CCO以外路径，python -c "import horovod.torch"，若无输出说明安装成功
9. 开始测试吧！

# 附录0-测试脚本仓库

https://github.com/mark14wu/hipress-examples

## hipress pytorch脚本

在hipress-example/powersgd/hipress_pytorch中

### VGG

模型代码：hipress-example/powersgd/hipress_pytorch/hipress_pytorch.py

1-8机脚本：hipress-example/powersgd/hipress_pytorch/10Gbps_VGG_BS16/

和

hipress-example/powersgd/hipress_pytorch/10Gbps_VGG_BS32/

### LSTM

模型代码：hipress-example/powersgd/hipress_pytorch/hipress_pytorch_lstm.py

1-8机脚本：hipress-example/powersgd/hipress_pytorch/10Gbps_LSTM_BS80/

### UGATIT

模型代码：hipress_pytorch_ugatit/main.py

1-8机脚本：hipress-example/powersgd/hipress_pytorch/10Gbps_UGATIT/

## torchddp 脚本

在hipress-example/powersgd/torchddp中

### VGG

模型代码：hipress-example/powersgd/torchddp/torchddp_vgg.py

1-8机脚本：在hipress-example/powersgd/torchddp/10Gbps_VGG_BS16/

### LSTM

模型代码：hipress-example/powersgd/torchddp/torchddp_lstm.py

1-8机脚本：hipress-example/powersgd/torchddp/10Gbps_LSTM_BS80/

### UGATIT

模型代码：hipress-example/powersgd/torchddp/torchddp_ugatit/main.py

1-8机脚本: hipress-example/powersgd/torchddp/10Gbps_UGATIT/

# 附录1-VGG脚本

1. VGG脚本需要安装tensorboardX，tqdm
   1. pip install tensorboardX tqdm

# 附录2-SSH配置

1. 可能需要安装ssh：apt install ssh
2. nano /etc/ssh/sshd_config，其中Port 22改为需要的端口(如1958)
3. nano ~/.ssh/config ，配置如下
   1. Host *
   2. Port 1958
4. 配置authorized_keys（把id_rsa.pub添加进去）
5. service ssh restart
6. 即可用ssh进行免密登陆
