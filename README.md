# PATEandMEAL
配置：

python 3.6

torch 1.5.0

torchvision 0.6.0

cuda 10.2

对于our method 通过运行一下命令来得到实验结果：

python main.py  --teachers [\'vgg19_BN\'] --student vgg19_BN --d_lr 1e-3 --fc_out 1 --pool_out avg --loss ce --adv 1 --out_layer [0,1,2,3,4] --out_dims [10000,5000,1000,500,10] --gamma [0.001,0.01,0.05,0.1,1] --eta [1,1,1,1,1] --name vgg_test

其中网络结构可换。运行时注意GPU ID。

对于PATE文件，我已经在原始PATE的基础上进行了修改，切换了数据集与网络结构，运行方式为：

python main.py

原始Torch版本PATE文件网址：https://github.com/kamathhrishi/PATE 可自行下载运行。
Tensorflow版本PATE可直接安装API

Teacher模型下载：https://pan.baidu.com/s/1Ov6UAEMzt2N8IOPK9vG_TQ

提取码：8jr4
