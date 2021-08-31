
fork from: https://github.com/QingyongHu/RandLA-Net.git
作者：https://zhuanlan.zhihu.com/p/105433460



## 1 install
```
Python 3.6, Tensorflow 2.6, CUDA 11.4 and cudnn ( /usr/local/cuda-11.4 没有用 conda 的cudatoolkit)
git clone --depth=1 https://github.com/luckyluckydadada/randla-net-tf2.git
conda create -n randlanet python=3.6 
conda activate randlanet
pip install tensorflow-gpu==2.6 -i https://pypi.tuna.tsinghua.edu.cn/simple  --timeout=120
pip install -r helper_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  --timeout=120
sh compile_op.sh
```
## 2 data-S3DIS
```
ls /home/$USER/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version
python utils/data_prepare_s3dis.py
```

## 3 配置
```
ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None
```
## 4 train

```
python -B main_S3DIS.py --gpu 0 --mode train --test_area 1 表示:
Area2-3-4-5-6共228个ply文件作为训练集
Area1共44个ply文件作为验证集
下面为6折交叉验证：
cat jobs_6_fold_cv_s3dis.sh 
python -B main_S3DIS.py --gpu 0 --mode train --test_area 1  # Area23456作为训练集，area1作为验证集
python -B main_S3DIS.py --gpu 0 --mode test --test_area 1
python -B main_S3DIS.py --gpu 0 --mode train --test_area 2
python -B main_S3DIS.py --gpu 0 --mode test --test_area 2
python -B main_S3DIS.py --gpu 0 --mode train --test_area 3
python -B main_S3DIS.py --gpu 0 --mode test --test_area 3
python -B main_S3DIS.py --gpu 0 --mode train --test_area 4
python -B main_S3DIS.py --gpu 0 --mode test --test_area 4
python -B main_S3DIS.py --gpu 0 --mode train --test_area 5
python -B main_S3DIS.py --gpu 0 --mode test --test_area 5
python -B main_S3DIS.py --gpu 0 --mode train --test_area 6
python -B main_S3DIS.py --gpu 0 --mode test --test_area 6

```

论文里说，RandLA-Net是一个完全端到端的网络架构，能够将整个点云作为输入，而不用拆分、合并等预处理、后处理操作。但是我在阅读代码的时候，发现网络在获取数据的部分，采用的是和KPConv类似的思路：先随机选择一个train文件，在文件中随机选择一批中心点，然后在这些中心点周围用KNN的方法选择K个点，作为一个batch训练所需的点云。我看代码里确实是这样做的，感觉和论文里说的有点出入，不知道是不是我理解的有问题。
对于 Semantic 3D 数据集，代码里先对原始点云进行 grid_size=0.01 的采样，得到点云1，然后对点云1进行 grid_size=0.06 的采样，得到点云2，最后对点云2进行学习。而最后test的时候，网络只把特征传播到点云1，意思就是语义分割的结果只预测到了点云1，并没有预测原始点云的分割结果，所以就对这里产生了一些疑问。S3DIS就没有这个问题，最后预测的是原始点云的语义分割结果。

## 5 vis
label
python  main_S3DIS.py --mode vis --test_area 1
origin
python vis_S3DIS.py 

## 6 code
helper_tf_util.py	封装了一些卷积池化操作代码
helper_tool.py	  有训练时各个数据集所用到的一些参数信息，还有一些预处理数据时的一些模块。
main_*.py	        训练对应数据的主文件
RandLANet.py	    定义网络的主题结构
tester_*.py	      测试对应数据的文件，该文件在main_*.py中被调用
utils	            对数据集预处理的模块以及KNN模块。
utils/data_prare_*.py 预处理，把point和label做了grid sampling，并且生成了一个kdtree保存下来


## 7 NET
将特征升维到8
encoder：由4个（dilated_res_block+random_sample）构成，形成特征金字塔
将金字塔尖的特征再次计算以下
decoder：由4个（nearest_interpolation+conv2d_transpose）构成，恢复到point-wise的特征
由point-wise经过一些MLP，得到f_out


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/semantic-segmentation-on-semantic3d)](https://paperswithcode.com/sota/semantic-segmentation-on-semantic3d?p=191111236)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=191111236)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds (CVPR 2020)

This is the official implementation of **RandLA-Net** (CVPR2020, Oral presentation), a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds. For technical details, please refer to:
 
**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** <br />
[Qingyong Hu](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang*](https://yang7879.github.io/), [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), [Stefano Rosa](https://www.cs.ox.ac.uk/people/stefano.rosa/), [Yulan Guo](http://yulanguo.me/), [Zhihua Wang](https://www.cs.ox.ac.uk/people/zhihua.wang/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). <br />
**[[Paper](https://arxiv.org/abs/1911.11236)] [[Video](https://youtu.be/Ar3eY_lwzMk)] [[Blog](https://zhuanlan.zhihu.com/p/105433460)] [[Project page](http://randla-net.cs.ox.ac.uk/)]** <br />
 
 
<p align="center"> <img src="http://randla-net.cs.ox.ac.uk/imgs/Fig3.png" width="100%"> </p>


	
### (1) Setup
This code has been tested with Python 3.5, Tensorflow 1.11, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04.
 
- Clone the repository 
```
git clone --depth=1 https://github.com/QingyongHu/RandLA-Net && cd RandLA-Net
```
- Setup python environment
```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
```

**Update 03/21/2020, pre-trained models and results are available now.** 
You can download the pre-trained models and results [here](https://drive.google.com/open?id=1iU8yviO3TP87-IexBXsu13g6NklwEkXB).
Note that, please specify the model path in the main function (e.g., `main_S3DIS.py`) if you want to use the pre-trained model and have a quick try of our RandLA-Net.

### (2) S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Start 6-fold cross validation:
```
sh jobs_6_fold_cv_s3dis.sh
```
- Move all the generated results (*.ply) in `/test` folder to `/data/S3DIS/results`, calculate the final mean IoU results:
```
python utils/6_fold_cv.py
```

Quantitative results of different approaches on S3DIS dataset (6-fold cross-validation):

![a](http://randla-net.cs.ox.ac.uk/imgs/S3DIS_table.png)

Qualitative results of our RandLA-Net:

| ![2](imgs/S3DIS_area2.gif)   | ![z](imgs/S3DIS_area3.gif) |
| ------------------------------ | ---------------------------- |



### (3) Semantic3D
7zip is required to uncompress the raw data in this dataset, to install p7zip:
```
sudo apt-get install p7zip-full
```
- Download and extract the dataset. First, please specify the path of the dataset by changing the `BASE_DIR` in "download_semantic3d.sh"    
```
sh utils/download_semantic3d.sh
```
- Preparing the dataset:
```
python utils/data_prepare_semantic3d.py
```
- Start training:
```
python main_Semantic3D.py --mode train --gpu 0
```
- Evaluation:
```
python main_Semantic3D.py --mode test --gpu 0
```
Quantitative results of different approaches on Semantic3D (reduced-8):

![a](http://randla-net.cs.ox.ac.uk/imgs/Semantic3D_table.png)

Qualitative results of our RandLA-Net:

| ![z](imgs/Semantic3D-1.gif)    | ![z](http://randla-net.cs.ox.ac.uk/imgs/Semantic3D-2.gif)   |
| -------------------------------- | ------------------------------- |
| ![z](imgs/Semantic3D-3.gif)    | ![z](imgs/Semantic3D-4.gif)   |



**Note:** 
- Preferably with more than 64G RAM to process this dataset due to the large volume of point cloud


### (4) SemanticKITTI

SemanticKITTI dataset can be found <a href="http://semantic-kitti.org/dataset.html#download">here</a>. Download the files
 related to semantic segmentation and extract everything into the same folder. Uncompress the folder and move it to 
`/data/semantic_kitti/dataset`.
 
- Preparing the dataset:
```
python utils/data_prepare_semantickitti.py
```

- Start training:
```
python main_SemanticKITTI.py --mode train --gpu 0
```

- Evaluation:
```
sh jobs_test_semantickitti.sh
```

Quantitative results of different approaches on SemanticKITTI dataset:

![s](http://randla-net.cs.ox.ac.uk/imgs/SemanticKITTI_table.png)

Qualitative results of our RandLA-Net:

![zzz](imgs/SemanticKITTI-2.gif)    


### (5) Demo

<p align="center"> <a href="https://youtu.be/Ar3eY_lwzMk"><img src="http://randla-net.cs.ox.ac.uk/imgs/demo_cover.png" width="80%"></a> </p>


### Citation
If you find our work useful in your research, please consider citing:

	@article{hu2019randla,
	  title={RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds},
	  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
	  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2020}
	}
	
	@article{hu2021learning,
	  title={Learning Semantic Segmentation of Large-Scale Point Clouds with Random Sampling},
	  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
	  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	  year={2021},
	  publisher={IEEE}
	}


### Acknowledgment
-  Part of our code refers to <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library and the the recent work <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a>.
-  We use <a href="https://www.blender.org/">blender</a> to make the video demo.


### License
Licensed under the CC BY-NC-SA 4.0 license, see [LICENSE](./LICENSE).


### Updates
* 21/03/2020: Updating all experimental results
* 21/03/2020: Adding pretrained models and results
* 02/03/2020: Code available!
* 15/11/2019: Initial release！

## Related Repos
1. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
2. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
5. [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds with 1000x Fewer Labels](https://github.com/QingyongHu/SQN) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SQN.svg?style=flat&label=Star)


