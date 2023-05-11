## Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds
Bo Yang, Jianan Wang, Ronald Clark, Qingyong Hu, Sen Wang, Andrew Markham, Niki Trigoni. [arXiv:1906.01140](https://arxiv.org/abs/1906.01140), 2019.
### (1) Setup
ubuntu 16.04 + cuda 8.0

python 2.7 or 3.6

tensorflow 1.2 or 1.4

scipy 1.3

h5py 2.9

open3d-python 0.3.0

#### Compile tf_ops
(1) To find tensorflow include path and library paths:

    import tensorflow as tf
    print(tf.sysconfig.get_include())
    print(tf.sysconfig.get_lib())

(2) To change the path in all the complie files, e.g. tf_ops/sampling/tf_sampling_compile.sh, and then compile:

    cd tf_ops/sampling
    chmod +x tf_sampling_compile.sh
    ./tf_sampling_compile.sh

### (2) Data
S3DIS: [https://drive.google.com/open?id=1hOsoOqOWKSZIgAZLu2JmOb_U8zdR04v0](https://drive.google.com/open?id=1hOsoOqOWKSZIgAZLu2JmOb_U8zdR04v0)

百度盘: [https://pan.baidu.com/s/1ww_Fs2D9h7_bA2HfNIa2ig](https://pan.baidu.com/s/1ww_Fs2D9h7_bA2HfNIa2ig) 密码:qpt7

Acknowledgement: we use the same data released by [JSIS3D](https://github.com/pqhieu/jsis3d).

### (3) Train/test
python main_train.py

python main_eval.py

### (4) Quantitative Results on ScanNet
![Arch Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_res_scannet.png)
### (5) Qualitative Results on ScanNet
![Arch Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_ins_scannet.png)

| ![2](./figs/fig_scannet_scene0015.gif)   | ![z](./figs/fig_scannet_scene0081.gif) |
| ---------------------------------------- | -------------------------------------- |
| ![z](./figs/fig_scannet_scene0088.gif)   | ![z](./figs/fig_scannet_scene0196.gif) |

#### More results of ScanNet validation split are available at: [More ScanNet Results](https://drive.google.com/file/d/1cV07rP02Yi3Eu6GQxMR2buigNPJEvCq0/view?usp=sharing)
To visualize:
python helper_data_scannet.py

### (6) Qualitative Results on S3DIS
| ![z](./figs/fig_s3dis_area2_auditorium.gif)   | ![z](./figs/fig_s3dis_area6_hallway1.gif) |
| --------------------------------------------- | ----------------------------------------- |

![Teaser Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_bb_s3dis.png)
### (7) Training Curves on S3DIS
![Teaser Image](https://github.com/Yang7879/3D-BoNet/blob/master/figs/fig_traincurv_s3dis.png)

### (8) Video Demo (Youtube)
<p align="center"> <a href="https://www.youtube.com/watch?v=Bk727Ec10Ao"><img src="./figs/fig_video_demo_cover.png" width="80%"></a> </p>

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{yang2019learning,
      title={Learning object bounding boxes for 3d instance segmentation on point clouds},
      author={Yang, Bo and Wang, Jianan and Clark, Ronald and Hu, Qingyong and Wang, Sen and Markham, Andrew and Trigoni, Niki},
      booktitle={Advances in Neural Information Processing Systems},
      pages={6737--6746},
      year={2019}
    }
    
## Related Repos
1. [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/RandLA-Net.svg?style=flat&label=Star)
2. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
3. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)
4. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
