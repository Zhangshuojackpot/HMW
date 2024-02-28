# Hyperspherical-Margin-Weighting (HMW)
This is the official PyTorch implementation of our work [Learning with Noisy Labels Using Hyperspherical Margin Weighting](https://assets.underline.io/lecture/92576/paper/d8dd287c7e8050f705e12e122847cdb0.pdf?Expires=1709306131&Signature=Pk7supOyqA3~Os3I2ScI23514svtHlgWd1YdcFrlw7pCTrywlU00UtvsWv5IHTnhe-wfyOXkAZJxRqnwwzjqIUAflHCTLPZLSVpUtX~nq32vt9cjn80JsQocMsq4jUx-4JDjPdVsXO1ALR9HGxqTtJX4Y-elbm0Y3p0ZLoCHaAMDC-5c6soDnxoo~ixLFjn2A~DVBPFSivXWnlnFWv6gJ7sBGpQB7VlL1aqzQNpWWisgLDncrouja26BE9vbdZQzh6xXw3LGv4NrgTHfJWHh9W~h6TlNfVLTT-JoQeKlAia9sJq0ZqgF89nGFRTcLGYnG6cdMRt2hKDFuB1TAzN-3Q__&Key-Pair-Id=K2CNXR0DE4O7J0), which has been published in AAAI-24. This repo contains some key codes of our HMW and its application in CIFAR10/CIFAR100 dataset.<br>
<div align=center>
<img width="800" src="https://github.com/Zhangshuojackpot/HMW/blob/main/4766.png"/>
</div>

### Abstract
Datasets often include noisy labels, but learning from them is difficult. Since mislabeled examples usually have larger loss values in training, the small-loss trick is regarded as a standard metric to identify the clean example from the training set for better performance. Nonetheless, this proposal ignores that some clean but hard-to-learn examples also generate large losses. They could be misidentified by this criterion. In this paper, we propose a new metric called the Integrated Area Margin (IAM), which is superior to the traditional small-loss trick, particularly in recognizing the clean but hard-to-learn examples. According to the IAM, we further offer the Hyperspherical Margin Weighting (HMW) approach. It is a new sample weighting strategy that restructures the importance of each example. It should be highlighted that our approach is universal and can strengthen various methods in this field. Experiments on both benchmark and real-world datasets indicate that our HMW outperforms many state-of-the-art approaches in learning with noisy label tasks.

### Preparation
The experimental environment is in [requirements.txt](https://github.com/Zhangshuojackpot/HMW/blob/main/requirements.txt).<br>

### Usage
Run [train_main.py](https://github.com/Zhangshuojackpot/HMW/blob/main/HMW_upload/train_main.py) to obtain the results. For example, if you want to obtain the result of the HMW+CCE under the noise rate of 0.2 of the symmetric noise on CIFAR10, you can type:<br>
```
python train_main.py --if_tkum 1 --if_anneal 1 --if_spherical 1 --seed 123 --dataset 'cifar10' --num_class 10 --data_path './data/cifar10' --noise_mode 'sym' --num_epochs 250 --milestones '125/200' --r 0.2 --method 'hmw+cce' --alpha_nmw 100 --beta_nmw 100 --top_rate_nmw 0.01 --check_loc './checkpoint_cifar10_cce_final_sym_hmw/'

```

If you want to obtain the result of the HMW+CCE under the noise rate of 0.2 of the symmetric noise on CIFAR100, you can type:<br>
```
python train_main.py --if_tkum 1 --if_anneal 1 --if_spherical 1 --seed 123 --dataset 'cifar100' --num_class 100 --data_path './data/cifar100' --noise_mode 'sym' --num_epochs 250 --milestones '125/200' --r 0.2 --method 'hmw+cce' --alpha_nmw 100 --beta_nmw 100 --top_rate_nmw 0.01 --check_loc './checkpoint_cifar100_cce_final_sym_hmw/'

```

### Citation
If you think this repo is useful in your research, please consider citing our paper (We will add the bibtex format of our paper when it has been officially published online by AAAI-24).

Meanwhile, our implementation uses parts of some public codes in [UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Karim_UniCon_Combating_Label_Noise_Through_Uniform_Selection_and_Contrastive_Learning_CVPR_2022_paper.html). Please consider citing this paper.
```
@InProceedings{Karim_2022_CVPR,
    author    = {Karim, Nazmul and Rizve, Mamshad Nayeem and Rahnavard, Nazanin and Mian, Ajmal and Shah, Mubarak},
    title     = {UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9676-9686}
}
```
