# Hyperspherical-Margin-Weighting (HMW)
This is the official PyTorch implementation of our work [Learning with Noisy Labels Using Hyperspherical Margin Weighting](https://assets.underline.io/lecture/92576/paper/d8dd287c7e8050f705e12e122847cdb0.pdf?Expires=1709020119&Signature=u-yG0NYH~LDQ-E1mYtHjivLzuNVTDqjD1Q3b4U~vTbFTRu9x2Y~~7Z7wBW8h3rQx7V9seEOVqFCESla4Kr7Akk8UZrwSf00DEioV-tCR1BvSza3pk8t3l2h4yS29r9iBPF6WC~IAQ0kO-2~Mw7q5f0lzpSLqooj6p0cYpCP1z8DdoVmRR9hRjjSCzPC3zPxI4yZuwIY0kr7bhnOsXP0X4WUmcf45rk1oIT9HDdwpPmVW~drxuOlPbeyiwGR6gmlH~a2R6hf~vKOv~kRb8OcA5ITEcZde~ZxFlg7g~wGczzA4l0fRqwvjhjg36djVFJcHlP2juf1uXQuZBlXOi2yXbw__&Key-Pair-Id=K2CNXR0DE4O7J0), which has been published in AAAI-24. This repo contains some key codes of our HMW and its application in CIFAR10/CIFAR100 dataset.<br>
<div align=center>
<img width="800" src="https://github.com/Zhangshuojackpot/HMW/blob/main/4766.png"/>
</div>

### Abstract
Datasets often include noisy labels, but learning from them is difficult. Since mislabeled examples usually have larger loss values in training, the small-loss trick is regarded as a standard metric to identify the clean example from the training set for better performance. Nonetheless, this proposal ignores that some clean but hard-to-learn examples also generate large losses. They could be misidentified by this criterion. In this paper, we propose a new metric called the Integrated Area Margin (IAM), which is superior to the traditional small-loss trick, particularly in recognizing the clean but hard-to-learn examples. According to the IAM, we further offer the Hyperspherical Margin Weighting (HMW) approach. It is a new sample weighting strategy that restructures the importance of each example. It should be highlighted that our approach is universal and can strengthen various methods in this field. Experiments on both benchmark and real-world datasets indicate that our HMW outperforms many state-of-the-art approaches in learning with noisy label tasks.

### Preparation
The experimental environment is in [requirements.txt](https://github.com/Zhangshuojackpot/Student-Loss/blob/main/requirements.txt).<br>

### Usage
Run [train_main.py](https://github.com/Zhangshuojackpot/Student-Loss/blob/main/codes_upload_real/main_lt.py) to obtain the results. For example, if you want to obtain the result of the LT-GCE loss under the noise rate of 0.2 of the symmetric noise on MNIST, you can type:<br>
```
python main_lt.py --dataset 'MNIST' --noise_type 'symmetric' --noise_rate 0.2 --is_student 1 --loss 'GCE'
```

### Citation
If you think this repo is useful in your research, please consider citing our paper.
```
@ARTICLE{10412669,
  author={Zhang, Shuo and Li, Jian-Qing and Fujita, Hamido and Li, Yu-Wen and Wang, Deng-Bao and Zhu, Ting-Ting and Zhang, Min-Ling and Liu, Cheng-Yu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Student Loss: Towards the Probability Assumption in Inaccurate Supervision}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TPAMI.2024.3357518}}
```
Meanwhile, our implementation uses parts of some public codes in [Learning With Noisy Labels via Sparse Regularization
](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Learning_With_Noisy_Labels_via_Sparse_Regularization_ICCV_2021_paper.html). Please consider citing this paper.
```
@InProceedings{Zhou_2021_ICCV,
    author    = {Zhou, Xiong and Liu, Xianming and Wang, Chenyang and Zhai, Deming and Jiang, Junjun and Ji, Xiangyang},
    title     = {Learning With Noisy Labels via Sparse Regularization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {72-81}
}
```
