# iSeeBetter: A Novel Approach to Video Super-Resolution using Recurrent-Generative Back-Projection Networks

Project for Stanford CS230: Deep Learning

```Python3 | PyTorch | GANs | CNNs | ResNets | RNNs```

## Required Packages

```
torch==1.3.0.post2
pytorch-ssim==0.1
numpy==1.16.4
scikit-image==0.15.0
tqdm==4.37.0
```

Also needed is [Pyflow](https://github.com/pathak22/pyflow) which is a Python wrapper for [Ce Liu's C++ implementation](https://people.csail.mit.edu/celiu/OpticalFlow/) of Coarse2Fine Optical Flow.
Pyflow binaries have been built for ubuntu and macOS and are available in the repository.
If you need to rebuild Pyflow, follow the instructions on the [Pyflow Git](https://github.com/pathak22/pyflow) and do a ```cp pyflow*.so ..``` once you have built a shared object file on your target machine.

To load,
```pip3 install -r requirements.txt```

## Overview

Recently, learning-based models have enhanced the performance of Single-Image Super-Resolution (SISR). However, applying SISR successively to each video frame leads to lack of temporal consistency. On the other hand, VSR models based on convolutional neural networks outperform traditional approaches in terms of image quality metrics such as Peak Signal to Noise Ratio (PSNR) and Structural SIMilarity (SSIM). While optimizing mean squared reconstruction error during training improves PSNR and SSIM, these metrics may not capture fine details in the image leading to misrepresentation of perceptual quality. We propose an Adaptive Frame Recurrent Video Super Resolution (AFRVSR) scheme that seeks to improve temporal consistency by utilizing information multiple similar adjacent frames (both future LR frames and previous SR estimates), in addition to the current frame. Further, to improve the “naturality” associated with the reconstructed image while eliminating artifacts seen with traditional algorithms, we combine the output of the AFRVSR algorithm with a Super-Resolution Generative Adversarial Network (SRGAN). The proposed idea thus not only considers spatial information in the current frame but also temporal information in the adjacent frames thereby offering superior reconstruction fidelity. Once our implementation is complete, we plan to show results on publicly available datasets that demonstrate that the proposed algorithms surpass current state-of-the-art performance in both accuracy and efficiency. 
 
![adjacent frame similarity](https://github.com/amanchadha/iSeeBetter/blob/master/images/iSeeBetter_AFS.jpg)
<div style="text-align:center">Figure 1: Adjacent frame similarity</div>
 
![network arch](https://github.com/amanchadha/iSeeBetter/blob/master/images/iSeeBetter_NNArch.jpg)
<center>Figure 2: Network architecture</center>

# Model Architecture

iSeeBetter uses RBPN and SRGAN as the generator and discriminator respectively. RBPN has two approaches. The horizontal flow (marked with blue arrows) enlarges LR(t) using SISR, as shown in Figure 4. The vertical flow (marked with red arrows) is based on MISR which is shown in Figure 3, and computes the residual features from a pair of LR(t) to neighbor frames (LR(t-1), ..., LR(t-n)) and the pre-computed dense motion flow maps (F(t-1), ..., F(t-n)). At each projection step, RBPN observes the missing details from LR(t) and extracts the residual features from each neighboring frame to recover those details. SISR and MISR thus extract missing details from different sources. Within the projection models, RBPN utilizes a recurrent encoder-decoder mechanism for incorporating details extracted in SISR and MISR paths through back-projection.

![ResNet_MISR](https://github.com/amanchadha/iSeeBetter/blob/master/images/ResNet_MISR.jpg)

![DBPN_SISR](https://github.com/amanchadha/iSeeBetter/blob/master/images/DBPN_SISR.png)

![Disc](https://github.com/amanchadha/iSeeBetter/blob/master/images/Disc.jpg)

## Dataset

To train iSeeBetter, we amalgamated diverse datasets with differing video lengths, resolutions, motion sequences and number of clips. Table 1 presents a summary of the datasets used. When training our model, we generated the corresponding LR frame for each HR input frame by performing 4$\times$ down-sampling using bicubic interpolation. We also applied data augmentation techniques such as rotation, flipping and random cropping. To extend our dataset further, we wrote scripts to collect additional data from YouTube, bringing our dataset total to about 170,000 clips which were shuffled for training and testing. Our training/validation/test split was 80\%/10\%/10%.

![results](https://github.com/amanchadha/iSeeBetter/blob/master/images/Dataset.jpg)

## Results

We compared iSeeBetter with six state-of-the-art VSR algorithms: DDBPN \cite{haris2018deep}, B$_{\text{123}}$ + T \cite{liu2017robust}, DRDVSR \cite{tao2017detail}, FRVSR \cite{sajjadi2018frame}, VSR-DUF \cite{jo2018deep} and RBPN/6-PF \cite{haris2019recurrent}.

![results1](https://github.com/amanchadha/iSeeBetter/blob/master/images/Res1.jpg)

![results2](https://github.com/amanchadha/iSeeBetter/blob/master/images/Res2.jpg)

![results3](https://github.com/amanchadha/iSeeBetter/blob/master/images/Res3.jpg)

## Pretrained Model
Model trained for N epochs included under ```weights/```

## Usage

# Training 

Train the model using (takes roughly 1.5 hours per epoch with a batch size of 2 on an NVIDIA Tesla V100):

```python iSeeBetterTrain.py```

# Testing

To use the pre-trained model and test on a random video from within the dataset:

```python iSeeBetterTest.py```

## Acknowledgements

Credits:
- [SRGAN Implementation](https://github.com/leftthomas/SRGAN) by LeftThomas.
- We used [RBPN-PyTorch](https://github.com/alterzero/RBPN-PyTorch) as a baseline for our Generator implementation.

## Citation
Cite the work as:
```
CVPR citation
```
