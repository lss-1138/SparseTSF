# SparseTSF

Welcome to the official repository of the SparseTSF paper: "[SparseTSF: Modeling Long-term Time Series Forecasting with *1k* Parameters](https://arxiv.org/pdf/2405.00946)"

## Updates
🚩 **News** (2024.06) SparseTSF paper has been selected for an **_Oral_ presentation at ICML 2024**.

🚩 **News** (2024.05) SparseTSF has been accepted as a paper at **_ICML 2024_**, receiving an **average rating of 7 with confidence of 4.5**.

## Introduction
SparseTSF is a novel, extremely lightweight model for Long-term Time Series Forecasting (LTSF).
At the heart of SparseTSF lies the **Cross-Period Sparse Forecasting** technique, which simplifies the forecasting task by decoupling the periodicity and trend in time series data.

Technically, it first downsamples the original sequences with constant periodicity into subsequences, then performs predictions on each downsampled subsequence, simplifying the original time series forecasting task into a cross-period trend prediction task. 

![image](Figures/Figure2.png)

Intuitively, SparseTSF can be perceived as a sparsely connected linear layer performing sliding prediction across periods

![image](Figures/Figure5.png)

This approach yields two benefits: (i) effective decoupling of data periodicity and trend, enabling the model to stably identify and extract periodic features while focusing on predicting trend changes, and (ii) extreme compression of the model's parameter size, significantly reducing the demand for computational resources.

![img.png](Figures/Table2.png)

SparseTSF achieves near state-of-the-art prediction performance with less than **_1k_** trainable parameters, which makes it **_1 ~ 4_** orders of magnitude smaller than its counterparts.

![img.png](Figures/Table3.png)

Additionally, SparseTSF showcases remarkable generalization capabilities (cross-domain), making it well-suited for scenarios with limited computational resources, small samples, or low-quality data.

![img.png](Figures/Table7.png)

## Getting Started

### Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n SparseTSF python=3.8
conda activate SparseTSF
pip install -r requirements.txt
```

### Data Preparation

All the datasets needed for SparseTSF can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. 
Create a separate folder named ```./dataset``` and place all the CSV files in this directory.
**Note**: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"

### Training Example

You can easily reproduce the results from the paper by running the provided script command. For instance, to reproduce the main results, execute the following command:

```
sh run_all.sh
```

Similarly, you can specify separate scripts to run independent tasks, such as obtaining results on etth1:

```
sh scripts/SparseTSF/etth1.sh
```

## Usage on Your Data

SparseTSF relies on the inherent periodicity in the data. If you intend to use SparseTSF on your data, please first ascertain **whether your data exhibits periodicity**, which can be determined through ACF analysis. 

We provide an example in the [ACF_ETTh1.ipynb](https://github.com/lss-1138/SparseTSF/blob/main/ACF_ETTh1.ipynb) notebook to determine the primary period of the ETTh1 dataset. You can utilize it to ascertain the periodicity of your dataset and set the `period_len` parameter accordingly.

It is important to note a special case where the dataset's period is excessively large. For instance, in ETTm1, due to dense sampling, its period is 144. Resampling with too large a period results in very short subsequences with sparse connections, leading to underutilization of information. In such cases, setting `period_len` to [2-6], i.e., adopting a denser sparse strategy, can be beneficial. For more details, refer to the discussion in Appendix C.2.
## Further Reading

The objective of this work is to explore an **ultra-lightweight** yet sufficiently powerful method to be applicable in edge scenarios with limited resources and small datasets for transfer learning and generalization. 

If you seek higher predictive performance, we recommend our alternative work, **[SegRNN](https://github.com/lss-1138/SegRNN)**, which is an innovative RNN-based model specifically designed for LTSF. By integrating Segment-wise Iterations and Parallel Multi-step Forecasting (PMF) strategies, SegRNN achieves state-of-the-art results with just a single layer of GRU, making it extremely lightweight and efficient.

## Citation
If you find this repo useful, please cite our paper.
```
@article{lin2024sparsetsf,
  title={SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters},
  author={Lin, Shengsheng and Lin, Weiwei and Wu, Wentai and Chen, Haojun and Yang, Junjie},
  journal={arXiv preprint arXiv:2405.00946},
  year={2024}
}
```


## Contact
If you have any questions or suggestions, feel free to contact:
- Shengsheng Lin ([linss2000@foxmail.com]())
- Weiwei Lin ([linww@scut.edu.cn]())
- Wentai Wu ([wentaiwu@jnu.edu.cn]())

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

https://github.com/lss-1138/SegRNN

https://github.com/VEWOXIC/FITS

https://github.com/yuqinie98/patchtst

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai


