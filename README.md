# MTGNN-RCBI
This is an adaptation of the PyTorch implementation of the paper: [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650), published in KDD-2020, but adding the uncertainty estimation using Bootstrap frequentist and Sensitivity Analysis with confidence intervals, also with Variance Calculation.

The Bayesian Network implementation, as well as the k-fold Cross Validation, don't work properly as they don't come to a solution, so it is not recommended to use them.

In the ```forward``` method inside the ```graph_constructor``` class defined in ```layer.py```, there are different adjacency matrix calculation approaches, as it was modified to study a potential weak point of the model.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Preparation
### Multivariate time series datasets

Fromt the Solar-Energy, Traffic, Electricity, Exchange-rate datasets extracted from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data), the Electricity demand one is the only one used in this repository.

## Model Training

Follow the instructions and commands used in the MTGNN.ipynb Notebook, adapted to use the files in this repository, using platforms like Google Colab (mainly recommended) or Jupyter Notebooks.

## Citation

```
@inproceedings{wu2020connecting,
  title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```
