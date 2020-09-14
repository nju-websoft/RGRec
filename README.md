# Rule-Guided Graph Neural Networks for Recommender Systems
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/nju-websoft/RGRec/issues)
[![License](https://img.shields.io/badge/License-GPL-lightgrey.svg?style=flat-square)](https://github.com/nju-websoft/RGRec/blob/master/LICENSE)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)

> To alleviate the cold start problem caused by collaborative ﬁltering in recommender systems, knowledge graphs (KGs) are increasingly employed by many methods as auxiliary resources. However, existing work incorporated with KGs cannot capture the explicit long-range semantics between users and items meanwhile consider various connectivity between items. In this paper, we propose RGRec, which combines rule learning and graph neural networks (GNNs) for recommendation. RGRec ﬁrst maps items to corresponding entities in KGs and adds users as new entities. Then, it automatically learns rules to model the explicit long-range semantics, and captures the connectivity between entities by aggregation to better encode various information. We show the eﬀectiveness of RGRec on three real-world datasets. Particularly, the combination of rule learning and GNNs achieves substantial improvement compared to methods only using either of them.

## Getting Started

### Package Description

```
src/
├── aggregators.py: a deprecated module whose methods have been moved into RGRec_model.py
├── ALogger.py: log management module
├── Args.py: global argument module
├── auc_f1_main.py: train module, and a part of test module which tests auc and f1
├── Graph.py: data structure module 
├── hit_ndcg_main.py: test module which involves testing hits@k and ndcg@k
├── pra_model.py: rule weights pre-training module
├── RGRec_model.py: the core module to implement RGRec
├── Sundries.py: provide a simple function
```

### Dependencies

* Python 3.x (tested on Python 3.6)
* Pytorch 1.x (tested on Pytorch 1.3)
* Numpy
* Scipy
* Sklearn

### Usage

1. Set your arguments in Args.py.
2. Run pra_model.py to get the pre-trained weights of rules.
3. Run auc_f1_main.py to train the model and obtain the test results about auc and f1.
4. Run hit_ndcg_main.py to obtain the test results about hits@k and ndcg@k.

## Experiment

We conduct experiments on three real-world datasets: [Last.FM](https://github.com/hwwang55/KGCN/tree/master/data/music), [MovieLens-1M](https://github.com/hwwang55/RippleNet/tree/master/data/movie) and [Dianping-Food](https://github.com/hwwang55/KGNN-LS/tree/master/data/restaurant). Currently, our project only provides MovieLens-1M, other datasets can be downloaded [here](https://1drv.ms/u/s!AqA_HIF7WyFosB3mBny-Zda23ZL3?e=GfCIOv).

The details of experiments can be found in the corresponding section of the paper.


## License

This project is licensed under the GPL License - see the [LICENSE](https://github.com/nju-websoft/RGRec/blob/master/LICENSE) file for details.

## Miscellaneous

During the construction of RGRec, we obtain much inspiration from [KGCN](https://github.com/hwwang55/KGCN), and we feel a great honor to make an innovation in the foundation of [this work](https://github.com/hwwang55/KGCN).

## Citation

If you use this work or code, please kindly cite the following paper:

```
@inproceedings{RGRec,
  author    = {Xinze Lyu and Guangyao Li and Jiacheng Huang and Wei Hu},
  title     = {Rule-Guided Graph Neural Networks for Recommender Systems},
  booktitle = {ISWC},
  year      = {2020},
}
```

## Contacts

If you have any questions, please feel free to contact [Guangyao Li](mailto:gyli.nju@gmail.com), we will reply it as soon as possible.

