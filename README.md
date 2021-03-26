# Cal-FLOPs-for-PLM
Calculating FLOPs of Pre-trained Models in NLP

This repository provides example script about calculating the FLOPs and Parameters for NLP models (mainly PLMs) in Pytorch Framework.

The example script exhibits the usage of two types of open-source FLOPs counter tools.

The FLOPs counter and Parameters counter are based on the open-source tool of [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) and [THOP: PyTorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)

## Install

From PyPI:

```
pip install thop
pip install ptflops
```

Requirements: 
* Pytorch >= 1.0.0
* transformers >= 2.2.0

## Example

see `example.py`

