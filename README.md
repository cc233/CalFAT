# CalFAT: Calibrated Federated Adversarial Training on Non-IID Data
This repository provides codes for NeurIPS 2022 paper [CalFAT: Calibrated Federated Adversarial Training with Label Skewness](https://arxiv.org/abs/2205.14926).




# Running the code
The code can be run as follows.

```shell
python3 fat.py --epochs=150 --local_ep=1 --lr=0.01 --dataset=cifar10  --beta=0.1  --num_users=5
```
| Parameter | Description | 
| :-:| :-: |
| dataset | Dataset to use|
| epochs| The total communication rounds|
|local_ep| The local training epochs|
|beta| The concentration parameter of the Dirichlet distribution for heterogeneous partition|
|num_users| Number of clients|
|lr| Learning rate|

# Reference
```
@inproceedings{chen2022calfat,
  title={CalFAT: Calibrated Federated Adversarial Training with Label Skewness},
  author={Chen, Chen and Liu, Yuchen and Ma, Xingjun and Lyu, Lingjuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```