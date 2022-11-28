![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![APM](https://img.shields.io/apm/l/python?color=%21%5BAPM%5D%28https%3A%2F%2Fimg.shields.io%2Fapm%2Fl%2Fpython%3Fstyle%3Dfor-the-badge%29&logo=%21%5BAPM%5D%28https%3A%2F%2Fimg.shields.io%2Fapm%2Fl%2Fpython%3Fstyle%3Dfor-the-badge%29&logoColor=%21%5BAPM%5D%28https%3A%2F%2Fimg.shields.io%2Fapm%2Fl%2Fpython%3Fstyle%3Dfor-the-badge%29&style=for-the-badge)
# Pytorch-Ray-Tune

Adjusting hyperparameters can make the difference between an average model and a highly accurate model. For instance, simple things like choosing a different learning rate or changing the size of a network layer can have a dramatic impact on the performance of your model.

Luckily, there are tools that help find the best combination of parameters. Ray Tune is an innovative industry standard tool for tuning distributed hyperparameters. Ray Tune includes the latest hyperparameter search algorithms, integrates with TensorBoard and other analysis libraries, and natively supports distributed training with Ray's distributed machine learning engine.

- Hyperparameter Optimization Checklist:
  - Manual Search.
  - Grid Search.
  - Randomized Search.
  - Halving Grid Search.
  - Halving Randomized Search.
  - HyperOpt-Sklearn.
  - Bayes Search.

Claim: This repository will be maintained, so that it can eventually be used for [Neuralprophet](https://neuralprophet.com/) too (Stanford, Meta tools). 
## What is Ray?

Ray is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a toolkit of libraries (Ray AIR) for simplifying ML compute:

<img src="https://github.com/ray-project/ray/raw/master/doc/source/images/what-is-ray-padded.svg" alt="what-is-ray">

&nbsp;

Learn more about [Ray AIR](https://docs.ray.io/en/latest/ray-air/getting-started.html) and its libraries.

Ray runs on any machine, cluster, cloud provider, and Kubernetes, and features a growing
[ecosystem of community integrations](https://docs.ray.io/en/latest/ray-air/getting-started.html).

## Why Ray?
Today's ML workloads are increasingly compute-intensive. As convenient as they are, single-node development environments such as your laptop cannot scale to meet these demands.
Ray is a unified way to scale Python and AI applications from a laptop to a cluster.
With Ray, you can seamlessly scale the same code from a laptop to a cluster. Ray is designed to be general-purpose, meaning that it can performantly run any kind of workload. If your application is written in Python, you can scale it with Ray, no other infrastructure required.

