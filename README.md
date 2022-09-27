# Scalable Knowledge Enhancement of Graph Neural Networks

This repository contains our experiments conducted with [Knowledge Enhanced Neural Network](https://arxiv.org/abs/2009.06087) [1] on the datasets ogbn-arxiv and ogbn-products from the [Open Graph Benchmark](https://ogb.stanford.edu/) [2]. In order to make KENN feasible on large graphs, we propose a graph sampling method called Restrictive Neighbourhood Sampling as graph-specific mini-batching method that allows to control the space complexity. We use [Weights&Biases](https://wandb.ai/site) [3] as a tool to keep track of our experiments. Our implementation is based on the Graph Neural Network Framework [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) [4].

This is a work of the [Tyrex Team](https://tyrex.inria.fr/). 
In case of comments or questions, feel free to [email us](luisa.werner@inria.fr).

KENN has been developed by Alessandro Daniele, Riccardo Mazzieri and Luciano Serafini. 

## Overview 
Knowledge Enhanced Neural Networks (KENNs) [1] integrate prior knowledge in the form of logical formulas into an Artificial Neural Network by adding Knowledge Enhancement (KE) layers to the network architecture. Previous results show that the model outperforms pure neural models as well as Neural-Symbolic models on small graphs but struggle to extend to larger ones. In this paper, we address the problem of knowledge enhancement of neural networks on large graphs and carry out experiments on the Open Graph Benchmark datasets (OGB) [2]. When dealing with large graphs, we show that neighbourhood explosion occurs and makes the full-batch training of the model unfeasible. To solve this problem, we first analyse the space complexity of the knowledge enhancement layers and propose a graph-specific mini-batching strategy to make it applicable to large-scale graphs. To show that our method is effective, we test our model on two datasets from the Open Graph Benchmark datasets.


## How to run the code 
1. In order to make sure that the right environment is used, the necessary Python packages and their versions are specified in `requirements.txt`. We tested our implementation on Python 3.9. 
To install the requirements, go in the project directory and run the command
```
pip install -r requirements.txt
```

2. For tracking the experiments, a free Weights&Biases account is required. Sign up for Weights&Biases at [https://wandb.ai/site](https://wandb.ai/site) and login with the command
```
wandb login 
``` 
Follow the instructions in the command line. 

Then, adapt the `project` and `entity` parameters in `wandb.init(...)` in `run_experiments.py` according to your project.
More instructions on Weights&Biases can be found [here](https://docs.wandb.ai/quickstart). 
In the following, all conducted experiments, their results and their hyperparameters will be tracked on Weights&Biases.

3. To run experiments, run the following command from the project directory. The parameters specified in `conf.json` are used.  
``` python run_experiments.py conf.json```

4. To change the parameters, the `conf.json` file can be modified. 

| Parameter             |  Description                                                                        | Default Value        |
| ----------------------|:------------------------------------------------------------------------------------| --------------------:|
| datasets              | Dataset to be used. Can be ogbn-products, ogbn-arxiv, Cora, CiteSeer or PubMed.     | ogbn-arxiv           |
| planetoid_split       | The type of dataset split for Planetoid data sets, see [4]                          | public               |
| sampling_neighbor_size| Number of neighbours to be sampled per step in sampling depth                       | -1 (all)             |
| batch_size            | Number of target nodes per batch                                                    | 10.000               |
| num_kenn_layers       | Number of KENN layers                                                               | 3                    |
| num_layers_sampling   | Sampling depth, describes n-hop neighbourhood to sample from                        | 3                    |
| hidden channels       | Number of hidden units in Base NN                                                   | 256                  |
| dropout               | Dropout rate                                                                        | 0.5                  |
| lr                    | Learning rate                                                                       | 0.01                 |
| epochs                | Number of epochs                                                                    | 300                  |
| runs                  | Number of independent runs                                                          | 10                   |
| model                 | model: "GCN", "MLP", "KENN_GCN", "KENN_MLP"                                         | "KENN_MLP"           |
| mode                  | training mode: transductive or inductive                                            | transductive         |
| binary preactivations | Artificial preactivations of binary predicates                                      | 500.0                |
| es_enabled            | Early Stopping enabled : True or False                                              | True                 |
| es_min_delta          | Early Stopping Minimum Delta                                                        | 0.001                |
| es_patience           | Early Stopping Patience                                                             | 10                   |
| full_batch            | Enable full-batch training: True or False                                           | False                |
| num_workers           | Number of parallel workers for NeighborLoader                                       | 0                    |
| seed                  | Random Seed                                                                         | 0                    |
| train_sampling        | How to sample the training batches: here only restrictive neighbourhood sampling    | "default"            |
| eval_steps            | How often to evaluate in the training loop                                          | 10                   |
| save_data_stats       | Save an overview of dataset stats as test file: True or False                       | False                |
| create_kb             | Create knowledge as defined in [1] if set to True, False: use knowledge_base        | True                 |
| knowledge_base        | Custom knowledge base                                                               | ""                   |



# References 
[1] A. Daniele, L. Serafini, Neural Network Enhancement with Logical Knowledge, 2020. URL: [https://arxiv.org/abs/2009.06087](https://arxiv.org/abs/2009.06087). doi:10.48550/ ARXIV.2009.06087.

[2] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. URL: [https://arxiv.org/abs/2005.00687](https://arxiv.org/abs/2005.00687), 2020. 10

[3] Lukas Biewald. Experiment tracking with weights and biases, 2020. URL [https://www.wandb.com/](https://www.wandb.com/). Software available from wandb.com. 8.

[4] M. Fey, J. E. Lenssen, Fast graph representation learning with PyTorch Geometric, in: ICLR Work- shop on Representation Learning on Graphs and Manifolds, 2019. 

We make use of the [KENN implementation](https://github.com/HEmile/KENN-PyTorch) (in PyTorch) and the [example baselines for OGB](https://github.com/snap-stanford/ogb), both publicly available on GitHub. The Software of OGB and KENN is licensed. 

