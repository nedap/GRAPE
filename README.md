# Self-Supervised Representation Learning in Point Clouds for Hierarchical Graph-Based Anatomical Structure Identification
by Niels Rocholl
Supervised by Thomas Markus and Matthia Sabatelli

Graph Representation of Advanced Part Encodings (GRAPE) is a novel framework for identifying anatomical structures in 3D point clouds of human bodies. GRAPE integrates self-supervised learning with graph neural networks to create graph-based representations of 3D objects. Each node in the graph is enriched with latent features derived from the object's parts, generated using a Masked Autoencoder (Point-MAE), which captures the geometric and spatial information of the point cloud.

<!-- I use an MAE approach, similar to [PointMAE](https://github.com/Pang-Yatian/Point-MAE) for SSL.  -->

# Environment
This code has been tested on
- PyTorch: 2.2.2
- Cuda: 12.1
- Cudnn: 8
- Python 3.11

The following image was used in our container: 
```
pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
```

# Installation steps

### python packages
The followin command should install **ALL** required packages. 

```
pip install -r requirements.txt
```

This codebase uses some custom cuda implementations to facilitate GPU acceleration of certain algorithms (FPS, KNN, Chamfer Distance). All these are included in the [requirements.txt](requirements.txt). Below you can find further information on these algorithms. 

### Chamfer Distance

This project uses a custom loss function based on the [Chamfer Distance](https://arxiv.org/pdf/1612.00603.pdf), which a method to compute the similarity between two sets of points in 3D space. This function is written in cuda and c++. Our implemtation is based on [GRNet](https://github.com/hzxie/GRNet/tree/master/extensions/chamfer_dist), credits go to Haozhe Xie.
I created a installable python package from this code, which is already included in our [requirements.txt](requirements.txt). 

However, it can also be installed manually via:

```
pip install git+https://github.com/nielsRocholl/chamfer-distance.git
```

### FPS

This project uses a c++/cuda implementation of FPS to allow for GPU acceleration of this algorithms. The original implemention can be found on [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib). 

I created a installable python package from this code, which is already included in our [requirements.txt](requirements.txt). 

However, it can also be installed manually via:

```
pip install git+https://github.com/nielsRocholl/fps-cuda.git
```

### KNN (knn_cuda)

We use the following implementation of KNN [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA) which is already installed through the [requirements.txt](requirements.txt). 

However, it can also be installed manually via:

```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

# How to Run

to train Point-MAE:
```
python main.py --config cfgs/train_ssl/train_ssl_caesar.yaml --wandb
```

to generate the Graph Dataset using the trained Point-MAE:
```
python main.py --config cfgs/build_gnn_ds/build_gnn_ds_ceasar.yaml --build_gnn_ds
```

to train the Graph Neural Network for node classification:
```
python main.py --config cfgs/train_gnn/train_gnn_caesar.yaml --train_gnn
```

## How to run specific experiments

run embedding noise experiment:
```
python main.py --config cfgs/noised_embedding_exp.yaml --train_gnn --k_fold --wandb
```

## How to build the default graph:

```
python main.py --config cfgs/default_graph/build_default_graph_ceasar.yaml --build_default_graph
```

### General Project Information.

We try to produce high quality, understandable code. Key features included in this repository are:

- **Configuration Management**: Utilizes YAML files for configuration management.
- **Experiment Tracking**: Incorporates Weights and Biases for precise experiment tracking.
- **Configuration Loaders and Writers**: Facilitates easy management of configuration data.
- **Custom Logger**: A logger designed to support distributed computing environments.
- **Custom Argument Parser**: A argument parser to streamline command-line interactions.
- **Class Registry**: A registry to effortlessly map strings to classes.
- **Dataset Class**: A dataset class to handle data operations seamlessly.
- **IO Class**: A dedicated IO class to manage data input and error handling efficiently.

Each function within this project is documented with comprehensive docstrings, comments, and proper typing. The primary adjustment required post-setup is updating the Weights and Biases Project name within the `main.py` file. In addition to example text in the configuration files. 

The project structure is illustrated below:

```plaintext
toy-problem/
│
├── DATASET.md
├── README.md
├── cfgs
│   ├── dataset_cfgs
│   │   └── part_net.yaml
│   └── train.yaml
│
├── dataset
│   ├── build.py
│   ├── io.py
│   └── part_net.py
│
├── experiments
│   └── train
│       └── cfgs
│           ├── TFBoard
│           │   └── experiment
│           └── experiment
│               └── wandb
│
├── main.py
├── models
│
├── output
│   ├── figures
│   └── trained-models
│
├── requirements.txt
├── tools
│   ├── model_tester.py
│   └── model_trainer.py
│
└── utils
    ├── config.py
    ├── logger.py
    ├── misc.py
    ├── parser.py
    └── registry.py
```


### Troubleshooting Errors

If you encounter the following error:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

Then run:

```
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```
