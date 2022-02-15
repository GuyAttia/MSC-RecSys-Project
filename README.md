# RecSys - Final Project - Sofia & Guy
In this project we have implemented 3 different models - Matrix Factorization (MF), Auto-Encoders (AutoRec) and Variational-Auto-Encoders (VAE).
We evaluate them on 2 different datasets - Movielens 1M and Books.
* We have used the <b>Pytorch-Ignite</b> package for cleaner and better implementation of our models trainers.
* We have used the <b>Optuna</b> package for cleaner and more stable hyperparameters tuning.

## Running Instructions
We split our different models in 3 different running notebooks for better parallelism and dependencies. In each notebook you can find 2 running pipelines, 1 for each dataset.
The pipeline steps are:
1. Define datasets hyperparameters (for full definitions there is a need in hard code editing the "src/hyperparameters_tuning.py" code).
2. Hyperparameters tuning
3. Full and final train using the tuned parameters for last validation on the test set.

For your convinient you can choose whether to run it on your local jupyter notebook or use Google Colab.
* We have used the <b>tensorboard</b> capabilities for logging the loss and evaluation metrices during the training and evaluation of our networks. In order to explore your results, use the following lines inside your notebook:
```
%load_ext tensorboard
%tensorboard --logdir=.
```

### Local Jupyter Notebook
- We suggest to run our code using the miniconda package manager
- First you have to install all the conda requirements, you can use the conda.yaml file in this repository.
- Open the relevant notebook in your jupyter-notebook
- Skip the "Colab Preparations" code blocks
- Select the run parameters and execute the rest of the notebook code blocks

### Google Colab Notebook
- Open the relevant notebook in your Google Colab environment
- Execute the "Colab Preparations" code blocks
- Select the run parameters and execute the rest of the notebook code blocks
