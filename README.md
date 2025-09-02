# Interpreting Particle Transformer

This repo contains the code used for experiments in 'Why is Attention Sparse in Particle Transformer?' This repo contains the scripts used to run analyses on the Particle Transformer (ParT) and sample data (~500 jets each) from each of the datasets referenced in the paper. What follows is a description of the workflow used to produce figures and some guidance, should others desire to reproduce results.

## Note on Dependencies

Our research utilizes a few specialized High-Energy Physics (HEP) Python packages, summarized in [packages.txt](./packages.txt). Most packages will be familiar to practicioners of ML.

## Training Models

We trained models using the [Weaver](https://github.com/hqucms/weaver-core) framework in following with descriptions from [the official ParT repo](github.com/jet-universe/particle_transformer/blob/main/README.md). The models used for experiment are stored under [models](./models/). To train a ParT model from scratch, one can first download the dataset they wish to train on:

```
./get_datasets.py [JetClass|QuarkGluon|TopLandscape] [-d DATA_DIR]
```
- Note: While QuarkGluon and TopLandscape are not difficult to download fully in a local environment (each ~2.5 GB), JetClass is extremely large by comparison (~190 GB). One is recommended to either host an experiment with JetClass remotely or use a pretrained model.

Then, run the desired `train.sh` script. For example, to run a training on QuarkGluon's full feature set (kinpid):
```
./train_QuarkGluon.sh ParT kinpid [options]
```

Additional details are available at [the official ParT repo](github.com/jet-universe/particle_transformer/blob/main/README.md).

## Workflow & Reproducing Results

add tmrw