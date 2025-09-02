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

Our workflow is sorted into 3 main stages - [preprocessing](./preprocessing/), training, and [postprocessing](./postprocessing/). 

### Preprocessing

In [preprocessing](./preprocessing/), we store the scripts used to convert datasets between relevant formats (HDF5, ROOT, parquet). These are used to prepare data either for the training stage (in which .parquet format was used) or the postprocessing stage (in which ROOT was used).
- Note: This is not necessary if you only desire to reproduce results, as there already exist both trained models and sample test data in their respective folders.

### Training

As described above, training is done using weaver. This demands two kinds of files: network configuration files (under [networks](./networks/)) and data configuration files (under [data](./data/)). Their structures are more or less self-explanatory. The `train.sh` files call one of each to pass to the weaver training algorithm.

### Postprocessing

Postprocessing comprises the data collection and visualization steps after a model has been fully trained. We produced the exact figures in the paper with the scripts in the [with_all_data](./postprocessing/with_all_data/) subfolder, while a much less intensive version utilizing sample data is in [with_sample_data](./postprocessing/with_sample_data/).

- [with_all_data](./postprocessing/with_all_data/): The `inference.py` scripts all have roughly the same structure. They check to see if `.npy` files that contain testing data exist, and generate them if not. Then they run inference on chunks of each dataset at a time, saving their progress using rudimentary `counter.txt` files. Finally, they create 20-bin histogram .npy files for later use in the attention score plotting. Note that `/path/to/storage` was used as a universal placeholder and unilaterally replacing this with one's own desired location may not result in consistent filepaths. The `get_[figure].py` files either use these histogram files to plot attentioh distributions or otherwise represent data using the testing data `.npy` files.

- [with_sample_data](./postprocessing/with_sample_data/): QuarkGluon/TopLandscape analysis and JetClass full/kinematic analysis each have their own `.ipynb` notebook. Both load limited chunks of the sample data (up to ~50 jets should work locally in our experience). The same analyses are performed, with the exception that JetClass' sample data in this case does not include the specific jets used for the Hadronic and Leptonic top decay figures in the paper (in the larger JetClass_example_100k.root file, they occur at indices 60,003 p/m 1 and 50,016 p/m 1).