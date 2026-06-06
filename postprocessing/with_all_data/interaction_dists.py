# This script generated the final subjet histograms in the plot

from typing import List, Optional
import timeit
import awkward as ak
import torch
import torch.nn as nn
from torch.nn import Parameter 
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
from typing import Optional, Tuple
_is_fastpath_enabled: bool = True
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
linear = torch._C._nn.linear
import math
import random
import warnings
import copy
from torch._C import _add_docstr, _infer_size

from functools import partial
from weaver.utils.logger import _logger
import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
import numpy as np
from tqdm.auto import tqdm
import subprocess
import argparse
import logging

import sys
sys.path.append('/home/jovyan/Interpreting-Particle-Transformers/')
import model_utils as mu

fontsize = 20

parser = argparse.ArgumentParser(description='Inference for interaction distributions.')
parser.add_argument('--dataset', '-d', type=str, default='jc_full', help='Dataset name (jc_full, jc_kin, jc_kinpid)')
parser.add_argument('--chunk', '-c', type=int, help='chunk number')
parser.add_argument('--num-chunks', '-n', type=int, default=10, help='total number of chunks')
parser.add_argument('--restart', '-r', action='store_true', help='Whether to restart the job from scratch, or continue from the last counter')
parser.add_argument('--plot', '-p', action='store_true', help='To plot instead of run inference')
args = parser.parse_args()

dataset = args.dataset
chunk = args.chunk
num_chunks = args.num_chunks
restart = args.restart
plot_q = args.plot

base_dir = '/moe-interpretability-pv/'
dataset_path = base_dir+'datasets/'
storage_path = base_dir+f'ParT_{dataset}_interaction_hists/'
counter_path = storage_path + f'chunk_{chunk}_counter.txt'
total_jets = int(10e5)
start_jet = counter = chunk*(total_jets//num_chunks)

if not os.path.exists(counter_path) or restart:
    subprocess.run(['sudo', 'mkdir', '-p', storage_path])
    with open('counter.txt', 'w') as f:
        f.write(str(start_jet))
    subprocess.run(['sudo', 'cp', 'counter.txt', counter_path])
else:
    subprocess.run(['sudo', 'cp', counter_path, 'counter.txt'])
    with open(counter_path, 'r') as f:
        counter = int(f.read().strip())

model = mu.get_model(dataset)
num_heads = model.num_heads
num_layers = model.num_layers
model_path = f'/home/jovyan/Interpreting-Particle-Transformers/models/ParT{dataset.split("_")[1]}.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu'))

def flatten_deep(iterable):
    for item in iterable:
        if isinstance(item, (list, tuple)):
            yield from flatten_deep(item)
        elif isinstance(item, torch.Tensor):
            output = item.numpy().flatten()  # Convert tensor to numpy array and flatten
            output = np.abs(output)  # Take absolute value of the flattened array
            yield from output  # Yield each element of the flattened array

kin_slice = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1], dtype=bool)
kinpid_slice = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1], dtype=bool)

if dataset == 'jc_kin':
    jc_full_pf_features = np.load(dataset_path+'jc_full_pf_features.npy')[:,kin_slice,:]
elif dataset == 'jc_kinpid':
    jc_full_pf_features = np.load(dataset_path+'jc_full_pf_features.npy')[:,kinpid_slice,:]
else:
    jc_full_pf_features = np.load(dataset_path+'jc_full_pf_features.npy')
jc_full_pf_vectors = np.load(dataset_path+'jc_full_pf_vectors.npy')
jc_full_pf_mask = np.load(dataset_path+'jc_full_pf_mask.npy')
jc_full_pf_points = np.load(dataset_path+'jc_full_pf_points.npy')
jc_full_labels = np.load(dataset_path+'jc_full_labels.npy')
howmanyjets = 500

while counter < start_jet + (total_jets//num_chunks) and not plot_q:
    jc_pf_features = torch.from_numpy(jc_full_pf_features[counter:counter+howmanyjets])
    jc_pf_vectors = torch.from_numpy(jc_full_pf_vectors[counter:counter+howmanyjets])
    jc_pf_mask = torch.from_numpy(jc_full_pf_mask[counter:counter+howmanyjets])
    jc_pf_points = torch.from_numpy(jc_full_pf_points[counter:counter+howmanyjets])
    jc_labels = torch.from_numpy(jc_full_labels[counter:counter+howmanyjets])

    hooks = mu.ParT_Hook(model)
    model.eval()
    with torch.no_grad():
        _ = model(jc_pf_features, jc_pf_vectors, jc_pf_mask, jc_pf_points)
        logging.info(f"Processed jets {counter} to {counter+howmanyjets} for chunk {chunk}")
    
    pad_limits = hooks.cut_padding(hooks.pre_softmax_attentions, jc_pf_mask)
    pre_softmax_attn = hooks.pre_softmax_attentions
    pre_softmax_inter = hooks.pre_softmax_interactions

    softmax_qk_attn = [[[torch.softmax(pre_softmax_attn[l][i][j, :pad_limits[i], :pad_limits[i]], dim=-1) for j in range(len(pre_softmax_attn[0][0]))] for i in range(len(pre_softmax_attn[0]))] for l in range(num_layers)]
    softmax_full_attn = [[[torch.softmax(pre_softmax_attn[l][i][j, :pad_limits[i], :pad_limits[i]] + pre_softmax_inter[0][i][j, :pad_limits[i], :pad_limits[i]], dim=-1) for j in range(len(pre_softmax_inter[0][0]))] for i in range(len(pre_softmax_inter[0]))] for l in range(num_layers)]
    diff_maps = [[[(softmax_full_attn[l][i][j] - softmax_qk_attn[l][i][j]) for j in range(len(pre_softmax_inter[0][0]))] for i in range(len(pre_softmax_inter[0]))] for l in range(num_layers)]

    flattened_contribution = list(flatten_deep(diff_maps))  # Flatten the nested list of tensors into a single list of values
 
    # Define number of bins for the probability distribution
    num_bins = 20
    # Define the bin edges between 0 and 1, using 20 evenly spaced bins
    bin_edges = np.linspace(0, 1, num_bins + 1)

    batch_hist, _ = np.histogram(flattened_contribution, bins=bin_edges)
    hist_path = storage_path + f'{counter}_to_{counter+howmanyjets}_hist.npy'
    np.save(hist_path, batch_hist)
    counter += howmanyjets
    with open('counter.txt', 'w') as f:
        f.write(str(counter))
    subprocess.run(['sudo', 'cp', 'counter.txt', counter_path])
    logging.info(f"Saved histogram for chunk {chunk} at {hist_path}")

all_hist = None

if plot_q:
    for file in sorted(os.listdir(storage_path)):
        if file.endswith('hist.npy'):
            file_path = os.path.join(storage_path, file)
            hist = np.load(file_path)
            if 'all_hist' is not None:
                all_hist = hist.astype(np.float64)
            else:
                all_hist += hist

if all_hist is None:
    print("No histograms found.")
else:
    # Normalize to probability distribution
    probabilities = all_hist / all_hist.sum()

    # Create bin edges (assuming scores are between 0 and 1)
    num_bins = len(probabilities)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    equal_width = bin_edges[1] - bin_edges[0]

    # Plot in your preferred format
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    #ax.bar(bin_centers, probabilities, width=equal_width, log=False, edgecolor="black")
    ax.hist(probabilities, bins=bin_edges, edgecolor="black", log=True)
    ax.set_xlabel("Interaction Contribution", fontsize=fontsize)
    ax.set_ylabel("Probability", fontsize=fontsize)
    plt.yscale("log")
    # Save if needed:
    plt.savefig(f"Interaction_dist_{dataset}.pdf", bbox_inches="tight")
