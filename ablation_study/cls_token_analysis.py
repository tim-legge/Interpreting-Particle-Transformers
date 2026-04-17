
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import sys
from sklearn.decomposition import PCA
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable

from functools import partial
from weaver.utils.logger import _logger
import os
import uproot
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch._torch_docs import reproducibility_notes, sparse_support_notes, tf32_notes

import subprocess
import sys
sys.path.append('../')
import model_utils as mu

import argparse
parser = argparse.ArgumentParser(description='select the interaction parameters to run inference over')
parser.add_argument('--num-models', '-n', type=int, default=101, help='total number of models / parameter values to run inference over')
parser.add_argument('--range', '-r', type=str, required=True, help='range of the parameter values to run over (zero-indexed, starting with maximum interaction). Must input in format: <start_idx>,<stop_idx>')
parser.add_argument('--dims-to-plot', '-d', type=int, nargs='+', default=[0,1], help='which PCA dimensions to plot (zero-indexed)')
args = parser.parse_args()

num_models = args.num_models
dims_to_plot = args.dims_to_plot
idx_range = args.range.split(',')
for idx, item in enumerate(idx_range):
    idx_range[idx] = int(item)
    if int(item) >= num_models:
        idx_range[idx] = num_models-1
        raise Warning(f'index {item} is out of bounds for number of models {num_models}. Setting to {num_models-1}')
model_path = '../models/ParT_full.pt'
models = [mu.get_model('jc_full', interaction_strength=1-m_idx/(num_models-1)) for m_idx in range(num_models)]
models = models[idx_range[0]:idx_range[1]]
state_dict = torch.load(model_path, map_location='cpu')
for model in models:
    model.load_state_dict(state_dict)

cls_hooks = []
for model in models:
    cls_hooks.append(mu.ParT_Hook(model, type='class token'))
# in pod - larger data sample
dataset_size = 100000
n_jets = 1000
storage_path = '/moe-interpretability-pv/part_2d_pca_plots/'
data_path = '/moe-interpretability-pv/datasets/'
jc_full_pf_features = np.load(os.path.join(data_path, 'jc_full_pf_features.npy'))[::dataset_size//n_jets]
jc_kin_pf_features = jc_full_pf_features[:,:7,:]
jc_full_pf_vectors = np.load(os.path.join(data_path, 'jc_full_pf_vectors.npy'))[::dataset_size//n_jets]
jc_full_pf_mask = np.load(os.path.join(data_path, 'jc_full_pf_mask.npy'))[::dataset_size//n_jets]
jc_full_pf_points = np.load(os.path.join(data_path, 'jc_full_pf_points.npy'))[::dataset_size//n_jets]
jc_full_labels = np.load(os.path.join(data_path, 'jc_full_labels.npy'))[::dataset_size//n_jets]

y_pred = []
for m_idx, model in enumerate(models):
    model.eval()
    with torch.no_grad():
        y_pred.append(model(torch.from_numpy(jc_full_pf_points), torch.from_numpy(jc_full_pf_features), 
                    torch.from_numpy(jc_full_pf_vectors), torch.from_numpy(jc_full_pf_mask)))

cls_tokens = []
for cls_hook in cls_hooks:
    cls_tokens.append(cls_hook.cls_tokens)

cls_tokens_t = np.concatenate(cls_tokens, axis=0)
#print(all_cls_tokens.shape)  # should be (num_models * n_jets, hidden_dim)
# PCA components on cls tokens
pca = PCA(n_components=max(dims_to_plot)+1)
cls_tokens_t = pca.fit_transform(all_cls_tokens)
components = pca.components_
if 0 in idx_range:
    np.save('components.npy', components)
else:
    last_components = np.load('components.npy')
    # test that corresponding components are roughly collinear, swap them if not
    if np.dot(components[0], last_components[0]) < 0.5 and np.dot(components[0], last_components[1]) > 0.5:
        components[[0,1]] = components[[1,0]]
    np.save('components.npy', components)
explained_variance = pca.explained_variance_ratio_
#print('pca components shape:', components.shape)
#print("Explained Variance Ratio:\n", explained_variance)

if not os.path.exists('./pca_plots'):
    subprocess.run(['mkdir', './pca_plots'])

idx_to_label = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
color_idxs = [[f'C{i}'] for i in range(jc_full_labels.shape[1])]
labels = [idx_to_label[np.argmax(jc_full_labels[i])] for i in range(n_jets)]
pred_labels = [y_pred[m_idx][i].argmax().item() for m_idx in range(len(models)) for i in range(n_jets)]
#if not os.path.exists('./pca_plots'):
#    subprocess.run(['mkdir', './pca_plots'])

for m_idx, model in enumerate(models):
    plt.figure(figsize=(8,6))
    real_m_idx = idx_range[0] + m_idx
    for idx, label in enumerate(idx_to_label):
        mask = np.where(jc_full_labels[:,idx] == 1)[0]
        if len(mask) == 0:
            continue
        else:
            plt.scatter(cls_tokens_t[m_idx*n_jets:(m_idx+1)*n_jets,dims_to_plot[0]][mask], 
                        cls_tokens_t[m_idx*n_jets:(m_idx+1)*n_jets,dims_to_plot[1]][mask], 
                        c=color_idxs[idx]*len(mask), label=label)
        for pred_idx, pred_label in enumerate(idx_to_label):
            pred_mask = np.where(np.array(pred_labels) == pred_idx)[0]
            if len(pred_mask) == 0:
                continue
            else:
                plt.scatter(cls_tokens_t[m_idx*n_jets:(m_idx+1)*n_jets,dims_to_plot[0]][pred_mask], 
                            cls_tokens_t[m_idx*n_jets:(m_idx+1)*n_jets,dims_to_plot[1]][pred_mask], 
                            c=color_idxs[pred_idx]*len(pred_mask), s=(mpl.rcParams['lines.markersize']/4)**2)
    plt.title(f'Projection of CLS Tokens, Interaction Strength: {round(model.mod.interaction_strength,3)}')
    #plt.xlim(-5, 9)
    #plt.ylim(-6, 6)
    plt.xlabel(f'Principal Component {dims_to_plot[0]+1}')
    plt.ylabel(f'Principal Component {dims_to_plot[1]+1}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    #plt.show()
    plt.savefig(f'./pca_plots/pca_cls_tokens_model_{real_m_idx}_dims_{dims_to_plot[0]}{dims_to_plot[1]}.png')
    subprocess.run(['sudo', 'cp', f'./pca_plots/pca_cls_tokens_model_{real_m_idx}_dims_{dims_to_plot[0]}{dims_to_plot[1]}.png', storage_path])