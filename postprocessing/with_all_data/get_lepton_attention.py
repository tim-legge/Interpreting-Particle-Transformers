# This script generated the final lepton histograms in the plot

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

from model_utils import *

parser = argparse.ArgumentParser(description='Lepton job specs.')
parser.add_argument('--class-to-analyze', '-a', type=str, help='Class to analyze (Hqql, Tbl)')
parser.add_argument('--chunk', '-c', type=int, help='chunk number')
parser.add_argument('--num-chunks', '-n', type=int, default=10, help='total number of chunks')
parser.add_argument('--restart', '-r', action='store_true', help='Whether to restart the job from scratch, or continue from the last counter')
args = parser.parse_args()

class_to_analyze = args.class_to_analyze
chunk = args.chunk
num_chunks = args.num_chunks
restart = args.restart

jc_full_model = get_model(model_type='jc_full', return_pre_softmax=True)

jc_full_hooks = Pre_Softmax_Hook(model=jc_full_model)

classes = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
start_indices = np.array([8, 0, 1, 2, 4, 3, 9, 7, 6, 5]) * 200000
total_jets = 200000

start_jet = counter = start_indices[classes.index(class_to_analyze)] + chunk*(total_jets//num_chunks)
assert class_to_analyze in ['Hqql', 'Tbl'], 'to get lepton attention plots, please specify class as Hqql or Tbl'

base_dir = '/moe-interpretability-pv/'

howmanyjets = 500

dataset_path = base_dir+'datasets/'
storage_path = base_dir+f'ParT_{class_to_analyze}_hists/'
counter_path = storage_path + f'chunk_{chunk}_counter.txt'

if not os.path.exists(counter_path) or restart:
    subprocess.run(['sudo', 'mkdir', '-p', storage_path])
    with open('counter.txt', 'w') as f:
        f.write(str(start_jet))
    subprocess.run(['sudo', 'cp', 'counter.txt', counter_path])
else:
    subprocess.run(['sudo', 'cp', counter_path, 'counter.txt'])
    with open(counter_path, 'r') as f:
        counter = int(f.read().strip())

jc_kin_lepton_attention = get_model('jck')

jc_full_pf_features = np.load(dataset_path+'jc_full_2M_features_0.npy')
jc_full_pf_vectors = np.load(dataset_path+'jc_full_2M_vectors.npy')
jc_full_pf_mask = np.load(dataset_path+'jc_full_2M_mask.npy')
jc_full_pf_points = np.load(dataset_path+'jc_full_2M_points.npy')
jc_full_labels = np.load(dataset_path+'jc_full_2M_labels.npy')

while counter < start_jet + (total_jets//num_chunks):
    jc_pf_features = jc_full_pf_features[counter:counter+howmanyjets]
    jc_pf_vectors = jc_full_pf_vectors[counter:counter+howmanyjets]
    jc_pf_mask = jc_full_pf_mask[counter:counter+howmanyjets]
    jc_pf_points = jc_full_pf_points[counter:counter+howmanyjets]
    jc_labels = jc_full_labels[counter:counter+howmanyjets]

    jc_kin_lepton_attention_hooks = Pre_Softmax_Hook(model=jc_kin_lepton_attention)
    init_lepton_attention = get_model('jck')
    init_lepton_attention_hooks = Pre_Softmax_Hook(model=init_lepton_attention)

    init_lepton_attention.eval()
    with torch.no_grad():
        init_pred = init_lepton_attention(torch.from_numpy(jc_pf_points),
                                    torch.from_numpy(jc_pf_features[:,0:7,:]),
                                    torch.from_numpy(jc_pf_vectors),torch.from_numpy(jc_pf_mask))

    jc_kin_lepton_attention.eval()
    with torch.no_grad():
        jck_y_pred= jc_kin_lepton_attention(torch.from_numpy(jc_pf_points),
                                            torch.from_numpy(jc_pf_features[:,0:7,:]),
                                            torch.from_numpy(jc_pf_vectors),torch.from_numpy(jc_pf_mask))
    jck_attention = jc_kin_lepton_attention.get_attention_matrix()
    jck_interaction = jc_kin_lepton_attention.get_interactionMatrix()

    print('JC full done!')

    jc_kin_padding = jc_kin_lepton_attention_hooks.cut_padding(jc_kin_lepton_attention_hooks.pre_softmax_attentions, jc_pf_mask)
    jc_kin_init_padding = init_lepton_attention_hooks.cut_padding(init_lepton_attention_hooks.pre_softmax_attentions, jc_pf_mask)

    attn = jc_kin_lepton_attention_hooks.pre_softmax_attentions.numpy()
    inter = jc_kin_lepton_attention_hooks.pre_softmax_interactions.numpy()

    init_attn = init_lepton_attention_hooks.pre_softmax_attentions.numpy()
    init_inter = init_lepton_attention_hooks.pre_softmax_interactions.numpy()

    # attn, inter: (L, N, H, 128, 128)
    # jck_pf_features: (N, 17, 128) with 9 = electron, 10 = muon
    ELECTRON_IDX = 9
    MUON_IDX     = 10

    init_ratios = []
    ratios = []
    interactionval = []
    totalval = []

    # optional: collect raw “unclipped” ratio to illustrate the blow-up
    raw_ratios = []

    for li, x in enumerate(tqdm(attn, desc="Layers")):         # x: (N, H, 128, 128)
        for ni, z in enumerate(x):                              # z: (H, 128, 128)
            # muon/electron key columns for THIS SAMPLE
            key_mask = (jc_pf_features[ni, ELECTRON_IDX, :].astype(bool) |
                        jc_pf_features[ni, MUON_IDX, :].astype(bool))
            key_cols = np.flatnonzero(key_mask)

            for hi, y in enumerate(z):                          # y: (128, 128)
                I = inter[li, ni, hi]

                # for logging (matches your original spirit)
                interactionval.append(np.nansum(I))
                totalval.append(np.nansum(y))

                # --- BAD (raw) definition that can explode (denom cancels to ~0) ---
                raw_total = np.nansum(y + I)
                raw_numer = np.nansum((y + I)[:, key_cols]) if key_cols.size else 0.0
                raw_ratios.append(raw_numer / (raw_total + 1e-12))

                # --- GOOD bounded definition: use positive part of (attn + inter) ---
                A_total = y + I
                A_pos = np.clip(A_total, 0, None)

                denom = np.nansum(A_pos)
                if denom == 0 or key_cols.size == 0:
                    ratios.append(0.0)
                else:
                    numer = np.nansum(A_pos[:, key_cols])
                    ratios.append(numer / denom)

    # (optional) quick sanity checks
    ratios = np.array(ratios, dtype=float)
    raw_ratios = np.array(raw_ratios, dtype=float)
    #print("Bounded ratio min/max:", np.nanmin(ratios), np.nanmax(ratios))
    #print("Raw ratio min/max (can be >1):", np.nanmin(raw_ratios), np.nanmax(raw_ratios))
    #print("Frac of cases with near-zero raw denom:",
    #    np.mean(np.isclose(raw_ratios * 0 + raw_numer, raw_numer) & (np.abs(raw_total) < 1e-8)))

    for li, x in enumerate(tqdm(init_attn, desc="Layers")):         # x: (N, H, 128, 128)
        for ni, z in enumerate(x):                              # z: (H, 128, 128)
            # muon/electron key columns for THIS SAMPLE
            key_mask = (jc_pf_features[ni, ELECTRON_IDX, :].astype(bool) |
                        jc_pf_features[ni, MUON_IDX, :].astype(bool))
            key_cols = np.flatnonzero(key_mask)

            for hi, y in enumerate(z):                          # y: (128, 128)
                I = inter[li, ni, hi]

                # for logging (matches your original spirit)
                interactionval.append(np.nansum(I))
                totalval.append(np.nansum(y))

                # --- BAD (raw) definition that can explode (denom cancels to ~0) ---
                raw_total = np.nansum(y + I)
                raw_numer = np.nansum((y + I)[:, key_cols]) if key_cols.size else 0.0
                #raw_ratios.append(raw_numer / (raw_total + 1e-12))

                # --- GOOD bounded definition: use positive part of (attn + inter) ---
                A_total = y + I
                A_pos = np.clip(A_total, 0, None)

                denom = np.nansum(A_pos)
                if denom == 0 or key_cols.size == 0:
                    init_ratios.append(0.0)
                else:
                    numer = np.nansum(A_pos[:, key_cols])
                    init_ratios.append(numer / denom)

    # (optional) quick sanity checks
    ratios = np.array(ratios, dtype=float)
    raw_ratios = np.array(raw_ratios, dtype=float)
    #print("Bounded ratio min/max:", np.nanmin(ratios), np.nanmax(ratios))
    #print("Raw ratio min/max (can be >1):", np.nanmin(raw_ratios), np.nanmax(raw_ratios))
    #print("Frac of cases with near-zero raw denom:",
    #    np.mean(np.isclose(raw_ratios * 0 + raw_numer, raw_numer) & (np.abs(raw_total) < 1e-8)))

    #print('These are the ratios of attention to lepton / overall:')
    #print(f'Model trained on JetClass Kinematic: {ratios}')
    #print(f'Untrained Model: {init_ratios}')

    np.save(f'{chunk}leptonratiosUNTRAINED.npy', init_ratios)
    np.save(f'{chunk}leptonratiosTRAINED.npy', ratios)
    subprocess.run(['sudo', 'mv', f'{chunk}leptonratiosUNTRAINED_{counter}_to_{counter+howmany_jets}.npy', storage_path])
    subprocess.run(['sudo', 'mv', f'{chunk}leptonratiosTRAINED.npy', storage_path])

    print(f"Saved ratios for chunk {chunk} to {storage_path} - processed jets {counter} to {counter+howmanyjets}")
    counter += howmanyjets
    with open('counter', 'w') as f:
        f.write(str(counter))
    subprocess.run(['sudo', 'cp', 'counter', counter_path])

# Load arrays
untrained = np.load('leptonratiosUNTRAINED.npy')
trained = np.load('leptonratiosTRAINED.npy')

# Create figure
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# Plot histograms as outlines only
ax.hist(untrained, bins=20, density=True, histtype='step',
        linewidth=2, label="Untrained")
ax.hist(trained, bins=20, density=True, histtype='step',
        linewidth=2, label="Trained")

# Labels and formatting
ax.set_xlabel("Proportion of attention to Lepton", fontsize=20)
ax.set_ylabel("Probability", fontsize=20)
ax.set_yscale('log')

# Tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=12)

# Legend
ax.legend(fontsize=20)

# Layout and save
plt.tight_layout()
plt.savefig('leptonAttention.pdf')
#plt.show()