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
import logging

from model_utils import *

parser = argparse.ArgumentParser(description='Subjet attention job specs.')
parser.add_argument('--class-to-analyze', '-a', type=str, help='Class to analyze (H4q, Tbqq)')
parser.add_argument('--chunk', '-c', type=int, help='chunk number')
parser.add_argument('--num-chunks', '-n', type=int, default=10, help='total number of chunks')
parser.add_argument('--restart', '-r', action='store_true', help='Whether to restart the job from scratch, or continue from the last counter')
parser.add_argument('--zero-u', '-z', action='store_true', help='Whether to only run the zeroed U model')
parser.add_argument('--plot', '-p', action='store_true', help='To plot instead of run inference')
args = parser.parse_args()

class_to_analyze = args.class_to_analyze
chunk = args.chunk
num_chunks = args.num_chunks
restart = args.restart
zero_u_only = args.zero_u
plot_q = args.plot

jc_full_model = get_model(model_type='jc_full', return_pre_softmax=True)

jc_full_hooks = Pre_Softmax_Hook(model=jc_full_model)

classes = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
subjets = [1, 2, 2, 2, 4, 3, 2, 2, 3, 2]
start_indices = np.array([8, 0, 1, 2, 4, 3, 9, 7, 6, 5]) * 10000 # start indices of each class in 100k dataset
total_jets = 10000

start_jet = counter = start_indices[classes.index(class_to_analyze)] + chunk*(total_jets//num_chunks)
N_SUBJETS = subjets[classes.index(class_to_analyze)]
assert class_to_analyze in ['H4q', 'Tbqq'], 'to get lepton attention plots, please specify class as Hqql or Tbl'

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

subprocess.run(['sudo', 'chmod', '666', 'counter.txt'])

jc_kin_lepton_attention = get_model('jck')
model_path = '/home/jovyan/Interpreting-Particle-Transformers/models/ParT_kin.pt'
zero_u_model_path = '/home/jovyan/Interpreting-Particle-Transformers/models/JetClass_Kin_ParT_zeroed_interaction.pt'
jc_kin_lepton_attention.load_state_dict(torch.load(model_path, map_location='cpu'))
init_lepton_attention = get_model('jck')
zero_u_jc_kin = get_model('jck')
zero_u_jc_kin.load_state_dict(torch.load(zero_u_model_path, map_location='cpu'))

kin_slice = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1], dtype=bool)
kinpid_slice = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1], dtype=bool)

jc_full_pf_features = np.load(dataset_path+'jc_full_pf_features.npy')[:,kin_slice,:]
jc_full_pf_vectors = np.load(dataset_path+'jc_full_pf_vectors.npy')
jc_full_pf_mask = np.load(dataset_path+'jc_full_pf_mask.npy')
jc_full_pf_points = np.load(dataset_path+'jc_full_pf_points.npy')
jc_full_labels = np.load(dataset_path+'jc_full_labels.npy')

print(f"Starting processing for chunk {chunk} of class {class_to_analyze} - jets {counter} to {counter+howmanyjets}")

while counter < start_jet + (total_jets//num_chunks) and not plot_q:
    jc_pf_features = jc_full_pf_features[counter:counter+howmanyjets]
    jc_pf_vectors = jc_full_pf_vectors[counter:counter+howmanyjets]
    jc_pf_mask = jc_full_pf_mask[counter:counter+howmanyjets]
    jc_pf_points = jc_full_pf_points[counter:counter+howmanyjets]
    jc_labels = jc_full_labels[counter:counter+howmanyjets]

    if not zero_u_only:

        assert jc_labels.shape[0] == howmanyjets, f"Expected {howmanyjets} jets, but got {jc_labels.shape[0]}"

        jc_kin_lepton_attention_hooks = Pre_Softmax_Hook(model=jc_kin_lepton_attention)
        init_lepton_attention_hooks = Pre_Softmax_Hook(model=init_lepton_attention)

        init_lepton_attention.eval()
        with torch.no_grad():
            init_pred = init_lepton_attention(torch.from_numpy(jc_pf_points),
                                        torch.from_numpy(jc_pf_features),
                                        torch.from_numpy(jc_pf_vectors),torch.from_numpy(jc_pf_mask))

        jc_kin_lepton_attention.eval()
        with torch.no_grad():
            jck_y_pred= jc_kin_lepton_attention(torch.from_numpy(jc_pf_points),
                                                torch.from_numpy(jc_pf_features),
                                                torch.from_numpy(jc_pf_vectors),torch.from_numpy(jc_pf_mask))

        logging.info('JC inference done!')

        jc_kin_padding = jc_kin_lepton_attention_hooks.cut_padding(jc_kin_lepton_attention_hooks.pre_softmax_attentions, jc_pf_mask)
        jc_kin_init_padding = init_lepton_attention_hooks.cut_padding(init_lepton_attention_hooks.pre_softmax_attentions, jc_pf_mask)
        
        attn = jc_kin_lepton_attention_hooks.pre_softmax_attentions.numpy()
        inter = jc_kin_lepton_attention_hooks.pre_softmax_interactions.numpy()

        init_attn = init_lepton_attention_hooks.pre_softmax_attentions.numpy()
        init_inter = init_lepton_attention_hooks.pre_softmax_interactions.numpy()

        # attn, inter: (L, N, H, 128, 128)
        # jck_pf_features: (N, 17, 128) with 9 = electron, 10 = muon

        init_ratios = []
        ratios = []
        zero_u_ratios = []
        interactionval = []
        totalval = []

        for li, x in enumerate(tqdm(attn, desc="Layers")):         # x: (N, H, 128, 128)
            for ni, z in enumerate(x):                              # z: (H, 128, 128)
                # Extract the 4-momentum components for the valid particles
                px = jc_pf_vectors[ni][0][0:jc_kin_padding[ni]]
                py = jc_pf_vectors[ni][1][0:jc_kin_padding[ni]]
                pz = jc_pf_vectors[ni][2][0:jc_kin_padding[ni]]
                e = jc_pf_vectors[ni][3][0:jc_kin_padding[ni]]
                
                # Get the subjets using the get_subjets function

                subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

                # muon/electron key columns for THIS SAMPLE
                #key_mask = (jc_pf_features[ni, ELECTRON_IDX, :].astype(bool) |
                #            jc_pf_features[ni, MUON_IDX, :].astype(bool))
                subjets_mask = np.empty((jc_kin_padding[ni],N_SUBJETS), dtype=bool)
                for si in range(N_SUBJETS):
                    subjets_mask[:,si] = np.where((si == subjets), True, False)

                for hi, y in enumerate(z):                          # y: (128, 128)
                    I = inter[li, ni, hi]

                    # for logging (matches your original spirit)
                    interactionval.append(np.nansum(I))
                    totalval.append(np.nansum(y))

                    # --- GOOD bounded definition: use positive part of (attn + inter) ---
                    A_total = y + I
                    A_pos = np.clip(A_total, 0, None)

                    # mask attention matrix to each subjet
                    A_total_subjet = np.zeros(N_SUBJETS)
                    for si in range(N_SUBJETS):
                        A_subjet = A_pos[:jc_kin_padding[ni], :jc_kin_padding[ni]][subjets_mask[:,si]][:,subjets_mask[:,si]]
            
                        # attention within the subjet
                        A_total_subjet[si] = np.nansum(A_subjet)
                    
                    # Attention between subjets is then A_pos - sum(A_total_subjet), and we can log that separately if desired
                    attn_between_subjets = np.nansum(A_pos) - np.nansum(A_total_subjet)

                    denom = np.nansum(A_pos)
                    if denom == 0:
                        ratios.append(0.0)
                    else:
                        numer = attn_between_subjets
                        ratios.append(numer / denom)

        for li, x in enumerate(tqdm(init_attn, desc="Layers")):         # x: (N, H, 128, 128)
            for ni, z in enumerate(x):                              # z: (H, 128, 128)
                # Extract the 4-momentum components for the valid particles
                px = jc_pf_vectors[ni][0][0:jc_kin_padding[ni]]
                py = jc_pf_vectors[ni][1][0:jc_kin_padding[ni]]
                pz = jc_pf_vectors[ni][2][0:jc_kin_padding[ni]]
                e = jc_pf_vectors[ni][3][0:jc_kin_padding[ni]]
                
                # Get the subjets using the get_subjets function

                subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

                # muon/electron key columns for THIS SAMPLE
                #key_mask = (jc_pf_features[ni, ELECTRON_IDX, :].astype(bool) |
                #            jc_pf_features[ni, MUON_IDX, :].astype(bool))
                subjets_mask = np.empty((jc_kin_padding[ni],N_SUBJETS), dtype=bool)
                for si in range(N_SUBJETS):
                    subjets_mask[:,si] = np.where((si == subjets), True, False)

                for hi, y in enumerate(z):                          # y: (128, 128)
                    I = init_inter[li, ni, hi]

                    # for logging (matches your original spirit)
                    interactionval.append(np.nansum(I))
                    totalval.append(np.nansum(y))

                    # --- GOOD bounded definition: use positive part of (attn + inter) ---
                    A_total = y + I
                    A_pos = np.clip(A_total, 0, None)

                    # mask attention matrix to each subjet
                    A_total_subjet = np.zeros(N_SUBJETS)
                    for si in range(N_SUBJETS):
                        A_subjet = A_pos[:jc_kin_padding[ni], :jc_kin_padding[ni]] * (subjets_mask[:,si][:,None] * subjets_mask[:,si][None,:])
                        # attention within the subjet
                        A_total_subjet[si] = np.nansum(A_subjet)
                    
                    # Attention between subjets is then A_pos - sum(A_total_subjet), and we can log that separately if desired
                    attn_between_subjets = np.nansum(A_pos) - np.nansum(A_total_subjet)

                    denom = np.nansum(A_pos)
                    if denom == 0:
                        init_ratios.append(0.0)
                    else:
                        numer = attn_between_subjets
                        init_ratios.append(numer / denom)
        
        untrained_file = f'testsubjetratiosUNTRAINED_{counter}_to_{counter+howmanyjets}.npy'
        trained_file = f'testsubjetratiosTRAINED_{counter}_to_{counter+howmanyjets}.npy'
        np.save(untrained_file, init_ratios)
        np.save(trained_file, ratios)
        subprocess.run(['sudo', 'mv', untrained_file, storage_path])
        subprocess.run(['sudo', 'mv', trained_file, storage_path])

    else:
        zero_u_jc_kin_hooks = Pre_Softmax_Hook(model=zero_u_jc_kin)

        zero_u_jc_kin.eval()
        with torch.no_grad():
            zero_u_jck_y_pred= zero_u_jc_kin(torch.from_numpy(jc_pf_points),
                                                torch.from_numpy(jc_pf_features),
                                                torch.from_numpy(jc_pf_vectors),torch.from_numpy(jc_pf_mask))
        
        jc_kin_padding = zero_u_jc_kin_hooks.cut_padding(zero_u_jc_kin_hooks.pre_softmax_attentions, jc_pf_mask)

        zero_u_attn = zero_u_jc_kin_hooks.pre_softmax_attentions.numpy()
        zero_u_inter = zero_u_jc_kin_hooks.pre_softmax_interactions.numpy()
        
        init_ratios = []
        ratios = []
        zero_u_ratios = []
        interactionval = []
        totalval = []
        print(zero_u_attn.shape)
        for li, x in enumerate(tqdm(zero_u_attn, desc="Layers")):         # x: (N, H, 128, 128)
            for ni, z in enumerate(x):                              # z: (H, 128, 128)
                # Extract the 4-momentum components for the valid particles
                px = jc_pf_vectors[ni][0][0:jc_kin_padding[ni]]
                py = jc_pf_vectors[ni][1][0:jc_kin_padding[ni]]
                pz = jc_pf_vectors[ni][2][0:jc_kin_padding[ni]]
                e = jc_pf_vectors[ni][3][0:jc_kin_padding[ni]]
                
                # Get the subjets using the get_subjets function

                subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

                # muon/electron key columns for THIS SAMPLE
                #key_mask = (jc_pf_features[ni, ELECTRON_IDX, :].astype(bool) |
                #            jc_pf_features[ni, MUON_IDX, :].astype(bool))
                subjets_mask = np.empty((jc_kin_padding[ni],N_SUBJETS), dtype=bool)
                for si in range(N_SUBJETS):
                    subjets_mask[:,si] = np.where((si == subjets), True, False)

                for hi, y in enumerate(z):                          # y: (128, 128)
                    I = zero_u_inter[li, ni, hi]

                    # for logging (matches your original spirit)
                    interactionval.append(np.nansum(I))
                    totalval.append(np.nansum(y))

                    # --- GOOD bounded definition: use positive part of (attn + inter) ---
                    A_total = y + I
                    A_pos = np.clip(A_total, 0, None)

                    # mask attention matrix to each subjet
                    A_total_subjet = np.zeros(N_SUBJETS)
                    for si in range(N_SUBJETS):
                        A_subjet = A_pos[:jc_kin_padding[ni], :jc_kin_padding[ni]][subjets_mask[:,si]][:,subjets_mask[:,si]]
            
                        # attention within the subjet
                        A_total_subjet[si] = np.nansum(A_subjet)
                    
                    # Attention between subjets is then A_pos - sum(A_total_subjet), and we can log that separately if desired
                    attn_between_subjets = np.nansum(A_pos) - np.nansum(A_total_subjet)

                    denom = np.nansum(A_pos)
                    if denom == 0:
                        zero_u_ratios.append(0.0)
                    else:
                        numer = attn_between_subjets
                        zero_u_ratios.append(numer / denom)

        #print('These are the ratios of attention to lepton / overall:')
        #print(f'Model trained on JetClass Kinematic: {ratios}')
        #print(f'Untrained Model: {init_ratios}')
        
        zero_u_file = f'subjetratiosZERO_U_{counter}_to_{counter+howmanyjets}.npy'
        np.save(zero_u_file, zero_u_ratios)
        subprocess.run(['sudo', 'mv', zero_u_file, storage_path])

    print(f"Saved ratios for chunk {chunk} to {storage_path} - processed jets {counter} to {counter+howmanyjets}")
    logging.info(f"Saved ratios for chunk {chunk} to {storage_path} - processed jets {counter} to {counter+howmanyjets}")
    counter += howmanyjets
    with open('counter.txt', 'w') as f:
        f.write(str(counter))
    subprocess.run(['sudo', 'cp', 'counter.txt', counter_path])

# collate all chunks for this class
untrained = np.array([])
trained = np.array([])
zero_u = np.array([])
for file in os.listdir(storage_path):
    if file.startswith('testsubjetratiosUNTRAINED') and file.endswith('.npy'):
        untrained = np.concatenate((untrained, np.load(os.path.join(storage_path, file))))
    elif file.startswith('testsubjetratiosTRAINED') and file.endswith('.npy'):
        trained = np.concatenate((trained, np.load(os.path.join(storage_path, file))))
#    elif file.startswith('subjetratiosZERO_U') and file.endswith('.npy'):
#        zero_u = np.concatenate((zero_u, np.load(os.path.join(storage_path, file))))

# Create figure
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# Plot histograms as outlines only
ax.hist(untrained, bins=50, density=True, histtype='step',
        linewidth=2, label="Untrained")
ax.hist(trained, bins=50, density=True, histtype='step',
        linewidth=2, label="Trained")
#ax.hist(zero_u, bins=50, density=True, histtype='step',
#        linewidth=2, label="Zeroed U")

# Labels and formatting
ax.set_xlabel("Proportion of attention between subjets", fontsize=20)
ax.set_ylabel("Probability", fontsize=20)
ax.set_yscale('log')

# Tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=12)

# Legend
ax.legend(fontsize=20)

# Layout and save
plt.tight_layout()
plt.savefig(f'subjetAttention_{class_to_analyze}.pdf')
subprocess.run(['sudo', 'cp', f'subjetAttention_{class_to_analyze}.pdf', storage_path])
#plt.show()