
'''
Usage:

python ./getting_eta_phi_plots.py <decay_type>

Looks for jets starting from jet_number to jet_number + 1000 for either Hadronic or Leptonic decays.
plots corresponding eta-phi attention maps if it can find them.

'''

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
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import argparse
from model_utils import *

plt.style.use(hep.style.ROOT)

parser = argparse.ArgumentParser(description='Plot attention maps for JetClass models.')
parser.add_argument('--decay-type', type=str, help='The decay type to plot (e.g. Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl)')
parser.add_argument('--layer', '-l', type=int, default=7, help='The layer number to plot (default: 7)')
parser.add_argument('--num-jets', '-nj', type=int, default=5, help='The number of jets to plot (default: 5)')
parser.add_argument('--num-heads', '-nh', type=int, default=1, help='The number of heads to plot (default: 1)')

args = parser.parse_args()

decay_type = args.decay_type
layer_number = args.layer
howmanyjets = args.num_jets
num_heads_plotted = args.num_heads

jck_model = get_model(model_type='jck', return_pre_softmax=True)
jc_full_model = get_model(model_type='jc_full', return_pre_softmax=True)

jc_kin_hooks = Pre_Softmax_Hook(model=jck_model)
jc_full_hooks = Pre_Softmax_Hook(model=jc_full_model)

# access data from local .npys 

import sys

start_jet = 0
classes = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
subjets = [1, 2, 2, 2, 4, 3, 2, 2, 3, 2]
if decay_type not in classes:
    raise ValueError(f"Decay type {decay_type} not recognized. Must be one of {classes}.")
howmanyjets = 8
jet_idx = 0
found_desired_jets = False

qgtrained_modelpath = './models/on-qg-run2_best_epoch_state.pt'
tltrained_modelpath = './models/on-tl-run4_best_epoch_state.pt'
hls4mltrained_modelpath = './models/on-hls4ml-run3_best_epoch_state.pt'
jcktrained_modelpath = './models/ParT_kin.pt'
jc_kinpidtrained_modelpath = './models/ParT_kinpid.pt'
jc_fulltrained_modelpath = './models/ParT_full.pt'

data_stem = '/moe-interpretability-pv/datasets/'
jck_state_dict = torch.load(jcktrained_modelpath, map_location=torch.device('cpu'))
jck_model.load_state_dict(jck_state_dict)

# JetClass kin model loading and inference
while not found_desired_jets:
    jck_labels = np.load(data_stem+'jc_full_labels.npy')[start_jet:start_jet+howmanyjets]
    for jet in range(jck_labels.shape[0]):
        if classes[np.argmax(jck_labels[jet])] == decay_type:
            label_idx = np.argmax(jck_labels[jet])
            print(f'Found desired jet of type {decay_type} at index {start_jet + jet}!')
            print(f'Plotting this and following ten jets')
            start_jet += jet
            found_desired_jets = True
            break
    if not found_desired_jets:
        start_jet += 10000    

jck_labels = np.load(data_stem+'jc_full_labels.npy')[start_jet:start_jet+howmanyjets]
jck_pf_features = np.load(data_stem+'jc_full_pf_features.npy')[start_jet:start_jet+howmanyjets]
jck_pf_vectors = np.load(data_stem+'jc_full_pf_vectors.npy')[start_jet:start_jet+howmanyjets]
jck_pf_mask = np.load(data_stem+'jc_full_pf_mask.npy')[start_jet:start_jet+howmanyjets]
jck_pf_points = np.load(data_stem+'jc_full_pf_points.npy')[start_jet:start_jet+howmanyjets]

# remove indices 6-15 on axis 1 for kinematic feats only
non_kin_feats = list(range(5, 15))
jck_pf_features = np.delete(jck_pf_features, non_kin_feats, axis=1)

jck_model.eval()
with torch.no_grad():
    jck_y_pred= jck_model(torch.from_numpy(jck_pf_points),torch.from_numpy(jck_pf_features),torch.from_numpy(jck_pf_vectors),torch.from_numpy(jck_pf_mask))
jck_attention = jck_model.get_attention_matrix()
jck_interaction = jck_model.get_interactionMatrix()

print('JCK done!')

# JetClass full model loading and inference

jc_full_state_dict = torch.load(jc_fulltrained_modelpath, map_location=torch.device('cpu'))
jc_full_model.load_state_dict(jc_full_state_dict)
jc_full_pf_features = np.load(data_stem+'jc_full_pf_features.npy')[start_jet:start_jet+howmanyjets]
jc_full_pf_vectors = np.load(data_stem+'jc_full_pf_vectors.npy')[start_jet:start_jet+howmanyjets]
jc_full_pf_mask = np.load(data_stem+'jc_full_pf_mask.npy')[start_jet:start_jet+howmanyjets]
jc_full_pf_points = np.load(data_stem+'jc_full_pf_points.npy')[start_jet:start_jet+howmanyjets]
jc_full_labels = np.load(data_stem+'jc_full_labels.npy')[start_jet:start_jet+howmanyjets]
jc_full_model.eval()
with torch.no_grad():
    jc_full_y_pred= jc_full_model(torch.from_numpy(jc_full_pf_points),torch.from_numpy(jc_full_pf_features),torch.from_numpy(jc_full_pf_vectors),torch.from_numpy(jc_full_pf_mask))
jc_full_attention = jc_full_model.get_attention_matrix()
jc_full_interaction = jc_full_model.get_interactionMatrix()

print('JC Full done!')

jc_kin_padding = jc_kin_hooks.cut_padding(jc_kin_hooks.pre_softmax_attentions, jck_pf_mask)
jc_kin_pre_softmax_inter = jc_kin_hooks.cut_padding(jc_kin_hooks.pre_softmax_interactions, jck_pf_mask)

jc_full_padding = jc_full_hooks.cut_padding(jc_full_hooks.pre_softmax_attentions, jc_full_pf_mask)

import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import fastjet

# Example usage based on your context (assuming pf_features, pf_mask, and attention are already defined)

jet = jet_idx-1
number = jet
num = jc_kin_padding[0]

print(f'Graphing for {decay_type} jet')

# Extract the 4-momentum components for the valid particles
px = jck_pf_vectors[jet][0][0:num]
py = jck_pf_vectors[jet][1][0:num]
pz = jck_pf_vectors[jet][2][0:num]
e = jck_pf_vectors[jet][3][0:num]

# Get the subjets using the get_subjets function

N_SUBJETS = subjets[label_idx]

subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

# Initialize and combine particle data from all types
deta_all = []
dphi_all = []
pt_all = []
subjets_all = []

# Append all particle types
def append_particles(deta, dphi, pt, subjets, deta_all, dphi_all, pt_all, subjets_all):
    deta_all.extend(deta)
    dphi_all.extend(dphi)
    pt_all.extend(pt)
    subjets_all.extend(subjets)

# Process the particles and combine them into one list
append_particles(jck_pf_features[jet][5][0:num], jck_pf_features[jet][6][0:num], jck_pf_features[jet][0][0:num], subjets,
                 deta_all, dphi_all, pt_all, subjets_all)

# Convert lists to numpy arrays for plotting
deta_all = np.array(deta_all)
dphi_all = np.array(dphi_all)
pt_all = np.array(pt_all)
subjets_all = np.array(subjets_all)

# softmax the pre-softmax attention matrix

import subprocess
if not os.path.exists('./JetClasskin_attn_plots'):
    subprocess.run(['mkdir', './JetClasskin_attn_plots'])
if not os.path.exists('./JetClassfull_attn_plots'):
    subprocess.run(['mkdir', './JetClassfull_attn_plots'])

jc_kin_pre_softmax_attentions = jc_kin_hooks.pre_softmax_attentions[layer_number][jet][:, :num, :num]
for head in range(jc_kin_pre_softmax_attentions.shape[0]):
    jc_kin_pre_softmax_attentions[head] = torch.nn.functional.softmax(jc_kin_pre_softmax_attentions[head], dim=-1)

# Example attention data, where `x` is the layer number
Decay = decay_type
for head_number in range(num_heads_plotted):
  jck_plot_attention_with_particles_and_ids(jc_kin_pre_softmax_attentions[head_number, 0:num, 0:num], jet, deta_all, dphi_all, pt_all, 
                                            subjets_all, layer_number, head_number, jck_pf_features, 
                                            output_filename=f'./JetClasskin_attn_plots/Jet_{jet}_Decay_{Decay}_Layer_{layer_number+1}_head_{head_number + 1}.pdf')
  #plot_attention_with_particles(srandom_matrix[0][1][4, 0:num, 0:num], jet, deta_all, dphi_all, pt_all, subjets_all, layer_number, head_number, pf_features, '/content/drive/MyDrive/networks/Plots/Jet' + str(jet) + str(Decay) + '/' + 'randomAttentionMatrix' + str(Decay) + '-layer'+str(layer_number + 1) + '-head' + str(head_number + 1) + '.pdf')

# Extract the 4-momentum components for the valid particles
px = jc_full_pf_vectors[jet][0][0:num]
py = jc_full_pf_vectors[jet][1][0:num]
pz = jc_full_pf_vectors[jet][2][0:num]
e = jc_full_pf_vectors[jet][3][0:num]

# Get the subjets using the get_subjets function

subjets, subjet_vectors = get_subjets(px, py, pz, e, N_SUBJETS=N_SUBJETS, JET_ALGO="kt")

# Initialize and combine particle data from all types
deta_all = []
dphi_all = []
pt_all = []
subjets_all = []

# Append all particle types
def append_particles(deta, dphi, pt, subjets, deta_all, dphi_all, pt_all, subjets_all):
    deta_all.extend(deta)
    dphi_all.extend(dphi)
    pt_all.extend(pt)
    subjets_all.extend(subjets)

# Process the particles and combine them into one list
append_particles(jc_full_pf_features[jet][15][0:num], jc_full_pf_features[jet][16][0:num], jc_full_pf_features[jet][0][0:num], subjets,
                 deta_all, dphi_all, pt_all, subjets_all)

# Convert lists to numpy arrays for plotting
deta_all = np.array(deta_all)
dphi_all = np.array(dphi_all)
pt_all = np.array(pt_all)
subjets_all = np.array(subjets_all)

#softmax the pre-softmax attention matrix for jc_full

jc_full_pre_softmax_attentions = jc_full_hooks.pre_softmax_attentions[layer_number][jet][:, :num, :num]
for head in range(jc_full_pre_softmax_attentions.shape[0]):
    jc_full_pre_softmax_attentions[head] = torch.nn.functional.softmax(jc_full_pre_softmax_attentions[head], dim=-1)

Decay = Decay
for head_number in range(num_heads_plotted):
    plot_attention_with_particles(jc_full_pre_softmax_attentions[head_number, 0:num, 0:num], jet, deta_all, dphi_all, pt_all, subjets_all, 
                                  layer_number, head_number, jc_full_pf_features, 
                                  output_filename=f'./JetClassfull_attn_plots/Jet_{jet}_Decay_{Decay}_Layer_{layer_number+1}_head_{head_number + 1}.pdf')