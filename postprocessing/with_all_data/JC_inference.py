# This script carries out the batched loading of the JetClass dataset and stores it in a specified 
# dir if it does not yet exist. It proceeds to run inference on each batch up to the full 100,000 jets 
# and store resulting attention matrices in .npy files. Finally, it collects the values in each matrix, 
# storing those into 20-bin histogram .npy files as well. This prepares the data for compile_all_batched_hists.

import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
import os
import shutil
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm
import torch
#from weaver.nn.model.ParticleTransformer import ParticleTransformer
#from weaver.utils.logger import _logger
import torch.optim as optim
#from EfficientParticleTransformer import EfficientParticleTransformer
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import mplhep as hep
from model_utils import *
import subprocess
import argparse

hep.style.use(hep.style.ROOT)
parser = argparse.ArgumentParser(description='Run inference on JetClass full dataset in batches and save predictions, attention matrices and histograms.')
parser.add_argument('--model_type', '-m', type=str, default='jc_full', help='Type of model (jc_full, jc_kin, jc_kinpid)')
parser.add_argument('--model_name', '-mn', type=str, default='ParT_full.pt', help='Name of the model to load for inference')
parser.add_argument('--chunk', '-c', type=int, default=0, help='Chunk number to process')
parser.add_argument('--num-chunks', '-n', type=int, default=10, help='Total number of chunks to divide the dataset into')
parser.add_argument('--random', '-rd', action='store_true', help='Use randomly shuffled data instead of sequential chunks')
parser.add_argument('--classes', '-cl', type=str, nargs='*', default=['QCD', 'Tbqq'], help='Classes to save attention for (e.g., QCD Tbqq)')
parser.add_argument('--save-predictions', '-sp', action='store_true', help='Save model output predictions to file')
parser.add_argument('--only-attention', '-oa', action='store_true', help='Only save attention matrices, skip histogram generation')
parser.add_argument('--restart', '-r', action='store_true', help='Restart processing from the beginning of the current chunk, overwriting existing outputs')
args = parser.parse_args()

model_type = args.model_type
model_name = args.model_name
chunk = args.chunk
num_chunks = args.num_chunks
balanced = args.random
classes_to_process = args.classes
save_predictions = args.save_predictions
only_attention = args.only_attention
restart = args.restart

model_path = f'./models/{model_name}'

base_dir = '/moe-interpretability-pv/'

howmanyjets = 1000

dataset_path = base_dir+'datasets/'
# Create balanced or standard suffix for storage path
balanced_suffix = f'_balanced' if balanced else ''
storage_path = base_dir+f'{model_name.replace(".pt", "")}_output{balanced_suffix}/'

# Create class-specific counter file
classes_suffix = '_'.join(classes_to_process).lower()
counter_path = storage_path + f'chunk_{chunk}_counter_{classes_suffix}.txt'

print('Loading models...')

classes = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
subjets = [1, 2, 2, 2, 4, 3, 2, 2, 3, 2]
start_indices = np.array([8, 0, 1, 2, 4, 3, 9, 7, 6, 5]) * 10000 # start indices of each class in 100k dataset

# Validate requested classes
for cls in classes_to_process:
    if cls not in classes:
        raise ValueError(f"Class {cls} not in available classes: {classes}")

# Get class indices for filtering
class_indices = {cls: idx for idx, cls in enumerate(classes)}

model = get_model(model_type)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
total_jets = 100000
start_jet = chunk*(total_jets//num_chunks)

# Initialize counter with improved system similar to get_subjet_attention.py
if not os.path.exists(counter_path) or restart:
    subprocess.run(['sudo', 'mkdir', '-p', storage_path])
    subprocess.run(['sudo', 'mkdir', '-p', storage_path + 'batched_attns/'])
    subprocess.run(['sudo', 'mkdir', '-p', storage_path + 'batched_preds/'])
    subprocess.run(['sudo', 'mkdir', '-p', storage_path + 'batched_hists/'])
    counter = start_jet
    with open('counter.txt', 'w') as f:
        f.write(str(counter))
    subprocess.run(['sudo', 'cp', 'counter.txt', counter_path])
else:
    subprocess.run(['sudo', 'cp', counter_path, 'counter.txt'])
    with open(counter_path, 'r') as f:
        counter = int(f.read().strip())

subprocess.run(['sudo', 'chmod', '666', 'counter.txt'])

# Define feature slicing for different model types
kin_slice = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1], dtype=bool)
kinpid_slice = np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1], dtype=bool)

print('Loading dataset...')
# Load all dataset files
jc_full_pf_features = np.load(dataset_path+'jc_full_pf_features.npy')
jc_full_pf_vectors = np.load(dataset_path+'jc_full_pf_vectors.npy')
jc_full_pf_mask = np.load(dataset_path+'jc_full_pf_mask.npy')
jc_full_pf_points = np.load(dataset_path+'jc_full_pf_points.npy')
jc_full_labels = np.load(dataset_path+'jc_full_labels.npy')

# Apply feature slicing based on model type
if model_type == 'jc_kinpid':
    jc_full_pf_features = jc_full_pf_features[:, kinpid_slice, :]
elif model_type == 'jc_kin':
    jc_full_pf_features = jc_full_pf_features[:, kin_slice, :]
elif model_type == 'jc_full':
    jc_full_pf_features = jc_full_pf_features

# Apply random sampling if requested
if balanced:
    np.random.seed(42)  # For reproducibility
    random_permutation = np.random.permutation(len(jc_full_pf_features))
    jc_full_pf_features = jc_full_pf_features[random_permutation]
    jc_full_pf_vectors = jc_full_pf_vectors[random_permutation]
    jc_full_pf_mask = jc_full_pf_mask[random_permutation]
    jc_full_pf_points = jc_full_pf_points[random_permutation]
    jc_full_labels = jc_full_labels[random_permutation]
    start_jet = chunk * (total_jets // num_chunks)
    # Adjust counter if necessary
    if counter < start_jet:
        counter = start_jet

print(f'Processing {len(classes_to_process)} classes: {classes_to_process}')
print(f'Balanced sampling: {balanced}')
print(f'Starting inference from jet {counter} in chunk {chunk}')

# Main inference loop
chunk_end = min((chunk + 1) * (total_jets // num_chunks), total_jets)

while counter < chunk_end:
    # Determine batch size and indices
    batch_start = counter
    batch_end = min(counter + howmanyjets, chunk_end)
    batch_indices = np.arange(batch_start, batch_end)
    
    if len(batch_indices) == 0:
        break
    
    print(f"\nProcessing batch: jets {batch_start} to {batch_end} (batch size: {len(batch_indices)})")
    
    # Load batch data
    points = jc_full_pf_points[batch_indices]
    features = jc_full_pf_features[batch_indices]
    vectors = jc_full_pf_vectors[batch_indices]
    mask = jc_full_pf_mask[batch_indices]
    labels = jc_full_labels[batch_indices]
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(points), torch.from_numpy(features), 
                           torch.from_numpy(vectors), torch.from_numpy(mask))
    
    # Get attention matrices from model
    attention_matrices = [tensor.numpy() for tensor in model.get_attention_matrix()]
    
    # Save predictions if requested
    if save_predictions:
        predictions_np = predictions.numpy()
        pred_dir = f'{storage_path}/batched_preds/'
        np.save(f'/predictions_batch_{batch_start}_{batch_end}.npy', predictions_np)
        subprocess.run(['sudo', 'cp', f'predictions_batch_{batch_start}_{batch_end}.npy', pred_dir])
        print(f'Saved predictions to batch file')
    
    # Process and save attention matrices for specified classes
    class_attention_dict = {cls: [] for cls in classes_to_process}
    
    for jet_idx, predicted_label in enumerate(predictions.argmax(dim=1).numpy()):
        true_label = np.argmax(labels[jet_idx])
        true_label_name = classes[true_label]
        
        # Check if this jet's true label is in classes to process
        if true_label_name in classes_to_process:
            # Extract attention for this jet across all layers
            jet_attention = [layer[jet_idx] for layer in attention_matrices]
            class_attention_dict[true_label_name].append(jet_attention)
    
    # Save attention by class
    for cls in classes_to_process:
        if len(class_attention_dict[cls]) > 0:
            attention_dir = f'{storage_path}/batched_attns/'
            np.save(f'{cls.lower()}_attention_batch_{batch_start}_{batch_end}.npy', np.array(class_attention_dict[cls], dtype=object))
            subprocess.run(['sudo', 'mv', f'{cls.lower()}_attention_batch_{batch_start}_{batch_end}.npy', attention_dir])
            print(f'  Saved {len(class_attention_dict[cls])} {cls} attention matrices')
    
    # Update counter
    counter = batch_end
    with open('counter.txt', 'w') as f:
        f.write(str(counter))
    subprocess.run(['sudo', 'cp', 'counter.txt', counter_path])
    print(f'  Counter updated to {counter}')

print(f'\nInference phase complete for chunk {chunk}!')

bin_edges = np.linspace(0, 1, 21)

# Function to process data in chunks and compute histogram
def process_in_chunks(attention_iterator, chunk_size=100000, bin_edges=bin_edges):
    hist_counts = np.zeros(len(bin_edges) - 1)  # Initialize histogram counts for bins

    # Process each chunk of attention data
    for chunk in attention_iterator:
        # Flatten the chunk to ensure it's 1D and processable by np.histogram
        chunk = np.array(chunk).flatten()

        # Calculate histogram for this chunk
        hist, _ = np.histogram(chunk, bins=bin_edges)

        # Accumulate the counts
        hist_counts += hist

    total_data_points = hist_counts.sum()  # Total number of points processed
    probabilities = hist_counts / total_data_points  # Normalize to get probabilities
    return probabilities

# Simulate loading a large dataset in chunks (e.g., from a file or other source)
def attention_generator(attention, chunk_size):
    """Simulate chunked data loader for large dataset."""
    for i in range(0, len(attention), chunk_size):
        yield attention[i:i + chunk_size]

# Skip histogram generation if only_attention flag is set
if only_attention:
    print('Skipping histogram generation (--only-attention flag set)')
else:
    print('\nStarting histogram generation phase...')
    
    # Process histograms for each class
    for cls in classes_to_process:
        cls_lower = cls.lower()
        hist_counter_file = f'{storage_path}/{cls_lower}_hist_counter.txt'
        
        # Initialize or resume histogram counter
        if not os.path.exists(hist_counter_file):
            hist_counter = 0
            with open(hist_counter_file, 'w') as f:
                f.write(str(hist_counter))
        else:
            print(f'Histogram counter for {cls} exists, resuming from last batch')
            with open(hist_counter_file, 'r') as f:
                hist_counter = int(f.read().strip())
        
        # Count total attention batch files for this class
        attention_files = [f for f in os.listdir(storage_path + 'batched_attns/') 
                          if f.startswith(f'{cls_lower}_attention_batch_') and f.endswith('.npy')]
        num_batches = len(attention_files)
        
        print(f'\nProcessing {cls} histograms: found {num_batches} batches')
        
        # Process histogram batches for this class
        while hist_counter < num_batches:
            # Find the batch file for this counter
            batch_file = None
            for f in attention_files:
                if f'batch_{hist_counter}' in f or f.endswith(f'_batch_{hist_counter}.npy'):
                    batch_file = f
                    break
            
            if batch_file is None:
                # Look for any file with pattern and sort to get nth file
                sorted_files = sorted(attention_files)
                if hist_counter < len(sorted_files):
                    batch_file = sorted_files[hist_counter]
                else:
                    print(f'No more batch files found for {cls}, completing histogram generation for this class')
                    break
            
            print(f"  Processing {cls} distribution batch {hist_counter}: {batch_file}")
            attention = np.load(f'{storage_path}/batched_attns/{batch_file}', allow_pickle=True)
            
            # Check if attention is empty
            if len(attention) == 0:
                print(f"    No jets found in batch, skipping...")
                with open(hist_counter_file, 'w') as f:
                    f.write(str(hist_counter + 1))
                hist_counter += 1
                continue
            
            # Flatten and compute histogram
            flattened_attention = np.stack(attention).flatten()
            attention_iter = attention_generator(flattened_attention, chunk_size=100000)
            probabilities = process_in_chunks(attention_iter, chunk_size=100000, bin_edges=bin_edges)
            
            # Save histogram
            hist_file = f'{storage_path}/batched_hists/{cls_lower}_hist_batch_{hist_counter}.npy'
            np.save(hist_file, probabilities)
            print(f"    Saved histogram distribution for batch {hist_counter}")
            
            # Update counter
            with open(hist_counter_file, 'w') as f:
                f.write(str(hist_counter + 1))
            hist_counter += 1
        
        print(f'Completed histogram generation for {cls}')
    
    print('\nHistogram generation phase complete!')
