#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:25:42 2026

@author: apple
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset #subset to slice out dataset for diff FL clients
import numpy as np

#Loading MNIST dataset
def load_mnist():
    
    transform = transforms.Compose([
        transforms.ToTensor(), #turning our images from pixels(0-255) to pytorch tensor values(0-1.0)
        transforms.Normalize((0.1307,), (0.3081,)) ]) # MNIST mean/std
    
    
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return trainset, testset

#Federated learning Data partition
#Each client gets 3000 samples (60000 / 20).
def get_client_data(trainset, client_id, num_clients=20):
    
    total = len(trainset)
    indices = list(range(total))
    split = total // num_clients

    #making sure data doesnt overlap
    start = client_id * split
    end   = start + split
    client_indices = indices[start:end]
    
    subset = Subset(trainset, client_indices)
    
    return DataLoader(subset, batch_size=32, shuffle=True)

#------------ Non-IID Block --------------------

def get_client_data_noniid(trainset, client_id, num_clients=20, alpha=0.5, seed=42):
 
    np.random.seed(seed)

    #Group indicing by class
    labels      = np.array(trainset.targets)
    num_classes = 10   #10 digit classes of MNIST
    class_indices = [ np.where(labels == c)[0].tolist() for c in range(num_classes)]

    #Shuffling within each class for randomness
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

#Dirichlet distribution
    # For each class, sample proportions for each client
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))

        class_size = len(class_indices[c])
        counts     = (proportions * class_size).astype(int)

#Fix rounding and no negatives
        counts[-1] = class_size - counts[:-1].sum()
        counts     = np.maximum(counts, 0)  
        
        #Assign indices to clients
        start = 0
        for client in range(num_clients):
            
            end = start + counts[client]
            client_indices[client].extend(class_indices[c][start:end])
            start = end

    # Log distribution for this client
    client_labels = labels[client_indices[client_id]]
    
    unique, counts_per_class = np.unique(client_labels, return_counts=True)
    
    distribution = dict(zip(unique.tolist(), counts_per_class.tolist()))

    print(f"  [NON-IID] Client {client_id} | "
          f"Total: {len(client_indices[client_id])} samples | "
          f"Distribution: {distribution}")

    subset = Subset(trainset, client_indices[client_id])
    return DataLoader(subset, batch_size=32, shuffle=True)