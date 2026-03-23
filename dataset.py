
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset #subset to slice out dataset for diff FL clients

import numpy as np


#Loading MNIST dataset
def load_mnist():
    transform = transforms.Compose([
        
        transforms.ToTensor(),#turning our images from pixels(0-255) to pytorch tensor values(0-1.0)
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return trainset, testset


#Federated learning Data partition
#Each client gets 6000 samples (60000 / 10).
def get_client_data(trainset, client_id, num_clients=10):
    
    total = len(trainset)
    indices = list(range(total))
    split = total // num_clients

    #making sure data doesnt overlap
    start = client_id * split
    end   = start + split
    client_indices = indices[start:end]
    
    subset = Subset(trainset, client_indices)
    
    return DataLoader(subset, batch_size=32, shuffle=True)
