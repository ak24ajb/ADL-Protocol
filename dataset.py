
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset #subset to slice out dataset for diff FL clients

import numpy as np


#Loading MNIST dataset
def load_mnist():

    
    transform = transforms.Compose([
        transforms.ToTensor(), #turning our images from pixels(0-255) to pytorch tensor values(0-1.0)
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    
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
    """
    Non-IID Dirichlet partition.

    Dirichlet(α) controls label skew across clients:
    - Low α  → extreme skew (clients specialise in few classes)
    - High α → mild skew (approaches IID)
    - α=0.5  → realistic heterogeneity, standard in FL literature

    Process:
    1. Group all training indices by class label
    2. For each class, sample a Dirichlet distribution over clients
       → this gives each client a proportion of that class
    3. Assign indices accordingly
    4. Each client ends up with different class distributions

    This is the partitioning method used in FLTrust, FLTG, and
    most Byzantine-resilient FL papers for Non-IID evaluation.
    """
    np.random.seed(seed)

    # Group indices by class
    labels      = np.array(trainset.targets)
    num_classes = 10   # MNIST has 10 digit classes
    class_indices = [
        np.where(labels == c)[0].tolist()
        for c in range(num_classes)
    ]

    # Shuffle within each class for randomness
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    # Dirichlet allocation
    # For each class, sample proportions for each client
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(
            alpha=np.repeat(alpha, num_clients)
        )

        # Convert proportions to actual counts
        class_size = len(class_indices[c])
        counts     = (proportions * class_size).astype(int)

        # Fix rounding — ensure all samples are allocated
        counts[-1] = class_size - counts[:-1].sum()
        counts     = np.maximum(counts, 0)   # no negatives

        # Assign indices to clients
        start = 0
        for client in range(num_clients):
            end = start + counts[client]
            client_indices[client].extend(
                class_indices[c][start:end]
            )
            start = end

    # Log distribution for this client
    client_labels = labels[client_indices[client_id]]
    unique, counts_per_class = np.unique(
        client_labels, return_counts=True
    )
    distribution = dict(zip(unique.tolist(), counts_per_class.tolist()))

    print(f"  [NON-IID] Client {client_id} | "
          f"Total: {len(client_indices[client_id])} samples | "
          f"Distribution: {distribution}")

    subset = Subset(trainset, client_indices[client_id])
    return DataLoader(subset, batch_size=32, shuffle=True)