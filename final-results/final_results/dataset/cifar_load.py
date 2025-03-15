import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_cifar_dataset_50(partition_id=0, num_partitions=1):
    """_summary_

    Args:
        partition_id (int, optional): _description_. Defaults to 0.
        num_partitions (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    transform = transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
    ])
    
    training_set = torchvision.datasets.CIFAR100(
        root='./dataset', train=True, download=True, transform=transform
    )
    
    testing_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    
    training_idx = np.where(np.array(training_set.targets) < 50)[0]
    testing_idx = np.where(np.array(testing_set.targets) < 50)[0]
    
    training_set = Subset(training_set, training_idx)
    testing_set = Subset(testing_set, testing_idx)
    
    num_train = len(training_set)
    idxs = np.random.permutation(num_train)
    batch_size = int(num_train/num_partitions)
    partition_idxs = idxs[partition_id * batch_size:(partition_id+1)*batch_size]
    
    training_set = Subset(training_set, partition_idxs)
    
    training_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=2)
    testing_loader = DataLoader(testing_set, batch_size=32, shuffle=False, num_workers=2)
    
    return training_loader, testing_loader