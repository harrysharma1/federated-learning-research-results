"""normal-run: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.datasets import CIFAR100

class LeNet(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, activation_function=nn.Sigmoid):
        super(LeNet, self).__init__()
        act = activation_function
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(768, 50)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cifar100_train = None
cifar100_test = None

def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global cifar100_train, cifar100_test
    
    transform = Compose([
        ToTensor(),
        Normalize((0.5,0.5,0.5),(0.5,0.5,0.5,))
    ])
    
    if cifar100_train is None:
        cifar100_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
        cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    training_idxs = [idx for idx, (_, label) in enumerate(cifar100_train) if 0<=label<50]
    testing_idxs = [idx for idx, (_, label) in enumerate(cifar100_test) if 0<=label<50]
    
    training_subset = Subset(cifar100_train, training_idxs)
    testing_subset = Subset(cifar100_test, testing_idxs)
    
    n_train = len(training_subset)
    n_test = len(testing_subset)
    
    training_partition_size = n_train // num_partitions
    testing_partition_size = n_test // num_partitions
    
    training_start_idx = partition_id * training_partition_size
    training_end_idx = min(training_start_idx + training_partition_size, n_train)
    
    testing_start_idx = partition_id * testing_partition_size
    testing_end_idx = min(testing_start_idx + testing_partition_size, n_test)
    
    client_training_indices = list(range(training_start_idx, training_end_idx))
    client_testing_indices = list(range(testing_start_idx, testing_end_idx))
    
    training_partition = Subset(training_subset, client_training_indices)
    testing_partition = Subset(testing_subset, client_testing_indices)
    
    training_loader = DataLoader(
        training_partition,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: {
            'img': torch.stack([item[0] for item in batch]),
            'label': torch.tensor([item[1] for item in batch])
        }
    )
    
    testing_loader = DataLoader(
        testing_partition,
        batch_size=32,
        collate_fn=lambda batch: {
            'img': torch.stack([item[0] for item in batch]),
            'label': torch.tensor([item[1] for item in batch])
        }
    )
    
    return training_loader, testing_loader
    
def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def init_weights(net):
    def weight_init(m):
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, 'bias'):
            m.bias.data.uniform_(-0.5, 0.5)
    
    net.apply(weight_init)
    return net

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
