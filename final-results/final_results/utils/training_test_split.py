import torch
import torch.nn as nn
import torch.optim as optim 
import time
from typing import Dict, List, Tuple

def train(
    model: nn.Module,
    training_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    use_dp: bool = False,
    noise_multiplier: float = 1.0,
    max_gradient_norm: float = 1.0,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    if use_dp:
        from ..security.differential_privacy import apply_differential_privacy
        model, optimizer, training_loader, _ = apply_differential_privacy(
            model,optimizer,training_loader,noise_multiplier=noise_multiplier,max_graient_norm=max_gradient_norm
        )
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            running_loss += loss.item()
    training_time = time.time() - start_time
    return running_loss/len(training_loader), training_time       

def test(
    model: nn.Module,
    testing_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float,float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    correct = 0
    total = 0
    loss = 0.0
    
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct/total) * 100
    loss /= len(testing_loader)
    
    return loss, accuracy