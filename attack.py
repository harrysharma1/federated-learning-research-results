import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import copy
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Differential Privacy noise addition
def add_differential_privacy_noise(gradients, noise_scale=0.01, clip_norm=1.0):
    """Add Gaussian noise to gradients for differential privacy"""
    # Clone gradients to avoid modifying the original
    dp_gradients = [g.clone() for g in gradients]
    
    # Clip gradients by norm (clip each parameter's gradient)
    for i, g in enumerate(dp_gradients):
        if g is not None:
            if torch.norm(g) > clip_norm:
                dp_gradients[i] = g * clip_norm / torch.norm(g)
    
    # Add Gaussian noise
    for i, g in enumerate(dp_gradients):
        if g is not None:
            noise = torch.randn_like(g) * noise_scale
            dp_gradients[i] = g + noise
    
    return dp_gradients

# Secure Aggregation Protocol simulation
def generate_pairwise_masks(num_participants):
    """Generate pairwise masks for each participant."""
    masks = []
    for _ in range(num_participants):
        masks.append(torch.randn(1))  # Random mask for each participant
    return masks

def participant_contribution(local_gradient, participant_id, num_participants):
    """Each participant generates a masked gradient."""
    masks = generate_pairwise_masks(num_participants)
    
    # Apply the mask to the local gradient
    masked_gradient = local_gradient + masks[participant_id]
    
    return masked_gradient, masks

def secure_aggregator(masked_gradients):
    """Aggregate the masked gradients."""
    # Sum the masked gradients
    aggregated_result = sum(masked_gradients)
    
    return aggregated_result

def apply_secure_aggregation(gradients, num_parties=10):
    """Simulate secure aggregation by adding masks and returning both masked gradients and masks."""
    
    # Create a list to hold masked gradients
    masked_gradients = [g.clone() if g is not None else None for g in gradients]
    
    # Generate random masks for each party and parameter
    masks = []
    for i in range(num_parties):
        party_masks = [torch.randn_like(g) if g is not None else None for g in gradients]
        masks.append(party_masks)
    
    # Add masks to gradients
    for party_mask in masks:
        for i, mask in enumerate(party_mask):
            if mask is not None:
                masked_gradients[i] += mask
    
    # Aggregate the masked gradients (simulating the aggregation step)
    aggregated_result = sum(masked_gradients)

    # Add small error to simulate potential numerical imprecision in real systems
    for i, g in enumerate(masked_gradients):
        if g is not None:
            error = torch.randn_like(g) * 1e-5
            masked_gradients[i] = g + error
    
    print(f"Mask: {mask}")
    print(f"Aggregate result: {aggregated_result}")
    # Return both masked gradients and masks for further analysis
    return masked_gradients


def compute_metrics(original_image, reconstructed_image):
    """Compute PSNR, MSE, and SSIM between original and reconstructed images"""
    # Convert tensors to numpy arrays
    if torch.is_tensor(original_image):
        original_np = original_image.detach().cpu().numpy().transpose(1, 2, 0)
        reconstructed_np = reconstructed_image.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        original_np = original_image
        reconstructed_np = reconstructed_image
    
    # Ensure values are in the valid range [0, 1] for metrics
    original_np = np.clip(original_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    
    # Compute metrics
    mse_value = mse(original_np, reconstructed_np)
    psnr_value = psnr(original_np, reconstructed_np, data_range=1.0)
    ssim_value = ssim(original_np, reconstructed_np, data_range=1.0, channel_axis=2, multichannel=True)
    
    return psnr_value, mse_value, ssim_value

def deep_leakage_from_gradients(model, origin_grad, input_size=(3, 32, 32), label_size=100,
                               num_iters=300, lr=1.0, debug=False):
    """Reconstruct data from gradients using Deep Leakage from Gradients attack"""
    # Initialize dummy data and label (no cuda)
    dummy_data = torch.randn(input_size).requires_grad_(True)
    dummy_label = torch.randn(label_size).requires_grad_(True)
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    history = []
    for iters in range(num_iters):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data.unsqueeze(0))
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label.unsqueeze(0), dim=-1))
            dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            # Calculate the L2 distance between dummy gradients and original gradients
            grad_diff = 0
            for dg, og in zip(dummy_gradients, origin_grad):
                if og is not None:  # Skip None gradients (frozen layers)
                    grad_diff += ((dg - og) ** 2).sum()
            
            # Add retain_graph=True to fix the double backward issue
            grad_diff.backward(retain_graph=True)
            if debug and iters % 50 == 0:
                print(f"Iteration {iters}, Gradient Difference: {grad_diff.item()}")
            return grad_diff
        
        optimizer.step(closure)
        
        if iters % 50 == 0:
            current = dummy_data.clone().detach()
            # Convert tensor to numpy for visualization
            current_np = current.numpy()
            # Clip values to [0,1] range for proper visualization
            current_np = np.clip(current_np, 0, 1)
            history.append(current_np)
    
    elapsed_time = time.time() - start_time
    
    return dummy_data, dummy_label, elapsed_time, history

def run_dlg_experiment(original_data, original_label, model, defense_type='none', 
                      noise_scale=0.01, clip_norm=1.0, num_parties=10):
    """Run DLG attack with different defense mechanisms"""
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass with the original data
    outputs = model(original_data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, original_label)
    
    # Compute gradients
    original_gradients = torch.autograd.grad(loss, model.parameters())
    
    # Apply defense mechanisms based on defense_type
    protected_gradients = original_gradients
    
    if defense_type == 'dp':
        protected_gradients = add_differential_privacy_noise(original_gradients, noise_scale, clip_norm)
    elif defense_type == 'secagg':
        protected_gradients = apply_secure_aggregation(original_gradients, num_parties)
    elif defense_type == 'both':
        dp_gradients = add_differential_privacy_noise(original_gradients, noise_scale, clip_norm)
        protected_gradients = apply_secure_aggregation(dp_gradients, num_parties)
    
    # Run DLG attack
    reconstructed_data, reconstructed_label, elapsed_time, history = deep_leakage_from_gradients(
        model, protected_gradients, 
        input_size=original_data.squeeze().size(),
        label_size=100,
        num_iters=300
    )
    
    # Calculate metrics
    psnr_value, mse_value, ssim_value = compute_metrics(
        original_data.squeeze().cpu(), 
        reconstructed_data.cpu()
    )
    
    return {
        'reconstructed_data': reconstructed_data,
        'reconstructed_label': reconstructed_label,
        'time': elapsed_time,
        'psnr': psnr_value,
        'mse': mse_value,
        'ssim': ssim_value,
        'history': history
    }

def visualize_reconstruction(original, reconstructed, defense_type, metrics):
    """Visualize original and reconstructed images with metrics"""
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original.transpose(1, 2, 0))
    plt.title("Original Image")
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.transpose(1, 2, 0))
    plt.title(f"Reconstructed ({defense_type})\nPSNR: {metrics['psnr']:.2f}, MSE: {metrics['mse']:.4f}, SSIM: {metrics['ssim']:.4f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"dlg_reconstruction_{defense_type}.png")
    plt.close()

def run_full_experiment():
    """Run DLG attack against different defense mechanisms on multiple CIFAR-100 images"""
    # Load CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    cifar_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create a simple model (no cuda)
    model = SimpleNet()
    
    # Parameters
    num_images =1  # Reduced from 100 to make CPU execution faster
    defense_types = ['none', 'dp', 'secagg', 'both']
    results = {defense: [] for defense in defense_types}
    
    # Select random images
    indices = random.sample(range(len(cifar_dataset)), num_images)
    
    # Run experiment for each image and defense mechanism
    for i, idx in enumerate(indices):
        print(f"Processing image {i+1}/{num_images} (index {idx})...")
        
        # Get image and label (no cuda)
        img, label = cifar_dataset[idx]
        img = img.unsqueeze(0)
        label = torch.tensor([label])
        
        # Run DLG attack with different defenses
        for defense in defense_types:
            print(f"  Applying {defense} defense...")
            result = run_dlg_experiment(img, label, model, defense_type=defense)
            results[defense].append(result)
            
            # Visualize first 5 images
            if i < 5:
                visualize_reconstruction(
                    img.squeeze().cpu().numpy(), 
                    result['reconstructed_data'].detach().cpu().numpy(),
                    defense,
                    {'psnr': result['psnr'], 'mse': result['mse'], 'ssim': result['ssim']}
                )
    
    # Calculate and print average metrics
    print("\nAverage Metrics by Defense Type:")
    print("-" * 60)
    print(f"{'Defense':<10} | {'Time (s)':<10} | {'PSNR':<10} | {'MSE':<10} | {'SSIM':<10}")
    print("-" * 60)
    
    for defense in defense_types:
        avg_time = np.mean([r['time'] for r in results[defense]])
        avg_psnr = np.mean([r['psnr'] for r in results[defense]])
        avg_mse = np.mean([r['mse'] for r in results[defense]])
        avg_ssim = np.mean([r['ssim'] for r in results[defense]])
        
        print(f"{defense:<10} | {avg_time:<10.2f} | {avg_psnr:<10.2f} | {avg_mse:<10.4f} | {avg_ssim:<10.4f}")
    
    # Save detailed results
    with open("dlg_defense_results.txt", "w") as f:
        f.write("Detailed Results by Defense Type:\n")
        f.write("-" * 80 + "\n")
        
        for defense in defense_types:
            f.write(f"\n{defense.upper()} DEFENSE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Image':<10} | {'Time (s)':<10} | {'PSNR':<10} | {'MSE':<10} | {'SSIM':<10}\n")
            f.write("-" * 60 + "\n")
            
            for i, r in enumerate(results[defense]):
                f.write(f"{i:<10} | {r['time']:<10.2f} | {r['psnr']:<10.2f} | {r['mse']:<10.4f} | {r['ssim']:<10.4f}\n")
    
    # Create comparative visualizations
    plot_comparative_metrics(results)
    
    return results

def plot_comparative_metrics(results):
    """Create plots comparing metrics across different defense mechanisms"""
    defense_types = list(results.keys())
    
    # Prepare data
    metrics = {
        'time': [np.mean([r['time'] for r in results[d]]) for d in defense_types],
        'psnr': [np.mean([r['psnr'] for r in results[d]]) for d in defense_types],
        'mse': [np.mean([r['mse'] for r in results[d]]) for d in defense_types],
        'ssim': [np.mean([r['ssim'] for r in results[d]]) for d in defense_types]
    }
    
    # Create bar charts
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time plot
    axs[0, 0].bar(defense_types, metrics['time'], color='blue')
    axs[0, 0].set_title('Average Execution Time (seconds)')
    axs[0, 0].set_ylabel('Time (s)')
    
    # PSNR plot
    axs[0, 1].bar(defense_types, metrics['psnr'], color='green')
    axs[0, 1].set_title('Average PSNR (dB)')
    axs[0, 1].set_ylabel('PSNR (dB)')
    
    # MSE plot
    axs[1, 0].bar(defense_types, metrics['mse'], color='red')
    axs[1, 0].set_title('Average MSE')
    axs[1, 0].set_ylabel('MSE')
    
    # SSIM plot
    axs[1, 1].bar(defense_types, metrics['ssim'], color='purple')
    axs[1, 1].set_title('Average SSIM')
    axs[1, 1].set_ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig("dlg_defense_metrics_comparison.png")
    plt.close()
    
    # Create box plots for distribution analysis
    metrics_boxplot = {
        'psnr': {d: [r['psnr'] for r in results[d]] for d in defense_types},
        'mse': {d: [r['mse'] for r in results[d]] for d in defense_types},
        'ssim': {d: [r['ssim'] for r in results[d]] for d in defense_types}
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # PSNR boxplot
    axs[0].boxplot([metrics_boxplot['psnr'][d] for d in defense_types])
    axs[0].set_xticklabels(defense_types)
    axs[0].set_title('PSNR Distribution by Defense Type')
    axs[0].set_ylabel('PSNR (dB)')
    
    # MSE boxplot
    axs[1].boxplot([metrics_boxplot['mse'][d] for d in defense_types])
    axs[1].set_xticklabels(defense_types)
    axs[1].set_title('MSE Distribution by Defense Type')
    axs[1].set_ylabel('MSE')
    
    # SSIM boxplot
    axs[2].boxplot([metrics_boxplot['ssim'][d] for d in defense_types])
    axs[2].set_xticklabels(defense_types)
    axs[2].set_title('SSIM Distribution by Defense Type')
    axs[2].set_ylabel('SSIM')
    
    plt.tight_layout()
    plt.savefig("dlg_defense_metrics_distributions.png")
    plt.close()

if __name__ == "__main__":
    print("Starting DLG attack experiment (CPU version)...")
    results = run_full_experiment()
    print("Experiment completed. Results saved to disk.")