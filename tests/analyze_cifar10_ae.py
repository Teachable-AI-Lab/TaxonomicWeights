"""
Analysis script for CIFAR-10 Taxonomic Autoencoder

Loads a trained model and performs various analyses:
- Filter visualization at each hierarchy level (encoder Conv & decoder Deconv)
- Latent space sparsity analysis
- Reconstruction quality metrics
- Feature activation patterns
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CIFAR10TaxonAutoencoder
from src.utils.dataloader import CIFAR10Loader


def load_model(checkpoint_path, latent_dim=256, temperature=1.0, device='cuda'):
    """Load trained model from checkpoint."""
    model = CIFAR10TaxonAutoencoder(latent_dim=latent_dim, temperature=temperature)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.6f}")
    print(f"Test loss: {checkpoint['test_loss']:.6f}")
    return model, checkpoint


def visualize_taxonconv_filters(model, save_dir, layer_name='taxon_conv1', n_cols=8):
    """Visualize hierarchical filters from a TaxonConv layer."""
    
    # Get the TaxonConv layer
    if layer_name == 'taxon_conv1':
        taxon_conv = model.encoder.taxon_conv1
    elif layer_name == 'taxon_conv2':
        taxon_conv = model.encoder.taxon_conv2
    elif layer_name == 'taxon_conv3':
        taxon_conv = model.encoder.taxon_conv3
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
    
    # Get hierarchy weights
    weights = taxon_conv.get_hierarchy_weights()
    
    # Create subdirectory for this layer
    layer_dir = os.path.join(save_dir, f'{layer_name}_filters')
    os.makedirs(layer_dir, exist_ok=True)
    
    # Visualize each level
    for level_idx, w_tensor in enumerate(weights):
        w_np = w_tensor.detach().cpu().numpy()
        n_filters, in_ch, k, _ = w_np.shape
        
        # Normalize per filter
        mins = w_np.min(axis=(1, 2, 3), keepdims=True)
        maxs = w_np.max(axis=(1, 2, 3), keepdims=True)
        w_norm = (w_np - mins) / (maxs - mins + 1e-5)
        
        # Grid setup
        n_rows = int(np.ceil(n_filters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(n_filters):
            filt = w_norm[i]
            if in_ch > 1:
                # For RGB, show as color image
                filt = np.transpose(filt, (1, 2, 0))
            else:
                # For grayscale
                filt = filt.squeeze()
            
            axes[i].imshow(filt, cmap='gray' if in_ch == 1 else None)
            axes[i].axis('off')
        
        # Turn off extra axes
        for ax in axes[n_filters:]:
            ax.axis('off')
        
        plt.suptitle(f'{layer_name} Level {level_idx} ({n_filters} filters, {in_ch}ch, {k}×{k})')
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, f'level_{level_idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {layer_name} level {level_idx} ({n_filters} filters)")
    
    print(f"Conv filter visualizations saved to {layer_dir}")


def visualize_taxondeconv_filters(model, save_dir, layer_name='taxon_deconv1', n_cols=8):
    """Visualize hierarchical filters from a TaxonDeconv layer."""
    
    # Get the TaxonDeconv layer
    if layer_name == 'taxon_deconv1':
        taxon_deconv = model.decoder.taxon_deconv1
    elif layer_name == 'taxon_deconv2':
        taxon_deconv = model.decoder.taxon_deconv2
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
    
    # Get hierarchy weights
    weights = taxon_deconv.get_hierarchy_weights()
    
    # Create subdirectory for this layer
    layer_dir = os.path.join(save_dir, f'{layer_name}_filters')
    os.makedirs(layer_dir, exist_ok=True)
    
    # Visualize each level
    for level_idx, w_tensor in enumerate(weights):
        w_np = w_tensor.detach().cpu().numpy()
        in_ch, out_ch, k, _ = w_np.shape
        
        # For deconv, we visualize a sample of filters (since there can be many)
        # Show first n_cols*n_cols filters
        max_filters = min(out_ch, n_cols * 8)
        
        # Normalize per filter (across input channels)
        w_np = w_np[:, :max_filters, :, :]  # Take subset
        
        # For visualization, average across input channels or show first input channel
        if in_ch > 3:
            # Too many input channels, show first channel
            w_display = w_np[0, :, :, :]  # (out_ch, k, k)
        else:
            # Few input channels, can display
            w_display = w_np.transpose(1, 2, 3, 0)  # (out_ch, k, k, in_ch)
        
        # Normalize
        mins = w_display.min(axis=tuple(range(1, w_display.ndim)), keepdims=True)
        maxs = w_display.max(axis=tuple(range(1, w_display.ndim)), keepdims=True)
        w_norm = (w_display - mins) / (maxs - mins + 1e-5)
        
        # Grid setup
        n_rows = int(np.ceil(max_filters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(max_filters):
            filt = w_norm[i]
            if filt.ndim == 3 and filt.shape[2] in [1, 3]:
                # Color or grayscale
                if filt.shape[2] == 1:
                    filt = filt.squeeze()
            axes[i].imshow(filt, cmap='gray' if filt.ndim == 2 else None)
            axes[i].axis('off')
        
        # Turn off extra axes
        for ax in axes[max_filters:]:
            ax.axis('off')
        
        plt.suptitle(f'{layer_name} Level {level_idx} ({out_ch} filters, {in_ch}→1ch, {k}×{k})')
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, f'level_{level_idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {layer_name} level {level_idx} (showing {max_filters}/{out_ch} filters)")
    
    print(f"Deconv filter visualizations saved to {layer_dir}")


def analyze_latent_sparsity(model, data_loader, device, save_dir, num_batches=50):
    """Analyze sparsity and statistics of the latent space."""
    
    model.eval()
    all_latents = []
    
    print(f"Encoding {num_batches} batches for latent space analysis...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(data_loader)):
            if i >= num_batches:
                break
            images = images.to(device)
            latents = model.encode(images)
            all_latents.append(latents.cpu().numpy())
    
    all_latents = np.concatenate(all_latents, axis=0)
    print(f"Collected {all_latents.shape[0]} latent vectors")
    
    # Compute statistics
    mean_activation = np.mean(np.abs(all_latents), axis=0)
    std_activation = np.std(all_latents, axis=0)
    
    # Sparsity metrics
    sparsity_per_sample = np.mean(np.abs(all_latents) < 0.1, axis=1)
    mean_sparsity = np.mean(sparsity_per_sample)
    
    # L0 norm (number of non-zero/significant activations)
    l0_norm = np.sum(np.abs(all_latents) > 0.1, axis=1)
    mean_l0 = np.mean(l0_norm)
    
    # L1 and L2 norms
    l1_norm = np.sum(np.abs(all_latents), axis=1)
    l2_norm = np.sqrt(np.sum(all_latents**2, axis=1))
    
    print(f"\nLatent Space Statistics:")
    print(f"  Average sparsity: {mean_sparsity:.4f} (fraction near-zero)")
    print(f"  Average L0 norm: {mean_l0:.2f} active dimensions")
    print(f"  Average L1 norm: {np.mean(l1_norm):.4f}")
    print(f"  Average L2 norm: {np.mean(l2_norm):.4f}")
    
    # Save statistics
    stats = {
        'mean_sparsity': float(mean_sparsity),
        'mean_l0': float(mean_l0),
        'mean_l1': float(np.mean(l1_norm)),
        'mean_l2': float(np.mean(l2_norm)),
        'latent_dim': all_latents.shape[1],
        'num_samples': all_latents.shape[0]
    }
    
    np.savez(
        os.path.join(save_dir, 'latent_statistics.npz'),
        all_latents=all_latents,
        mean_activation=mean_activation,
        std_activation=std_activation,
        sparsity_per_sample=sparsity_per_sample,
        l0_norm=l0_norm,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        **stats
    )
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Mean activation per dimension
    axes[0, 0].bar(range(len(mean_activation)), mean_activation)
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Mean |Activation|')
    axes[0, 0].set_title('Mean Activation per Dimension')
    
    # Std activation per dimension
    axes[0, 1].bar(range(len(std_activation)), std_activation)
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Std Activation')
    axes[0, 1].set_title('Activation Std per Dimension')
    
    # Sparsity distribution
    axes[0, 2].hist(sparsity_per_sample, bins=50, edgecolor='black')
    axes[0, 2].axvline(mean_sparsity, color='red', linestyle='--', label=f'Mean: {mean_sparsity:.3f}')
    axes[0, 2].set_xlabel('Sparsity (fraction near-zero)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Sparsity Distribution')
    axes[0, 2].legend()
    
    # L0 norm distribution
    axes[1, 0].hist(l0_norm, bins=50, edgecolor='black')
    axes[1, 0].axvline(mean_l0, color='red', linestyle='--', label=f'Mean: {mean_l0:.1f}')
    axes[1, 0].set_xlabel('L0 Norm (active dimensions)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('L0 Norm Distribution')
    axes[1, 0].legend()
    
    # L1 norm distribution
    axes[1, 1].hist(l1_norm, bins=50, edgecolor='black')
    axes[1, 1].axvline(np.mean(l1_norm), color='red', linestyle='--', label=f'Mean: {np.mean(l1_norm):.2f}')
    axes[1, 1].set_xlabel('L1 Norm')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('L1 Norm Distribution')
    axes[1, 1].legend()
    
    # L2 norm distribution
    axes[1, 2].hist(l2_norm, bins=50, edgecolor='black')
    axes[1, 2].axvline(np.mean(l2_norm), color='red', linestyle='--', label=f'Mean: {np.mean(l2_norm):.2f}')
    axes[1, 2].set_xlabel('L2 Norm')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('L2 Norm Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Latent space analysis saved to {save_dir}")


def analyze_reconstruction_quality(model, data_loader, device, save_dir, num_batches=20):
    """Analyze reconstruction quality metrics."""
    
    model.eval()
    mse_losses = []
    mae_losses = []
    
    print(f"Computing reconstruction metrics on {num_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(data_loader)):
            if i >= num_batches:
                break
            images = images.to(device)
            reconstructed = model(images)
            
            mse = ((images - reconstructed) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            mae = torch.abs(images - reconstructed).mean(dim=(1, 2, 3)).cpu().numpy()
            
            mse_losses.extend(mse)
            mae_losses.extend(mae)
    
    mse_losses = np.array(mse_losses)
    mae_losses = np.array(mae_losses)
    
    print(f"\nReconstruction Quality:")
    print(f"  Mean MSE: {np.mean(mse_losses):.6f}")
    print(f"  Mean MAE: {np.mean(mae_losses):.6f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(mse_losses, bins=50, edgecolor='black')
    axes[0].axvline(np.mean(mse_losses), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(mse_losses):.4f}')
    axes[0].set_xlabel('MSE Loss')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MSE Distribution')
    axes[0].legend()
    
    axes[1].hist(mae_losses, bins=50, edgecolor='black')
    axes[1].axvline(np.mean(mae_losses), color='red', linestyle='--',
                    label=f'Mean: {np.mean(mae_losses):.4f}')
    axes[1].set_xlabel('MAE Loss')
    axes[1].set_ylabel('Count')
    axes[1].set_title('MAE Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    np.savez(
        os.path.join(save_dir, 'reconstruction_metrics.npz'),
        mse_losses=mse_losses,
        mae_losses=mae_losses
    )
    
    print(f"Reconstruction metrics saved to {save_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trained CIFAR-10 Taxonomic Autoencoder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Latent dimension (must match training)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature (must match training)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for analysis')
    parser.add_argument('--data-root', type=str, default='./data/cifar10',
                        help='CIFAR-10 data directory')
    
    args = parser.parse_args()
    
    # Create output directory
    save_dir = f'outputs/analysis/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("CIFAR-10 Taxonomic Autoencoder Analysis")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(args.checkpoint, args.latent_dim, args.temperature, device)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    loader = CIFAR10Loader(batch_size=args.batch_size, root=args.data_root)
    train_loader, test_loader = loader.get_loaders()
    
    # Analysis 1: Visualize encoder Conv filters
    print("\n" + "=" * 60)
    print("1. Visualizing Encoder TaxonConv Filters")
    print("=" * 60)
    for layer_name in ['taxon_conv1', 'taxon_conv2', 'taxon_conv3']:
        print(f"\nProcessing {layer_name}...")
        visualize_taxonconv_filters(model, save_dir, layer_name=layer_name)
    
    # Analysis 2: Visualize decoder Deconv filters
    print("\n" + "=" * 60)
    print("2. Visualizing Decoder TaxonDeconv Filters")
    print("=" * 60)
    for layer_name in ['taxon_deconv1', 'taxon_deconv2']:
        print(f"\nProcessing {layer_name}...")
        visualize_taxondeconv_filters(model, save_dir, layer_name=layer_name)
    
    # Analysis 3: Latent space sparsity
    print("\n" + "=" * 60)
    print("3. Analyzing Latent Space")
    print("=" * 60)
    analyze_latent_sparsity(model, test_loader, device, save_dir)
    
    # Analysis 4: Reconstruction quality
    print("\n" + "=" * 60)
    print("4. Analyzing Reconstruction Quality")
    print("=" * 60)
    analyze_reconstruction_quality(model, test_loader, device, save_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
