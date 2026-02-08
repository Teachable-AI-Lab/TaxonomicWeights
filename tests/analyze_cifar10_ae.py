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
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CIFAR10TaxonAutoencoder
from src.utils.dataloader import CIFAR10Loader


def load_model(checkpoint_path, latent_dim=256, temperature=1.0, device='cuda',
               encoder_kernel_sizes=None, decoder_kernel_sizes=None,
               encoder_strides=None, decoder_strides=None, 
               encoder_n_layers=None, decoder_n_layers=None, use_maxpool=True):
    """Load trained model from checkpoint."""
    model = CIFAR10TaxonAutoencoder(
        latent_dim=latent_dim, 
        temperature=temperature,
        encoder_kernel_sizes=encoder_kernel_sizes,
        decoder_kernel_sizes=decoder_kernel_sizes,
        encoder_strides=encoder_strides,
        decoder_strides=decoder_strides,
        encoder_n_layers=encoder_n_layers,
        decoder_n_layers=decoder_n_layers,
        use_maxpool=use_maxpool
    )
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
            
            if in_ch == 1:
                filt = filt.squeeze()
                cmap = 'gray'
            elif in_ch == 3:
                filt = np.transpose(filt, (1, 2, 0))
                cmap = None
            else:
                # Too many channels - average across all input channels
                filt = filt.mean(axis=0)
                cmap = 'viridis'
            
            axes[i].imshow(filt, cmap=cmap)
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
    elif layer_name == 'taxon_deconv3':
        taxon_deconv = model.decoder.taxon_deconv3
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
        
        # For deconv, visualize a subset of output filters
        max_filters = min(out_ch, n_cols * 8)
        w_subset = w_np[:, :max_filters, :, :]
        
        # Average across input channels for visualization
        w_display = w_subset.mean(axis=0)
        
        # Normalize
        mins = w_display.min(axis=(1, 2), keepdims=True)
        maxs = w_display.max(axis=(1, 2), keepdims=True)
        w_norm = (w_display - mins) / (maxs - mins + 1e-5)
        
        # Grid setup
        n_rows = int(np.ceil(max_filters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i in range(max_filters):
            axes[i].imshow(w_norm[i], cmap='viridis')
            axes[i].axis('off')
        
        # Turn off extra axes
        for ax in axes[max_filters:]:
            ax.axis('off')
        
        plt.suptitle(f'{layer_name} Level {level_idx} ({out_ch} filters, avg of {in_ch} in_ch, {k}×{k})')
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


def visualize_multiple_reconstructions(model, data_loader, device, save_dir, num_images=8, num_reconstructions=8):
    """Visualize reconstructions for multiple sets of images in grids (originals on top, reconstructions below)."""
    
    model.eval()
    
    # Create subdirectory for reconstructions
    recon_dir = os.path.join(save_dir, 'multiple_reconstructions')
    os.makedirs(recon_dir, exist_ok=True)
    
    # Get multiple batches to create multiple sets
    data_iter = iter(data_loader)
    
    # Create num_reconstructions sets (each with num_images images)
    for set_idx in range(num_reconstructions):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            # Reset iterator if we run out
            data_iter = iter(data_loader)
            images, _ = next(data_iter)
        
        images = images[:num_images].to(device)
        
        # Create figure: 2 rows (originals + reconstructions), num_images columns
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.5, 5))
        
        with torch.no_grad():
            for i in range(num_images):
                img = images[i:i+1]
                
                # Original image (unnormalize from [-1, 1] to [0, 1])
                img_display = (img.cpu() * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
                axes[0, i].imshow(img_display)
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
                
                # Generate single reconstruction
                reconstructed = model(img)
                recon_display = (reconstructed.cpu() * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
                axes[1, i].imshow(recon_display)
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_title('Reconstruction', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, f'reconstructions_set_{set_idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved reconstruction set {set_idx+1}/{num_reconstructions} with {num_images} images")
    
    print(f"Reconstructions saved to {recon_dir}")


def visualize_layer_activations(model, data_loader, device, save_dir, num_images=3):
    """Visualize activations at each layer for random images."""
    
    model.eval()
    
    # Get random images
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        for img_idx in range(num_images):
            img = images[img_idx:img_idx+1]
            
            # Create directory for this image
            img_dir = os.path.join(save_dir, f'activations_image_{img_idx+1}')
            os.makedirs(img_dir, exist_ok=True)
            
            # Save original image
            img_display = (img.cpu() * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
            plt.figure(figsize=(4, 4))
            plt.imshow(img_display)
            plt.axis('off')
            plt.title(f'Original Image {img_idx+1}')
            plt.savefig(os.path.join(img_dir, 'original.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Encoder activations
            x = img
            
            # TaxonConv1
            x = model.encoder.taxon_conv1(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'encoder_taxon_conv1.png'), 'Encoder TaxonConv1', max_maps=16)
            x = F.relu(x)
            if model.encoder.use_maxpool:
                x = F.max_pool2d(x, model.encoder.pool_stride1)
            else:
                x = F.avg_pool2d(x, model.encoder.pool_stride1)
            
            # TaxonConv2
            x = model.encoder.taxon_conv2(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'encoder_taxon_conv2.png'), 'Encoder TaxonConv2', max_maps=16)
            x = F.relu(x)
            if model.encoder.use_maxpool:
                x = F.max_pool2d(x, model.encoder.pool_stride2)
            else:
                x = F.avg_pool2d(x, model.encoder.pool_stride2)
            
            # TaxonConv3
            x = model.encoder.taxon_conv3(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'encoder_taxon_conv3.png'), 'Encoder TaxonConv3', max_maps=16)
            
            # Latent (flattened, so visualize as 1D)
            latent = model.encode(img)
            plt.figure(figsize=(12, 3))
            plt.bar(range(latent.shape[1]), latent.cpu().squeeze().numpy())
            plt.xlabel('Latent Dimension')
            plt.ylabel('Activation')
            plt.title(f'Latent Space (dim={latent.shape[1]})')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'latent.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Decoder activations - need to check decoder structure
            # Get latent and reconstruct through decoder layers
            x = model.decoder.fc(latent)
            
            # Reshape to initial spatial dimensions
            x = x.view(-1, model.decoder.initial_channels, model.decoder.initial_spatial_size, model.decoder.initial_spatial_size)
            
            # TaxonDeconv1 (includes upsampling via transposed conv)
            x = model.decoder.taxon_deconv1(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'decoder_taxon_deconv1.png'), 'Decoder TaxonDeconv1', max_maps=16)
            x = F.relu(x)
            
            # TaxonDeconv2 (includes upsampling via transposed conv)
            x = model.decoder.taxon_deconv2(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'decoder_taxon_deconv2.png'), 'Decoder TaxonDeconv2', max_maps=16)
            x = F.relu(x)
            
            # TaxonDeconv3 (final layer to RGB)
            x = model.decoder.taxon_deconv3(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'decoder_taxon_deconv3.png'), 'Decoder TaxonDeconv3 (RGB)', max_maps=3)
            
            # Final reconstruction
            reconstructed = model(img)
            recon_display = (reconstructed.cpu() * 0.5 + 0.5).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
            plt.figure(figsize=(4, 4))
            plt.imshow(recon_display)
            plt.axis('off')
            plt.title('Reconstruction')
            plt.savefig(os.path.join(img_dir, 'reconstruction.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Activations for image {img_idx+1} saved to {img_dir}")
    
    print(f"Layer activations saved to {save_dir}")


def visualize_feature_maps(feature_tensor, save_path, title, max_maps=16, nrow=4):
    """Helper to visualize feature maps from a layer."""
    
    # feature_tensor: (1, C, H, W)
    maps = feature_tensor[0].detach().cpu()
    num_maps = min(maps.shape[0], max_maps)
    
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 2, nrow * 2))
    axes = axes.flatten()
    
    for i in range(num_maps):
        ax = axes[i]
        fmap = maps[i].numpy()
        # Normalize for visualization
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        ax.imshow(fmap, cmap='viridis')
        ax.axis('off')
    
    # Turn off extra axes
    for ax in axes[num_maps:]:
        ax.axis('off')
    
    plt.suptitle(f'{title} (showing {num_maps}/{maps.shape[0]} feature maps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trained CIFAR-10 Taxonomic Autoencoder')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Latent dimension (must match training)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature (must match training)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for analysis')
    parser.add_argument('--data-root', type=str, default=None,
                        help='CIFAR-10 data directory')
    parser.add_argument('--encoder-kernel-sizes', type=int, nargs='+', default=None,
                        help='Encoder kernel sizes (must match training)')
    parser.add_argument('--decoder-kernel-sizes', type=int, nargs='+', default=None,
                        help='Decoder kernel sizes (must match training)')
    parser.add_argument('--encoder-strides', type=int, nargs='+', default=None,
                        help='Encoder strides (must match training)')
    parser.add_argument('--decoder-strides', type=int, nargs='+', default=None,
                        help='Decoder strides (must match training)')
    parser.add_argument('--use-maxpool', action='store_true', default=False,
                        help='Use max pooling (must match training)')
    parser.add_argument('--use-avgpool', action='store_true', default=False,
                        help='Use average pooling (must match training)')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        batch_size = config['data']['batch_size']
        data_root = config['data']['data_root']
        latent_dim = config['model']['latent_dim']
        temperature = config['model']['temperature']
        encoder_kernel_sizes = config['model']['encoder_kernel_sizes']
        decoder_kernel_sizes = config['model']['decoder_kernel_sizes']
        encoder_strides = config['model']['encoder_strides']
        decoder_strides = config['model']['decoder_strides']
        encoder_n_layers = config['model'].get('encoder_n_layers', None)
        decoder_n_layers = config['model'].get('decoder_n_layers', None)
        use_maxpool = config['model']['use_maxpool']
        
        # Analysis-specific parameters (optional)
        checkpoint_path = config.get('analysis', {}).get('checkpoint_path', None)
        num_latent_batches = config.get('analysis', {}).get('num_latent_batches', 50)
        num_reconstruction_batches = config.get('analysis', {}).get('num_reconstruction_batches', 20)
        num_multiple_recon_images = config.get('analysis', {}).get('num_multiple_recon_images', 8)
        num_reconstructions_per_image = config.get('analysis', {}).get('num_reconstructions_per_image', 8)
        num_activation_images = config.get('analysis', {}).get('num_activation_images', 3)
        save_dir_prefix = config.get('output', {}).get('analysis_save_dir', 'outputs/analysis')
    else:
        # Defaults
        checkpoint_path = None
        batch_size = 128
        data_root = './data/cifar10'
        latent_dim = 256
        temperature = 1.0
        encoder_kernel_sizes = None
        decoder_kernel_sizes = None
        encoder_strides = None
        decoder_strides = None
        encoder_n_layers = None
        decoder_n_layers = None
        use_maxpool = True
        num_latent_batches = 50
        num_reconstruction_batches = 20
        num_multiple_recon_images = 8
        num_reconstructions_per_image = 8
        num_activation_images = 3
        save_dir_prefix = 'outputs/analysis'
    
    # Command line args override config
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.data_root is not None:
        data_root = args.data_root
    if args.latent_dim is not None:
        latent_dim = args.latent_dim
    if args.temperature is not None:
        temperature = args.temperature
    if args.encoder_kernel_sizes is not None:
        encoder_kernel_sizes = args.encoder_kernel_sizes
    if args.decoder_kernel_sizes is not None:
        decoder_kernel_sizes = args.decoder_kernel_sizes
    if args.encoder_strides is not None:
        encoder_strides = args.encoder_strides
    if args.decoder_strides is not None:
        decoder_strides = args.decoder_strides
    
    # Determine pooling type
    if args.use_avgpool:
        use_maxpool = False
    elif args.use_maxpool:
        use_maxpool = True
    
    # Require checkpoint
    if checkpoint_path is None:
        raise ValueError("Checkpoint path must be provided via --checkpoint or config file")
    
    # Create output directory
    save_dir = f'{save_dir_prefix}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("CIFAR-10 Taxonomic Autoencoder Analysis")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {device}")
    print(f"Encoder kernel sizes: {encoder_kernel_sizes}")
    print(f"Decoder kernel sizes: {decoder_kernel_sizes}")
    print(f"Encoder strides: {encoder_strides}")
    print(f"Decoder strides: {decoder_strides}")
    print(f"Use max pooling: {use_maxpool}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(
        checkpoint_path, 
        latent_dim, 
        temperature, 
        device,
        encoder_kernel_sizes=encoder_kernel_sizes,
        decoder_kernel_sizes=decoder_kernel_sizes,
        encoder_strides=encoder_strides,
        decoder_strides=decoder_strides,
        encoder_n_layers=encoder_n_layers,
        decoder_n_layers=decoder_n_layers,
        use_maxpool=use_maxpool
    )
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    loader = CIFAR10Loader(batch_size=batch_size, root=data_root)
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
    for layer_name in ['taxon_deconv1', 'taxon_deconv2', 'taxon_deconv3']:
        print(f"\nProcessing {layer_name}...")
        visualize_taxondeconv_filters(model, save_dir, layer_name=layer_name)
    
    # Analysis 3: Latent space sparsity
    print("\n" + "=" * 60)
    print("3. Analyzing Latent Space")
    print("=" * 60)
    analyze_latent_sparsity(model, test_loader, device, save_dir, num_batches=num_latent_batches)
    
    # Analysis 4: Reconstruction quality
    print("\n" + "=" * 60)
    print("4. Analyzing Reconstruction Quality")
    print("=" * 60)
    analyze_reconstruction_quality(model, test_loader, device, save_dir, num_batches=num_reconstruction_batches)
    
    # Analysis 5: Multiple reconstructions per image
    print("\n" + "=" * 60)
    print("5. Visualizing Multiple Reconstructions")
    print("=" * 60)
    visualize_multiple_reconstructions(model, test_loader, device, save_dir, 
                                      num_images=num_multiple_recon_images, 
                                      num_reconstructions=num_reconstructions_per_image)
    
    # Analysis 6: Layer activations for random images
    print("\n" + "=" * 60)
    print("6. Visualizing Layer Activations")
    print("=" * 60)
    visualize_layer_activations(model, test_loader, device, save_dir, num_images=num_activation_images)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
