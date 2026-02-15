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
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm
from datetime import datetime
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CIFAR10TaxonAutoencoder
from src.model.taxon_layers import TaxonConv, TaxonDeconv
from src.utils.dataloader import CIFAR10Loader


def load_model(checkpoint_path, latent_dim=256, temperature=1.0, device='cuda',
               encoder_kernel_sizes=None, decoder_kernel_sizes=None,
               encoder_strides=None, decoder_strides=None, 
               encoder_n_layers=None, decoder_n_layers=None,
               encoder_n_filters=None, decoder_n_filters=None,
               encoder_layer_types=None, decoder_layer_types=None,
               decoder_paddings=None, decoder_output_paddings=None, 
               use_maxpool=True, random_init_alphas=False,
               alpha_init_distribution="uniform", alpha_init_range=None,
               alpha_init_seed=None):
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
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_layer_types=encoder_layer_types,
        decoder_layer_types=decoder_layer_types,
        decoder_paddings=decoder_paddings,
        decoder_output_paddings=decoder_output_paddings,
        use_maxpool=use_maxpool,
        random_init_alphas=random_init_alphas,
        alpha_init_distribution=alpha_init_distribution,
        alpha_init_range=alpha_init_range,
        alpha_init_seed=alpha_init_seed
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.6f}")
    print(f"Test loss: {checkpoint['test_loss']:.6f}")
    return model, checkpoint


def visualize_taxonconv_filters(model, save_dir, layer_name='encoder_layer_1', n_cols=8):
    """Visualize hierarchical filters from a TaxonConv layer."""
    
    # Get the TaxonConv layer
    if layer_name.startswith('encoder_layer_'):
        layer_idx = int(layer_name.split('_')[-1]) - 1
        taxon_conv = model.encoder.conv_layers[layer_idx]
    elif layer_name == 'final_conv':
        # final_conv is now a regular Conv2D, skip visualization
        print(f"  Skipping {layer_name} - it's a regular Conv2D, not a TaxonConv")
        return
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


def visualize_taxondeconv_filters(model, save_dir, layer_name='decoder_layer_1', n_cols=8):
    """Visualize hierarchical filters from a TaxonDeconv layer."""
    
    # Get the TaxonDeconv layer
    if layer_name.startswith('decoder_layer_'):
        layer_idx = int(layer_name.split('_')[-1]) - 1
        taxon_deconv = model.decoder.deconv_layers[layer_idx]
    else:
        raise ValueError(f"Unknown layer: {layer_name}")
    
    # Get hierarchy weights
    weights = taxon_deconv.get_hierarchy_weights()
    
    # Create subdirectory for this layer
    layer_dir = os.path.join(save_dir, f'{layer_name}_filters')
    os.makedirs(layer_dir, exist_ok=True)
    
    # Visualize each level with cumulative filter indices
    cumulative_filter_count = 0
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
            axes[i].set_title(f'F{cumulative_filter_count + i}', fontsize=8)
        
        # Update cumulative count
        cumulative_filter_count += out_ch
        
        # Turn off extra axes
        for ax in axes[max_filters:]:
            ax.axis('off')
        
        plt.suptitle(f'{layer_name} Level {level_idx} ({out_ch} filters, avg of {in_ch} in_ch, {k}×{k})')
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, f'level_{level_idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {layer_name} level {level_idx} (showing {max_filters}/{out_ch} filters)")
    
    print(f"Deconv filter visualizations saved to {layer_dir}")


def visualize_taxonomy_tree(layer, layer_name, save_dir, max_depth=4, activations=None):
    """Visualize the taxonomic hierarchy as a tree with alpha parameters and filter/activation images.
    
    Args:
        layer: TaxonConv or TaxonDeconv layer
        layer_name: Name for the layer (e.g., 'taxon_conv1')
        save_dir: Directory to save visualization
        max_depth: Maximum depth to visualize (including root)
        activations: Optional activation tensor (1, C, H, W) to visualize activations instead of filters
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    # Get alpha parameters and apply sigmoid (alphas is a ParameterList)
    # Each element in alphas has shape (2^i, 1) where i is the level
    alpha_values = []
    for i, alpha in enumerate(layer.alphas):
        # Apply sigmoid to get mixing coefficients in [0, 1]
        alpha_sig = torch.sigmoid(alpha / layer.temperature).detach().cpu().numpy()
        alpha_values.append(alpha_sig)
        print(f"  Alpha level {i}: shape {alpha_sig.shape}, mean={alpha_sig.mean():.4f}, "
              f"min={alpha_sig.min():.4f}, max={alpha_sig.max():.4f}")
    
    # Get hierarchy weights (filters at each level) or use activations
    is_deconv = isinstance(layer, TaxonDeconv)
    if activations is not None:
        # Use activations instead of filters
        acts = activations[0].detach().cpu().numpy()  # (C, H, W)
        image_type = "activations"
    else:
        hierarchy_weights = layer.get_hierarchy_weights()
        image_type = "filters"
    
    # Determine actual depth (including root = depth 0)
    n_layers = layer.n_layers
    actual_depth = min(n_layers + 1, max_depth, 4)  # Cap at 4 layers
    
    # Create figure with larger size to accommodate high-resolution images
    max_nodes = 2 ** (actual_depth - 1)
    # Scale up for high-resolution images
    fig_width = max(60, max_nodes * 6.0)  # Horizontal space unchanged
    fig_height = 30  # Condensed height to ~1/3 of previous
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.axis('off')
    
    # Track node positions and connections
    positions = {}
    node_labels = {}
    node_image_indices = {}  # Will store either (level, filter_idx) or (start_ch, end_ch)
    edges = []
    edge_labels = {}
    
    # Level 0: Root
    positions[0] = (0.5, 0.88)  # Moved down from 0.95 to leave space for title
    node_labels[0] = "Root"

    # Channel accounting per node
    per_node_channels = 1 if not is_deconv else layer.out_channels
    if image_type == "activations":
        node_image_indices[0] = (0, per_node_channels)
    else:
        node_image_indices[0] = (0, 0)  # (level, filter_idx)

    node_counter = 1
    cumulative_start = per_node_channels  # next available channel start
    vertical_spacing = 0.55 / actual_depth  # Condensed vertical spacing
    
    # Process each level
    for level in range(1, actual_depth):
        if level > n_layers:
            break
            
        num_nodes = 2 ** level
        y_pos = 0.88 - level * vertical_spacing  # Adjusted to match new root position
        
        # Horizontal spacing: spread nodes evenly across width
        # Add padding on the sides
        padding = 0.10  # Increased padding for better horizontal spacing
        available_width = 1.0 - 2 * padding
        
        for node_idx in range(num_nodes):
            node_id = node_counter + node_idx
            
            # Evenly space nodes horizontally
            if num_nodes == 1:
                x_pos = 0.5
            else:
                x_pos = padding + available_width * node_idx / (num_nodes - 1)
            
            positions[node_id] = (x_pos, y_pos)
            node_labels[node_id] = f"L{level}N{node_idx}"

            # Store image index differently based on type
            if image_type == "activations":
                start_ch = cumulative_start + node_idx * per_node_channels
                end_ch = start_ch + per_node_channels
                node_image_indices[node_id] = (start_ch, end_ch)
            else:
                # For filters: use node_idx for both conv and deconv (indexed per level)
                node_image_indices[node_id] = (level, node_idx)
            
            # Find parent - FIXED CALCULATION
            parent_idx = node_idx // 2
            # Parent is at level-1, and its node_id is the sum of all previous levels + parent_idx
            parent_id = sum(2**i for i in range(level - 1)) + parent_idx if level > 0 else None
            
            if parent_id is not None:
                # Determine which child this is (0 or 1)
                child_position = node_idx % 2
                
                # Get alpha value for this edge
                alpha_idx = n_layers - level
                if 0 <= alpha_idx < len(alpha_values):
                    alpha_tensor = alpha_values[alpha_idx]
                    
                    if parent_idx < alpha_tensor.shape[0]:
                        alpha_val = alpha_tensor[parent_idx, 0]
                        if child_position == 1:
                            alpha_val = 1.0 - alpha_val
                        
                        edges.append((parent_id, node_id))
                        edge_labels[(parent_id, node_id)] = f"α={alpha_val:.3f}"
                    else:
                        edges.append((parent_id, node_id))
                        edge_labels[(parent_id, node_id)] = ""
                else:
                    edges.append((parent_id, node_id))
                    edge_labels[(parent_id, node_id)] = ""
        
        if image_type == "activations":
            cumulative_start += per_node_channels * num_nodes
        
        node_counter += num_nodes
    
    # Pre-render all image thumbnails to avoid expensive inline rendering
    image_thumbnails = {}
    image_sizes = {}
    print(f"  Pre-rendering {len(positions)} {image_type} thumbnails...")
    for node_id, (x, y) in positions.items():
        img_idx = node_image_indices[node_id]
        
        if image_type == "activations":
            # img_idx is a channel slice (start, end)
            start_ch, end_ch = img_idx
            if end_ch <= acts.shape[0]:
                # Aggregate across the slice (mean over channels within the node)
                img_data = acts[start_ch:end_ch, :, :].mean(axis=0)
                
                # Store original size
                image_sizes[node_id] = img_data.shape
                
                # Normalize to [0, 1] - NO RESIZING, keep original resolution
                img_min, img_max = img_data.min(), img_data.max()
                if img_max > img_min:
                    img_norm = (img_data - img_min) / (img_max - img_min)
                else:
                    img_norm = np.zeros_like(img_data)
                
                image_thumbnails[node_id] = img_norm
        else:
            # img_idx is (level, filter_idx) for filters
            level_idx, filter_idx = img_idx
            
            if level_idx < len(hierarchy_weights):
                w_np = hierarchy_weights[level_idx].detach().cpu().numpy()
                
                # Handle Conv vs Deconv shapes
                if len(w_np.shape) == 4 and w_np.shape[0] == layer.in_channels:
                    # Deconv: (in_ch, out_ch * nodes, k, k)
                    nodes = 2 ** level_idx
                    out_ch = layer.out_channels
                    w_np = w_np.transpose(1, 0, 2, 3)  # (out_ch*nodes, in_ch, k, k)
                    w_np = w_np.reshape(nodes, out_ch, layer.in_channels, w_np.shape[2], w_np.shape[3])
                    if filter_idx >= nodes:
                        filter_idx = 0
                    img_data = w_np[filter_idx].mean(axis=0).mean(axis=0)
                elif len(w_np.shape) == 4:
                    # Conv: (num_filters, in_ch, k, k)
                    if filter_idx >= w_np.shape[0]:
                        filter_idx = 0
                    img_data = w_np[filter_idx, :, :, :]
                    img_data = img_data.mean(axis=0)
                else:
                    img_data = w_np[0] if len(w_np.shape) > 0 else w_np
                
                # Store original size
                image_sizes[node_id] = img_data.shape
                
                # Normalize to [0, 1]
                img_min, img_max = img_data.min(), img_data.max()
                if img_max > img_min:
                    img_norm = (img_data - img_min) / (img_max - img_min)
                else:
                    img_norm = np.zeros_like(img_data)
                
                image_thumbnails[node_id] = img_norm
    
    # Determine zoom based on image size to maintain consistent visual size
    # Target visual size in inches (adjust as needed)
    if image_type == "activations":
        target_size_inches = 2.5  # Larger activations
    else:
        target_size_inches = 2.5  # Larger filters
    
    # Get a representative image size
    if image_sizes:
        sample_size = list(image_sizes.values())[0]
        sample_pixels = max(sample_size)
        # Calculate zoom to achieve target size
        # zoom * pixels / dpi = inches, so zoom = inches * dpi / pixels
        # Assume 100 dpi for OffsetImage
        base_zoom = (target_size_inches * 100) / sample_pixels
    else:
        base_zoom = 35.0
    
    # Draw edges first (so they appear behind nodes)
    print(f"  Drawing {len(edges)} edges...")
    for (parent_id, child_id) in edges:
        px, py = positions[parent_id]
        cx, cy = positions[child_id]
        
        # Draw line with minimal thickness
        ax.plot([px, cx], [py, cy], 'k-', linewidth=0.5, zorder=1, alpha=0.3)
        
        # Add alpha label at midpoint
        mid_x, mid_y = (px + cx) / 2, (py + cy) / 2
        label = edge_labels[(parent_id, child_id)]
        
        if label:  # Only draw label if it exists
                bbox_props = dict(boxstyle='round,pad=0.7', facecolor='lightyellow', 
                         edgecolor='gray', alpha=0.9, linewidth=1.3)
                ax.text(mid_x, mid_y, label, ha='center', va='center', 
                    fontsize=16, fontweight='bold', bbox=bbox_props, zorder=2)
    
    # Draw nodes with image visualizations using OffsetImage for better control
    print(f"  Drawing {len(positions)} nodes with {image_type}...")
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    for node_id, (x, y) in positions.items():
        if node_id in image_thumbnails:
            # Use nearest-neighbor interpolation to preserve pixel boundaries
            imagebox = OffsetImage(image_thumbnails[node_id], cmap='viridis', 
                                 zoom=base_zoom, interpolation='nearest')
            imagebox.image.axes = ax
            
            # Create AnnotationBbox to place the image
            ab = AnnotationBbox(imagebox, (x, y),
                               frameon=True,
                               pad=0.0,
                               bboxprops=dict(edgecolor='black', linewidth=1.5, facecolor='none'))
            ax.add_artist(ab)
            
            # Add label below
            label_offset = 0.08  # Reduced to bring labels closer to images
            ax.text(x, y - label_offset, node_labels[node_id], 
                    ha='center', va='top', fontsize=11, fontweight='bold', zorder=5)
        else:
            # Fallback: draw circle
            circle = plt.Circle((x, y), 0.015, color='lightgray', ec='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y - 0.03, node_labels[node_id], 
                    ha='center', va='top', fontsize=10, fontweight='bold', zorder=5)
    
    # Add title with layer info and extra padding
    print(f"  Adding title and saving...")
    title_text = f"{layer_name} Taxonomy Tree\n"
    if actual_depth < n_layers + 1:
        title_text += f"(Showing first {actual_depth} levels, truncated from {n_layers + 1})"
    else:
        title_text += f"(Full hierarchy: {actual_depth} levels)"
    
    # Use text instead of set_title for better control of positioning
    ax.text(0.5, 0.98, title_text, ha='center', va='top', 
            transform=ax.transAxes, fontsize=18, fontweight='bold')
    
    # Add legend
    legend_text = (
        f"Total layers: {n_layers}\n"
        f"Nodes at each level: 1, 2, 4, 8, ...\n"
        f"α = Sigmoid weight for child selection\n"
        f"{'Brighter = Higher activation' if image_type == 'activations' else 'Filter weights shown'}"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save with higher DPI to preserve detail
    os.makedirs(save_dir, exist_ok=True)
    safe_name = layer_name.lower().replace(' ', '_')
    save_path = os.path.join(save_dir, f'{safe_name}_tree.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved taxonomy tree to {save_path} (full resolution preserved)")


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
            result = model.encode(images)
            # Handle tuple return (latents, kl) from KL layers
            if isinstance(result, tuple):
                latents = result[0]
            else:
                latents = result
            all_latents.append(latents.cpu().numpy())
    
    all_latents = np.concatenate(all_latents, axis=0)
    print(f"Collected {all_latents.shape[0]} latent vectors")
    
    # Handle spatial latent representations (flatten if needed)
    if all_latents.ndim > 2:
        print(f"Spatial latent shape detected: {all_latents.shape}")
        batch_size = all_latents.shape[0]
        all_latents = all_latents.reshape(batch_size, -1)
        print(f"Flattened to: {all_latents.shape}")
    
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
            
            # Process each encoder layer
            for i, conv_layer in enumerate(model.encoder.conv_layers):
                x = conv_layer(x)
                
                # Check if this is a TaxonConv layer
                if isinstance(conv_layer, TaxonConv):
                    visualize_taxonomy_tree(conv_layer, f'Encoder Layer {i+1}', img_dir, 
                                          max_depth=4, activations=x)
                else:
                    # Regular conv - use grid visualization
                    visualize_feature_maps(x, os.path.join(img_dir, f'encoder_layer_{i+1}.png'),
                                         f'Encoder Layer {i+1}', max_maps=16)
                
                x = F.relu(x)
                if i < len(model.encoder.strides) and model.encoder.strides[i] > 1:
                    if model.encoder.use_maxpool:
                        x = F.max_pool2d(x, model.encoder.strides[i])
                    else:
                        x = F.avg_pool2d(x, model.encoder.strides[i])
            
            # Latent (may be spatial or flat, so flatten for visualization)
            result = model.encode(img)
            # Handle tuple return (latents, kl) from KL layers
            if isinstance(result, tuple):
                latent = result[0]
            else:
                latent = result
            
            # Flatten if spatial
            if latent.ndim > 2:
                latent_flat = latent.view(latent.size(0), -1)
            else:
                latent_flat = latent
            
            plt.figure(figsize=(12, 3))
            plt.bar(range(latent_flat.shape[1]), latent_flat.cpu().squeeze().numpy())
            plt.xlabel('Latent Dimension')
            plt.ylabel('Activation')
            plt.title(f'Latent Space (dim={latent_flat.shape[1]})')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'latent.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Decoder activations - latent is already spatial (B, C, H, W)
            x = latent
            
            # Process each decoder layer
            for i, deconv_layer in enumerate(model.decoder.deconv_layers):
                x = deconv_layer(x)
                
                # Check if this is a TaxonDeconv or TaxonConv layer
                if isinstance(deconv_layer, (TaxonDeconv, TaxonConv)):
                    visualize_taxonomy_tree(deconv_layer, f'Decoder Layer {i+1}', img_dir,
                                          max_depth=4, activations=x)
                else:
                    # Regular deconv - use grid visualization
                    visualize_feature_maps(x, os.path.join(img_dir, f'decoder_layer_{i+1}.png'),
                                         f'Decoder Layer {i+1}', max_maps=16)
                
                x = F.relu(x)
            
            # Final Conv2D layer to RGB
            x = model.decoder.final_conv(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'decoder_final_conv.png'), 'Final Conv (RGB)', max_maps=3)
            
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


def parse_layer_config(config):
    """Parse layer-by-layer config format and auto-infer parameters.
    
    Supports both legacy format (separate lists) and new layer-by-layer format.
    """
    model_config = config['model']
    
    # Check if using new layer-by-layer format
    if 'encoder_layers' in model_config:
        encoder_layers = model_config['encoder_layers']
        decoder_layers = model_config['decoder_layers']
        
        # Extract parameters from layer configs
        encoder_kernel_sizes = [layer['kernel_size'] for layer in encoder_layers]
        encoder_strides = [layer['stride'] for layer in encoder_layers]
        
        # Check for n_layers (taxonomic) or n_filters (regular)
        encoder_n_layers = [layer.get('n_layers') for layer in encoder_layers] if 'n_layers' in encoder_layers[0] else None
        encoder_n_filters = [layer.get('n_filters') for layer in encoder_layers] if 'n_filters' in encoder_layers[0] else None
        encoder_layer_types = [layer.get('layer_type', 'taxonomic') for layer in encoder_layers]
        
        decoder_kernel_sizes = [layer['kernel_size'] for layer in decoder_layers]
        decoder_strides = [layer['stride'] for layer in decoder_layers]
        decoder_n_layers = [layer.get('n_layers') for layer in decoder_layers] if 'n_layers' in decoder_layers[0] else None
        decoder_n_filters = [layer.get('n_filters') for layer in decoder_layers] if 'n_filters' in decoder_layers[0] else None
        decoder_layer_types = [layer.get('layer_type', 'taxonomic') for layer in decoder_layers]
        
        # Auto-infer decoder paddings if not specified
        decoder_paddings = []
        for layer in decoder_layers:
            if 'padding' in layer:
                decoder_paddings.append(layer['padding'])
            else:
                # Auto-calculate: for kernel k, use k//2
                k = layer['kernel_size']
                decoder_paddings.append(k // 2)
        
        # Auto-infer decoder output_paddings if not specified
        decoder_output_paddings = []
        for layer in decoder_layers:
            if 'output_padding' in layer:
                decoder_output_paddings.append(layer['output_padding'])
            else:
                # Default: 1 for stride>1, 0 for stride=1
                decoder_output_paddings.append(1 if layer['stride'] > 1 else 0)
    else:
        # Legacy format - use separate lists
        encoder_kernel_sizes = model_config.get('encoder_kernel_sizes')
        encoder_strides = model_config.get('encoder_strides')
        encoder_n_layers = model_config.get('encoder_n_layers', None)
        encoder_n_filters = model_config.get('encoder_n_filters', None)
        encoder_layer_types = model_config.get('encoder_layer_types', None)
        
        decoder_kernel_sizes = model_config.get('decoder_kernel_sizes')
        decoder_strides = model_config.get('decoder_strides')
        decoder_n_layers = model_config.get('decoder_n_layers', None)
        decoder_n_filters = model_config.get('decoder_n_filters', None)
        decoder_layer_types = model_config.get('decoder_layer_types', None)
        decoder_paddings = model_config.get('decoder_paddings', None)
        decoder_output_paddings = model_config.get('decoder_output_paddings', None)
    
    return {
        'encoder_kernel_sizes': encoder_kernel_sizes,
        'encoder_strides': encoder_strides,
        'encoder_n_layers': encoder_n_layers,
        'encoder_n_filters': encoder_n_filters,
        'encoder_layer_types': encoder_layer_types,
        'decoder_kernel_sizes': decoder_kernel_sizes,
        'decoder_strides': decoder_strides,
        'decoder_n_layers': decoder_n_layers,
        'decoder_n_filters': decoder_n_filters,
        'decoder_layer_types': decoder_layer_types,
        'decoder_paddings': decoder_paddings,
        'decoder_output_paddings': decoder_output_paddings
    }


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
        use_maxpool = config['model']['use_maxpool']
        random_init_alphas = config['model'].get('random_init_alphas', False)
        alpha_init_distribution = config['model'].get('alpha_init_distribution', 'uniform')
        alpha_init_range = config['model'].get('alpha_init_range', None)
        alpha_init_seed = config['model'].get('alpha_init_seed', None)
        
        # Parse layer configurations (supports both formats)
        layer_params = parse_layer_config(config)
        encoder_kernel_sizes = layer_params['encoder_kernel_sizes']
        encoder_strides = layer_params['encoder_strides']
        encoder_n_layers = layer_params['encoder_n_layers']
        encoder_n_filters = layer_params['encoder_n_filters']
        encoder_layer_types = layer_params['encoder_layer_types']
        decoder_kernel_sizes = layer_params['decoder_kernel_sizes']
        decoder_strides = layer_params['decoder_strides']
        decoder_n_layers = layer_params['decoder_n_layers']
        decoder_n_filters = layer_params['decoder_n_filters']
        decoder_layer_types = layer_params['decoder_layer_types']
        decoder_paddings = layer_params['decoder_paddings']
        decoder_output_paddings = layer_params['decoder_output_paddings']
        
        # Analysis-specific parameters (optional)
        checkpoint_path = config.get('analysis', {}).get('checkpoint_path', None)
        num_latent_batches = config.get('analysis', {}).get('num_latent_batches', 50)
        num_reconstruction_batches = config.get('analysis', {}).get('num_reconstruction_batches', 20)
        num_multiple_recon_images = config.get('analysis', {}).get('num_multiple_recon_images', 8)
        num_reconstructions_per_image = config.get('analysis', {}).get('num_reconstructions_per_image', 8)
        num_activation_images = config.get('analysis', {}).get('num_activation_images', 3)
        
        # Use experiment_name from config, or fall back to analysis_save_dir
        experiment_name = config.get('experiment_name', None)
        if experiment_name:
            save_dir_prefix = f"outputs/cifar10/analysis/{experiment_name}"
        else:
            save_dir_prefix = config.get('output', {}).get('analysis_save_dir', 'outputs/cifar10/analysis')
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
        encoder_n_filters = None
        decoder_n_filters = None
        encoder_layer_types = None
        decoder_layer_types = None
        decoder_paddings = None
        decoder_output_paddings = None
        use_maxpool = True
        random_init_alphas = False
        alpha_init_distribution = 'uniform'
        alpha_init_range = None
        alpha_init_seed = None
        num_latent_batches = 50
        num_reconstruction_batches = 20
        num_multiple_recon_images = 8
        num_reconstructions_per_image = 8
        num_activation_images = 3
        save_dir_prefix = 'outputs/cifar10/analysis'
    
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
    # Use experiment name if provided in config, otherwise use timestamp
    if args.config and 'experiment_name' in config:
        save_dir = save_dir_prefix
    else:
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
    print(f"Random alpha init: {random_init_alphas} (dist={alpha_init_distribution}, range={alpha_init_range}, seed={alpha_init_seed})")
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
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_layer_types=encoder_layer_types,
        decoder_layer_types=decoder_layer_types,
        decoder_paddings=decoder_paddings,
        decoder_output_paddings=decoder_output_paddings,
        use_maxpool=use_maxpool,
        random_init_alphas=random_init_alphas,
        alpha_init_distribution=alpha_init_distribution,
        alpha_init_range=alpha_init_range,
        alpha_init_seed=alpha_init_seed
    )
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    loader = CIFAR10Loader(batch_size=batch_size, root=data_root)
    train_loader, test_loader = loader.get_loaders()
    
    # Analysis 1: Visualize encoder Conv filters
    print("\n" + "=" * 60)
    print("1. Visualizing Encoder TaxonConv Filters")
    print("=" * 60)
    for i in range(len(model.encoder.conv_layers)):
        layer_name = f'encoder_layer_{i+1}'
        print(f"\nProcessing {layer_name}...")
        visualize_taxonconv_filters(model, save_dir, layer_name=layer_name)
    
    # Analysis 2: Visualize taxonomy trees for encoder
    print("\n" + "=" * 60)
    print("2. Visualizing Encoder Taxonomy Trees")
    print("=" * 60)
    encoder_layers = []
    for i, layer in enumerate(model.encoder.conv_layers):
        layer_name = f'encoder_layer_{i+1}'
        encoder_layers.append((layer, layer_name))
    
    for layer, layer_name in encoder_layers:
        print(f"\nProcessing {layer_name}...")
        layer_save_dir = os.path.join(save_dir, f"{layer_name}_filters")
        visualize_taxonomy_tree(layer, layer_name, layer_save_dir, max_depth=4)
    
    # Analysis 3: Visualize decoder Deconv filters
    print("\n" + "=" * 60)
    print("3. Visualizing Decoder TaxonDeconv Filters")
    print("=" * 60)
    for i in range(len(model.decoder.deconv_layers)):
        layer_name = f'decoder_layer_{i+1}'
        print(f"\nProcessing {layer_name}...")
        visualize_taxondeconv_filters(model, save_dir, layer_name=layer_name)
    
    # Analysis 4: Visualize taxonomy trees for decoder
    print("\n" + "=" * 60)
    print("4. Visualizing Decoder Taxonomy Trees")
    print("=" * 60)
    decoder_layers = []
    for i, layer in enumerate(model.decoder.deconv_layers):
        layer_name = f'decoder_layer_{i+1}'
        decoder_layers.append((layer, layer_name))
    
    for layer, layer_name in decoder_layers:
        print(f"\nProcessing {layer_name}...")
        layer_save_dir = os.path.join(save_dir, f"{layer_name}_filters")
        visualize_taxonomy_tree(layer, layer_name, layer_save_dir, max_depth=4)
    
    # Note: final_conv is now a regular Conv2D, not a TaxonConv, so we skip its visualization
    print(f"\nSkipping final_conv - it's a regular Conv2D for RGB projection")
    
    # Analysis 5: Latent space sparsity
    print("\n" + "=" * 60)
    print("5. Analyzing Latent Space")
    print("=" * 60)
    analyze_latent_sparsity(model, test_loader, device, save_dir, num_batches=num_latent_batches)
    
    # Analysis 6: Reconstruction quality
    print("\n" + "=" * 60)
    print("6. Analyzing Reconstruction Quality")
    print("=" * 60)
    analyze_reconstruction_quality(model, test_loader, device, save_dir, num_batches=num_reconstruction_batches)
    
    # Analysis 7: Multiple reconstructions per image
    print("\n" + "=" * 60)
    print("7. Visualizing Multiple Reconstructions")
    print("=" * 60)
    visualize_multiple_reconstructions(model, test_loader, device, save_dir, 
                                      num_images=num_multiple_recon_images, 
                                      num_reconstructions=num_reconstructions_per_image)
    
    # Analysis 8: Layer activations for random images
    print("\n" + "=" * 60)
    print("8. Visualizing Layer Activations")
    print("=" * 60)
    visualize_layer_activations(model, test_loader, device, save_dir, num_images=num_activation_images)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
