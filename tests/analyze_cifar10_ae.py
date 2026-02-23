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
from src.model.taxon_layers import (TaxonConv, TaxonDeconv,
                                     TaxonConvKL, TaxonDeconvKL,
                                     MultiHierarchyTaxonConv, MultiHierarchyTaxonDeconv,
                                     TaxonResnetConv, TaxonResnetDeconv,
                                     MultiHierarchyTaxonResnetConv, MultiHierarchyTaxonResnetDeconv)
from src.utils.dataloader import CIFAR10Loader


# ---------------------------------------------------------------------------
# Multi-hierarchy utilities
# ---------------------------------------------------------------------------

def _is_multi_hierarchy(layer):
    """Return True if layer is a multi-hierarchy wrapper."""
    return isinstance(layer, (MultiHierarchyTaxonConv, MultiHierarchyTaxonDeconv,
                               MultiHierarchyTaxonResnetConv, MultiHierarchyTaxonResnetDeconv))


def _is_any_taxon_layer(layer):
    """Return True for any taxonomic layer type (single or multi)."""
    return isinstance(layer, (TaxonConv, TaxonDeconv,
                               TaxonConvKL, TaxonDeconvKL,
                               MultiHierarchyTaxonConv, MultiHierarchyTaxonDeconv,
                               TaxonResnetConv, TaxonResnetDeconv,
                               MultiHierarchyTaxonResnetConv, MultiHierarchyTaxonResnetDeconv))


def _is_kl_or_resnet_layer(layer):
    """Return True for KL or Resnet taxonomic layers that skip ReLU."""
    return isinstance(layer, (TaxonConvKL, TaxonDeconvKL,
                               TaxonResnetConv, TaxonResnetDeconv,
                               MultiHierarchyTaxonResnetConv, MultiHierarchyTaxonResnetDeconv))


def _get_sub_layers(layer):
    """Return list of (sub_layer, hierarchy_index) pairs.

    For single-hierarchy layers returns ``[(layer, 0)]``.
    For multi-hierarchy wrappers returns
    ``[(layer.hierarchies[h], h) for h in range(n_hierarchies)]``.
    """
    if _is_multi_hierarchy(layer):
        return [(sub, h) for h, sub in enumerate(layer.hierarchies)]
    return [(layer, 0)]


def _channels_per_single_hierarchy(layer):
    """Return the number of output channels produced by one hierarchy tree."""
    if _is_multi_hierarchy(layer):
        return layer.hierarchies[0].num_output_channels()
    return layer.num_output_channels()


# ---------------------------------------------------------------------------


def load_model(checkpoint_path, latent_dim=256, temperature=1.0, device='cuda',
               encoder_kernel_sizes=None, decoder_kernel_sizes=None,
               encoder_strides=None, decoder_strides=None,
               encoder_n_layers=None, decoder_n_layers=None,
               encoder_n_filters=None, decoder_n_filters=None,
               encoder_layer_types=None, decoder_layer_types=None,
               decoder_paddings=None, decoder_output_paddings=None,
               use_maxpool=True, encoder_n_hierarchies=None, decoder_n_hierarchies=None,
               random_init_alphas=False,
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
        encoder_n_hierarchies=encoder_n_hierarchies,
        decoder_n_hierarchies=decoder_n_hierarchies,
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
    """Visualize hierarchical filters from a TaxonConv layer, covering all hierarchies."""

    # Get the layer (may be TaxonConv or MultiHierarchyTaxonConv)
    if layer_name.startswith('encoder_layer_'):
        layer_idx = int(layer_name.split('_')[-1]) - 1
        layer = model.encoder.conv_layers[layer_idx]
    elif layer_name == 'final_conv':
        # final_conv is now a regular Conv2D, skip visualization
        print(f"  Skipping {layer_name} - it's a regular Conv2D, not a TaxonConv")
        return
    else:
        raise ValueError(f"Unknown layer: {layer_name}")

    # Base directory for this layer's filter visualizations
    base_layer_dir = os.path.join(save_dir, f'{layer_name}_filters')

    sub_layers = _get_sub_layers(layer)
    multi = len(sub_layers) > 1
    if multi:
        print(f"  {layer_name}: {len(sub_layers)} independent hierarchies")

    for sub_layer, h_idx in sub_layers:
        # Per-hierarchy subdirectory for multi-hierarchy; flat dir for single
        if multi:
            layer_dir = os.path.join(base_layer_dir, f'hierarchy_{h_idx:02d}')
            h_label = f" (hierarchy {h_idx})"
        else:
            layer_dir = base_layer_dir
            h_label = ""
        os.makedirs(layer_dir, exist_ok=True)

        # Get hierarchy weights from this individual sub-layer
        weights = sub_layer.get_hierarchy_weights()

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

            plt.suptitle(f'{layer_name}{h_label} Level {level_idx} ({n_filters} filters, {in_ch}ch, {k}×{k})')
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f'level_{level_idx}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved {layer_name}{h_label} level {level_idx} ({n_filters} filters)")

        print(f"Conv filter visualizations saved to {layer_dir}")


def visualize_taxondeconv_filters(model, save_dir, layer_name='decoder_layer_1', n_cols=8):
    """Visualize hierarchical filters from a TaxonDeconv layer, covering all hierarchies."""

    # Get the layer (may be TaxonDeconv or MultiHierarchyTaxonDeconv)
    if layer_name.startswith('decoder_layer_'):
        layer_idx = int(layer_name.split('_')[-1]) - 1
        layer = model.decoder.deconv_layers[layer_idx]
    else:
        raise ValueError(f"Unknown layer: {layer_name}")

    # Base directory for this layer's filter visualizations
    base_layer_dir = os.path.join(save_dir, f'{layer_name}_filters')

    sub_layers = _get_sub_layers(layer)
    multi = len(sub_layers) > 1
    if multi:
        print(f"  {layer_name}: {len(sub_layers)} independent hierarchies")

    for sub_layer, h_idx in sub_layers:
        # Per-hierarchy subdirectory for multi-hierarchy; flat dir for single
        if multi:
            layer_dir = os.path.join(base_layer_dir, f'hierarchy_{h_idx:02d}')
            h_label = f" (hierarchy {h_idx})"
        else:
            layer_dir = base_layer_dir
            h_label = ""
        os.makedirs(layer_dir, exist_ok=True)

        # Get hierarchy weights from this individual sub-layer
        weights = sub_layer.get_hierarchy_weights()

        # Visualize each level with cumulative filter indices (reset per hierarchy)
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

            plt.suptitle(f'{layer_name}{h_label} Level {level_idx} ({out_ch} filters, avg of {in_ch} in_ch, {k}×{k})')
            plt.tight_layout()
            plt.savefig(os.path.join(layer_dir, f'level_{level_idx}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved {layer_name}{h_label} level {level_idx} (showing {max_filters}/{out_ch} filters)")

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
    # Determine whether this layer exposes alpha parameters. KL variants provide
    # an empty `alphas` list for compatibility; when absent we skip alpha visuals.
    has_alphas = (hasattr(layer, 'alphas') and len(layer.alphas) > 0)
    alpha_values = []
    if has_alphas:
        for i, alpha in enumerate(layer.alphas):
            # Apply sigmoid to get mixing coefficients in [0, 1]
            alpha_sig = torch.sigmoid(alpha / layer.temperature).detach().cpu().numpy()
            alpha_values.append(alpha_sig)
            print(f"  Alpha level {i}: shape {alpha_sig.shape}, mean={alpha_sig.mean():.4f}, "
                  f"min={alpha_sig.min():.4f}, max={alpha_sig.max():.4f}")
    else:
        print(f"  No alpha parameters present for {layer_name}; skipping alpha visualization.")
    
    # Get hierarchy weights (filters at each level) or use activations
    is_deconv = isinstance(layer, (TaxonDeconv, TaxonResnetDeconv))
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


def visualize_taxonomy_subtree(layer, layer_name, save_dir, start_level, start_node_idx,
                                subtree_depth=4, activations=None):
    """Visualize a sub-hierarchy of the taxonomy tree rooted at a specific node.

    Renders a tree of ``subtree_depth`` levels starting from the node at global
    position ``(start_level, start_node_idx)`` and expanding downward toward the
    leaves.  All filter images and alpha weights are looked up using the correct
    *global* level and node index so the displayed content is exactly the portion
    of the full hierarchy associated with that sub-tree.

    Args:
        layer: TaxonConv or TaxonDeconv layer
        layer_name: Human-readable layer name
        save_dir: Directory in which to save this sub-tree image
        start_level: Global hierarchy level of the sub-tree root (0 = true root)
        start_node_idx: Index of the root node within start_level
        subtree_depth: Number of levels to render, including the root
        activations: Optional activation tensor (1, C, H, W)
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    has_alphas = (hasattr(layer, 'alphas') and len(layer.alphas) > 0)
    alpha_values = []
    if has_alphas:
        for i, alpha in enumerate(layer.alphas):
            alpha_sig = torch.sigmoid(alpha / layer.temperature).detach().cpu().numpy()
            alpha_values.append(alpha_sig)

    is_deconv = isinstance(layer, (TaxonDeconv, TaxonResnetDeconv))
    if activations is not None:
        acts = activations[0].detach().cpu().numpy()  # (C, H, W)
        image_type = "activations"
    else:
        hierarchy_weights = layer.get_hierarchy_weights()
        image_type = "filters"

    n_layers = layer.n_layers
    # Cap depth so we never descend past the actual leaves
    actual_depth = min(subtree_depth, n_layers - start_level + 1)

    max_nodes = 2 ** (actual_depth - 1)
    fig_width = max(30, max_nodes * 6.0)
    fig_height = 30
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.axis('off')

    positions = {}
    node_labels = {}
    node_image_indices = {}
    edges = []
    edge_labels = {}

    # Channels per output node (Conv = 1, Deconv = out_channels)
    per_node_channels = 1 if not is_deconv else layer.out_channels

    # ---- Sub-tree root (sub-level 0) ----
    positions[0] = (0.5, 0.88)
    node_labels[0] = f"L{start_level}N{start_node_idx}"
    if image_type == "activations":
        # BFS channel offset: (2^GL - 1 + GN) * per_node_channels
        root_ch = ((2 ** start_level - 1) + start_node_idx) * per_node_channels
        node_image_indices[0] = (root_ch, root_ch + per_node_channels)
    else:
        node_image_indices[0] = (start_level, start_node_idx)

    node_counter = 1
    vertical_spacing = 0.55 / actual_depth

    for sub_level in range(1, actual_depth):
        global_level = start_level + sub_level
        if global_level > n_layers:
            break

        num_nodes = 2 ** sub_level
        y_pos = 0.88 - sub_level * vertical_spacing
        padding = 0.10
        available_width = 1.0 - 2 * padding

        for sub_node_idx in range(num_nodes):
            node_id = node_counter + sub_node_idx
            x_pos = (0.5 if num_nodes == 1
                     else padding + available_width * sub_node_idx / (num_nodes - 1))
            positions[node_id] = (x_pos, y_pos)

            # Global node index within global_level
            global_node_idx = start_node_idx * (2 ** sub_level) + sub_node_idx
            node_labels[node_id] = f"L{global_level}N{global_node_idx}"

            if image_type == "activations":
                chan_start = ((2 ** global_level - 1) + global_node_idx) * per_node_channels
                node_image_indices[node_id] = (chan_start, chan_start + per_node_channels)
            else:
                node_image_indices[node_id] = (global_level, global_node_idx)

            # Parent in sub-tree coordinate space
            sub_parent_idx = sub_node_idx // 2
            parent_id = sum(2 ** i for i in range(sub_level - 1)) + sub_parent_idx

            child_position = sub_node_idx % 2
            # Alpha lookup: same formula as the full-tree renderer, using global level
            alpha_idx = n_layers - global_level
            global_parent_node_idx = start_node_idx * (2 ** (sub_level - 1)) + sub_parent_idx

            if has_alphas and 0 <= alpha_idx < len(alpha_values):
                alpha_tensor = alpha_values[alpha_idx]
                if global_parent_node_idx < alpha_tensor.shape[0]:
                    alpha_val = float(alpha_tensor[global_parent_node_idx, 0])
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

        node_counter += num_nodes

    # ---- Pre-render thumbnails ----
    image_thumbnails = {}
    image_sizes = {}
    for node_id in positions:
        img_idx = node_image_indices[node_id]
        if image_type == "activations":
            start_ch, end_ch = img_idx
            if end_ch <= acts.shape[0]:
                img_data = acts[start_ch:end_ch].mean(axis=0)
                image_sizes[node_id] = img_data.shape
                img_min, img_max = img_data.min(), img_data.max()
                img_norm = ((img_data - img_min) / (img_max - img_min + 1e-8)
                            if img_max > img_min else np.zeros_like(img_data))
                image_thumbnails[node_id] = img_norm
        else:
            global_level_idx, global_filter_idx = img_idx
            if global_level_idx < len(hierarchy_weights):
                w_np = hierarchy_weights[global_level_idx].detach().cpu().numpy()
                if len(w_np.shape) == 4 and w_np.shape[0] == layer.in_channels:
                    # Deconv: (in_ch, out_ch * nodes_at_level, k, k)
                    nodes_at_level = 2 ** global_level_idx
                    out_ch = layer.out_channels
                    w_t = w_np.transpose(1, 0, 2, 3)
                    w_t = w_t.reshape(nodes_at_level, out_ch, layer.in_channels,
                                      w_t.shape[2], w_t.shape[3])
                    fi = global_filter_idx if global_filter_idx < nodes_at_level else 0
                    img_data = w_t[fi].mean(axis=0).mean(axis=0)
                elif len(w_np.shape) == 4:
                    # Conv: (num_filters_at_level, in_ch, k, k)
                    fi = global_filter_idx if global_filter_idx < w_np.shape[0] else 0
                    img_data = w_np[fi].mean(axis=0)
                else:
                    img_data = w_np[0] if w_np.ndim > 0 else w_np
                image_sizes[node_id] = img_data.shape
                img_min, img_max = img_data.min(), img_data.max()
                img_norm = ((img_data - img_min) / (img_max - img_min + 1e-8)
                            if img_max > img_min else np.zeros_like(img_data))
                image_thumbnails[node_id] = img_norm

    target_size_inches = 2.5
    if image_sizes:
        sample_pixels = max(list(image_sizes.values())[0])
        base_zoom = (target_size_inches * 100) / sample_pixels
    else:
        base_zoom = 35.0

    # ---- Draw edges ----
    for (parent_id, child_id) in edges:
        px, py = positions[parent_id]
        cx, cy = positions[child_id]
        ax.plot([px, cx], [py, cy], 'k-', linewidth=0.5, zorder=1, alpha=0.3)
        mid_x, mid_y = (px + cx) / 2, (py + cy) / 2
        label = edge_labels[(parent_id, child_id)]
        if label:
            bbox_props = dict(boxstyle='round,pad=0.7', facecolor='lightyellow',
                              edgecolor='gray', alpha=0.9, linewidth=1.3)
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                    fontsize=16, fontweight='bold', bbox=bbox_props, zorder=2)

    # ---- Draw nodes ----
    for node_id, (x, y) in positions.items():
        if node_id in image_thumbnails:
            imagebox = OffsetImage(image_thumbnails[node_id], cmap='viridis',
                                   zoom=base_zoom, interpolation='nearest')
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (x, y), frameon=True, pad=0.0,
                                bboxprops=dict(edgecolor='black', linewidth=1.5,
                                               facecolor='none'))
            ax.add_artist(ab)
            ax.text(x, y - 0.08, node_labels[node_id],
                    ha='center', va='top', fontsize=11, fontweight='bold', zorder=5)
        else:
            circle = plt.Circle((x, y), 0.015, color='lightgray', ec='black',
                                 linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y - 0.03, node_labels[node_id],
                    ha='center', va='top', fontsize=10, fontweight='bold', zorder=5)

    title_text = (f"{layer_name} Sub-Hierarchy: Root at L{start_level}N{start_node_idx}\n"
                  f"Global levels {start_level}–{min(start_level + actual_depth - 1, n_layers)} "
                  f"of {n_layers}")
    ax.text(0.5, 0.98, title_text, ha='center', va='top',
            transform=ax.transAxes, fontsize=18, fontweight='bold')

    legend_lines = [
        f"Sub-tree root: L{start_level}N{start_node_idx}",
        f"Showing {actual_depth} levels ({image_type})",
        f"Total layers in model: {n_layers}",
    ]
    if has_alphas:
        legend_lines.append("α = Sigmoid weight for child selection")
    legend_lines.append('Brighter = Higher activation'
                        if image_type == 'activations' else 'Filter weights shown')
    ax.text(0.02, 0.98, "\n".join(legend_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'subtree_L{start_level}N{start_node_idx:04d}_tree.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved sub-tree L{start_level}N{start_node_idx} → {save_path}")


def visualize_all_subtree_hierarchies(layer, layer_name, base_save_dir,
                                       subtree_depth=4, activations=None):
    """Visualize all bottom-up sub-hierarchies for a taxonomic layer.

    Identifies the level that is (subtree_depth - 1) levels above the leaves
    (the "4th from the bottom" when subtree_depth=4) and renders a separate
    subtree_depth-level tree for every node at that level, expanding downward
    to the leaves.  All sub-tree images are saved directly under
    ``base_save_dir/sub_hierarchies/``.

    For example, with n_layers=7 and subtree_depth=4 the function picks
    start_level=4 and produces 2^4=16 sub-tree visualisations covering
    global levels 4–7.
    """
    n_layers = layer.n_layers
    start_level = max(0, n_layers - (subtree_depth - 1))

    if start_level == 0:
        print(f"  {layer_name}: hierarchy has only {n_layers} levels; bottom-up "
              f"sub-hierarchies (subtree_depth={subtree_depth}) would cover the "
              f"entire tree — skipping to avoid duplication.")
        return

    num_subtrees = 2 ** start_level
    sub_dir = os.path.join(base_save_dir, 'sub_hierarchies')
    os.makedirs(sub_dir, exist_ok=True)
    print(f"  Generating {num_subtrees} bottom-up sub-hierarchies for {layer_name} "
          f"(root level={start_level}, n_layers={n_layers}, subtree_depth={subtree_depth})...")

    for node_idx in range(num_subtrees):
        visualize_taxonomy_subtree(
            layer, layer_name, sub_dir,
            start_level=start_level,
            start_node_idx=node_idx,
            subtree_depth=subtree_depth,
            activations=activations,
        )

    print(f"  All {num_subtrees} sub-hierarchies saved under {sub_dir}")


def analyze_partonomy_sparsity(model, data_loader, device, save_dir,
                               num_batches=30, sparsity_threshold=0.1,
                               ablation_images=16, n_clusters=8):
    """Five-part partonomy sparsity analysis suite.

    1. Activation overlap (Jaccard) – tests feature specialisation.
    2. Feature selectivity & entropy – quantifies polysemanticity.
    3. Causal unit ablations – tests additive compositionality.
    4. Cross-layer sparsity dependency – tests hierarchical sparsity.
    5. Dimension clustering – evaluates stable sub-structure alignment.
    """
    prt_dir = os.path.join(save_dir, 'partonomy_sparsity')
    os.makedirs(prt_dir, exist_ok=True)

    model.eval()

    # ── Collect latents + per-layer encoder activations ──────────────────────
    all_latents = []
    hook_data = {}      # layer_idx -> list of (B, C, H, W) tensors
    ablation_imgs = None

    hooks = []
    for i, layer in enumerate(model.encoder.conv_layers):
        def _make_hook(idx):
            def _hook(module, inp, out):
                # KL layers return (tensor, kl_loss) tuples — unpack
                t = out[0] if isinstance(out, tuple) else out
                hook_data.setdefault(idx, []).append(t.detach().cpu())
            return _hook
        hooks.append(layer.register_forward_hook(_make_hook(i)))

    print(f"Collecting activations over {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc='Partonomy data')):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            result = model.encode(images)
            z = result[0] if isinstance(result, tuple) else result
            z_np = z.detach().cpu().numpy()
            if z_np.ndim > 2:
                z_np = z_np.reshape(z_np.shape[0], -1)
            all_latents.append(z_np)
            if ablation_imgs is None:
                ablation_imgs = images[:ablation_images]

    for h in hooks:
        h.remove()

    all_latents = np.concatenate(all_latents, axis=0)   # (N, D)
    N, D = all_latents.shape

    # Spatially pool hook data
    layer_acts = {}   # idx -> (N, C)
    for idx, act_list in hook_data.items():
        pooled = [a.mean(dim=(2, 3)).numpy() for a in act_list]   # each (B, C)
        layer_acts[idx] = np.concatenate(pooled, axis=0)          # (N, C)

    print(f"  {N} samples, latent dim={D}")
    for idx, a in sorted(layer_acts.items()):
        print(f"  Encoder layer {idx+1}: {a.shape[1]} channels")

    binary = (np.abs(all_latents) > sparsity_threshold).astype(np.float32)   # (N, D)

    # ── 1. Jaccard activation overlap ─────────────────────────────────────────
    print("  [1/5] Jaccard activation overlap...")
    rng = np.random.default_rng(0)
    n_pairs = min(3000, N * (N - 1) // 2)
    idx_a = rng.integers(0, N, n_pairs)
    idx_b = rng.integers(0, N, n_pairs)
    idx_b[idx_a == idx_b] = (idx_b[idx_a == idx_b] + 1) % N

    inter = (binary[idx_a] * binary[idx_b]).sum(axis=1)
    union = ((binary[idx_a] + binary[idx_b]) > 0).sum(axis=1).astype(np.float32)
    jaccard = np.where(union > 0, inter / union, 0.0)

    lifetime_sparsity = binary.mean(axis=0)   # (D,) – fraction of samples activating each dim

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(jaccard, bins=50, edgecolor='black', color='steelblue')
    axes[0].axvline(jaccard.mean(), color='red', linestyle='--', label=f'Mean={jaccard.mean():.3f}')
    axes[0].set_xlabel('Jaccard Similarity'); axes[0].set_ylabel('Count')
    axes[0].set_title('Pairwise Activation Overlap\n(lower = more specialisation)')
    axes[0].legend()

    axes[1].bar(range(min(D, 200)), sorted(lifetime_sparsity, reverse=True)[:200], color='darkorange')
    axes[1].set_xlabel('Rank-ordered Feature'); axes[1].set_ylabel('Fraction Active')
    axes[1].set_title('Lifetime Sparsity per Dim (sorted)')

    dead = float((lifetime_sparsity < 0.01).mean())
    selective = float(((lifetime_sparsity >= 0.01) & (lifetime_sparsity < 0.2)).mean())
    dense = float((lifetime_sparsity >= 0.5).mean())
    moderate = 1.0 - dead - selective - dense
    axes[2].bar(['Dead\n(<1%)', 'Selective\n(1-20%)', 'Moderate\n(20-50%)', 'Dense\n(>50%)'],
                [v * 100 for v in [dead, selective, moderate, dense]],
                color=['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4'])
    axes[2].set_ylabel('% of Features'); axes[2].set_title('Feature Activity Categories')
    plt.suptitle('1. Activation Overlap & Feature Specialisation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, '01_jaccard_overlap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    np.savez(os.path.join(prt_dir, '01_jaccard_stats.npz'),
             jaccard=jaccard, lifetime_sparsity=lifetime_sparsity,
             mean_jaccard=jaccard.mean(), median_jaccard=float(np.median(jaccard)),
             dead_frac=dead, selective_frac=selective, dense_frac=dense)
    print(f"    Mean Jaccard={jaccard.mean():.4f}  dead={dead*100:.1f}%  "
          f"selective={selective*100:.1f}%  dense={dense*100:.1f}%")

    # ── 2. Feature selectivity and entropy ────────────────────────────────────
    print("  [2/5] Feature selectivity and entropy...")
    p = lifetime_sparsity.clip(1e-6, 1 - 1e-6)
    per_dim_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))   # bits

    try:
        from scipy.stats import kurtosis as _sp_kurt
        per_dim_kurtosis = np.array([_sp_kurt(np.abs(all_latents[:, d]), fisher=True)
                                      for d in range(D)])
    except Exception:
        per_dim_kurtosis = np.zeros(D)

    per_dim_max  = np.abs(all_latents).max(axis=0)
    per_dim_mean = np.abs(all_latents).mean(axis=0)
    selectivity_idx = (per_dim_max - per_dim_mean) / (per_dim_max + per_dim_mean + 1e-8)
    polysemantic_frac = float((np.abs(p - 0.5) < 0.15).mean())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(per_dim_entropy, bins=50, edgecolor='black', color='purple')
    axes[0, 0].axvline(per_dim_entropy.mean(), color='red', linestyle='--',
                       label=f'Mean={per_dim_entropy.mean():.3f} bits')
    axes[0, 0].set_xlabel('Binary Entropy (bits)'); axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Per-Dim Activation Entropy\n(0 bits = maximally selective)')
    axes[0, 0].legend()

    axes[0, 1].hist(per_dim_kurtosis.clip(-10, 50), bins=50, edgecolor='black', color='teal')
    axes[0, 1].axvline(per_dim_kurtosis.mean(), color='red', linestyle='--',
                       label=f'Mean={per_dim_kurtosis.mean():.2f}')
    axes[0, 1].set_xlabel('Activation Kurtosis (Fisher)'); axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Per-Dim Kurtosis\n(high = impulse-like / sparse)')
    axes[0, 1].legend()

    axes[1, 0].hist(selectivity_idx, bins=50, edgecolor='black', color='darkgreen')
    axes[1, 0].axvline(selectivity_idx.mean(), color='red', linestyle='--',
                       label=f'Mean={selectivity_idx.mean():.3f}')
    axes[1, 0].set_xlabel('Selectivity Index (max-mean)/(max+mean)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Feature Selectivity\n(1.0 = fires for one sample only)')
    axes[1, 0].legend()

    axes[1, 1].scatter(lifetime_sparsity, per_dim_entropy, alpha=0.3, s=8, c='navy')
    axes[1, 1].set_xlabel('Lifetime Sparsity (frac. active)')
    axes[1, 1].set_ylabel('Binary Entropy (bits)')
    axes[1, 1].set_title(f'Sparsity vs Entropy\nPolysemantic (p≈0.5): {polysemantic_frac*100:.1f}%')

    plt.suptitle('2. Feature Selectivity & Entropy', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, '02_feature_selectivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    np.savez(os.path.join(prt_dir, '02_selectivity_stats.npz'),
             per_dim_entropy=per_dim_entropy, per_dim_kurtosis=per_dim_kurtosis,
             selectivity_idx=selectivity_idx, polysemantic_frac=polysemantic_frac,
             mean_entropy=float(per_dim_entropy.mean()),
             mean_kurtosis=float(per_dim_kurtosis.mean()))
    print(f"    Mean entropy={per_dim_entropy.mean():.4f} bits  "
          f"mean kurtosis={per_dim_kurtosis.mean():.3f}  "
          f"polysemantic={polysemantic_frac*100:.1f}%")

    # ── 3. Causal unit ablations ──────────────────────────────────────────────
    print("  [3/5] Causal ablation analysis...")
    ablation_imgs = ablation_imgs.to(device)
    n_abl = ablation_imgs.shape[0]

    with torch.no_grad():
        result_abl = model.encode(ablation_imgs)
        z_orig = result_abl[0] if isinstance(result_abl, tuple) else result_abl
        spatial = (z_orig.ndim == 4)          # True if (B, C, H, W)
        n_units = z_orig.shape[1]             # channels to ablate

        # Baseline reconstruction
        base_recon = model(ablation_imgs)
        if isinstance(base_recon, (tuple, list)):
            base_recon = base_recon[0]
        if base_recon.shape[2:] != ablation_imgs.shape[2:]:
            base_recon = F.interpolate(base_recon, size=ablation_imgs.shape[2:],
                                       mode='bilinear', align_corners=False)
        base_mse = ((ablation_imgs - base_recon) ** 2).mean(dim=(1, 2, 3))  # (n_abl,)

        # Active units across ablation images
        if spatial:
            unit_active = (z_orig.abs().mean(dim=(2, 3)) > sparsity_threshold).cpu().numpy()
        else:
            unit_active = (z_orig.abs() > sparsity_threshold).cpu().numpy()   # (n_abl, n_units)
        candidate_units = np.where(unit_active.any(axis=0))[0]

        # Cap cost: prioritise highest-mean-activation channels
        if len(candidate_units) > 256:
            mean_act_per_unit = (z_orig.abs().mean(dim=(2, 3)) if spatial
                                 else z_orig.abs()).mean(dim=0).cpu().numpy()
            candidate_units = candidate_units[
                np.argsort(-mean_act_per_unit[candidate_units])[:256]]

        print(f"    Ablating {len(candidate_units)} active units of {n_units} total...")
        unit_importance = np.zeros(n_units)

        for d in candidate_units:
            z_abl = z_orig.clone()
            if spatial:
                z_abl[:, d, :, :] = 0.0
            else:
                z_abl[:, d] = 0.0

            if hasattr(model, 'decode'):
                recon_abl = model.decode(z_abl)
            else:
                recon_abl = model.decoder(z_abl)
            if isinstance(recon_abl, (tuple, list)):
                recon_abl = recon_abl[0]
            if recon_abl.shape[2:] != ablation_imgs.shape[2:]:
                recon_abl = F.interpolate(recon_abl, size=ablation_imgs.shape[2:],
                                          mode='bilinear', align_corners=False)
            delta = ((ablation_imgs - recon_abl) ** 2).mean(dim=(1, 2, 3))
            unit_importance[d] = (delta - base_mse).clamp(min=0).mean().item()

    sorted_imp = np.sort(unit_importance[unit_importance > 0])[::-1]
    total_imp = sorted_imp.sum()
    cumulative = np.cumsum(sorted_imp) / (total_imp + 1e-12)
    p50 = int(np.searchsorted(cumulative, 0.5)) + 1 if total_imp > 0 else 0
    p90 = int(np.searchsorted(cumulative, 0.9)) + 1 if total_imp > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    show_n = min(50, len(sorted_imp))
    axes[0].bar(range(show_n), sorted_imp[:show_n], color='firebrick')
    axes[0].set_xlabel('Unit (rank)'); axes[0].set_ylabel('Mean MSE Increase')
    axes[0].set_title(f'Top-{show_n} Units by Ablation Importance')

    axes[1].hist(sorted_imp, bins=40, edgecolor='black', color='salmon')
    if len(sorted_imp):
        axes[1].axvline(sorted_imp.mean(), color='blue', linestyle='--',
                        label=f'Mean={sorted_imp.mean():.5f}')
    axes[1].set_xlabel('MSE Increase when Zeroed'); axes[1].set_ylabel('Count')
    axes[1].set_title('Ablation Importance Distribution'); axes[1].legend()

    if len(sorted_imp):
        axes[2].plot(range(1, len(sorted_imp) + 1), cumulative * 100, color='darkblue')
        axes[2].axvline(p50, color='orange', linestyle='--', label=f'50% in top-{p50}')
        axes[2].axvline(p90, color='red',    linestyle='--', label=f'90% in top-{p90}')
    axes[2].set_xlabel('Number of Units'); axes[2].set_ylabel('Cumulative Importance (%)')
    axes[2].set_title('Cumulative Ablation Importance\n(concentrated vs. distributed)')
    axes[2].legend()

    plt.suptitle('3. Causal Ablation: Unit Indispensability', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, '03_causal_ablations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    np.savez(os.path.join(prt_dir, '03_ablation_importance.npz'),
             unit_importance=unit_importance,
             top_units=np.argsort(-unit_importance)[:50],
             p50_units=p50, p90_units=p90)
    print(f"    50% importance in top-{p50} units;  90% in top-{p90} units")

    # ── 4. Cross-layer sparsity dependency ────────────────────────────────────
    print("  [4/5] Cross-layer sparsity dependency...")
    enc_indices = sorted(layer_acts.keys())
    n_enc = len(enc_indices)

    if n_enc >= 2:
        max_ch = 64    # cap channels per layer for heatmap readability
        # Relative threshold per layer
        layer_bin = {}
        for idx in enc_indices:
            a = layer_acts[idx]
            thr = sparsity_threshold * np.abs(a).mean()
            layer_bin[idx] = (np.abs(a) > thr).astype(np.float32)

        n_pairs = n_enc - 1
        fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 7), squeeze=False)
        dep_sparsities = []

        for pi, (l1, l2) in enumerate(zip(enc_indices[:-1], enc_indices[1:])):
            B1 = layer_bin[l1][:, :min(layer_bin[l1].shape[1], max_ch)]    # (N, C1)
            B2 = layer_bin[l2][:, :min(layer_bin[l2].shape[1], max_ch)]    # (N, C2)
            sup1 = B1.sum(axis=0) + 1e-8  # (C1,)
            # P(c2 active | c1 active)
            M = (B1.T @ B2) / sup1[:, None]   # (C1, C2)
            dep_sp = float((M < 0.1).mean())
            dep_sparsities.append(dep_sp)

            ax = axes[0, pi]
            im = ax.imshow(M, aspect='auto', cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(f'Layer {l2+1} channels')
            ax.set_ylabel(f'Layer {l1+1} channels')
            ax.set_title(f'Layer {l1+1}→{l2+1} Cond. Co-activation\n'
                         f'Dep. sparsity: {dep_sp*100:.1f}% near-zero')

        plt.suptitle('4. Cross-Layer Sparsity Dependency', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, '04_cross_layer_dependency.png'), dpi=150, bbox_inches='tight')
        plt.close()
        np.savez(os.path.join(prt_dir, '04_dependency_stats.npz'),
                 dep_sparsity_per_pair=np.array(dep_sparsities),
                 mean_dep_sparsity=float(np.mean(dep_sparsities)))
        print(f"    Mean cross-layer dep. sparsity: {np.mean(dep_sparsities)*100:.1f}%")
    else:
        print("    Skipping cross-layer analysis (<2 encoder layers captured).")

    # ── 5. Cluster active dimensions ──────────────────────────────────────────
    print("  [5/5] Clustering latent dimensions...")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA as _PCA
        _sklearn_ok = True
    except ImportError:
        _sklearn_ok = False
        print("    sklearn not available; skipping clustering.")

    if _sklearn_ok and N >= n_clusters * 5:
        n_pca = min(50, D, N - 1)
        X = all_latents
        if D > n_pca:
            X = _PCA(n_components=n_pca).fit_transform(X)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0

        cluster_mean = np.zeros((n_clusters, D))
        for c in range(n_clusters):
            m = labels == c
            if m.sum() > 0:
                cluster_mean[c] = np.abs(all_latents[m]).mean(axis=0)

        exclusivity = cluster_mean.max(axis=0) / (cluster_mean.sum(axis=0) + 1e-8)  # (D,)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        n_show = min(D, 64)
        top_dims = np.argsort(-cluster_mean.mean(axis=0))[:n_show]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].bar(range(n_clusters), cluster_sizes, color='steelblue')
        axes[0].set_xlabel('Cluster'); axes[0].set_ylabel('Samples')
        axes[0].set_title(f'Cluster Sizes  (Silhouette={sil:.3f})')

        im = axes[1].imshow(cluster_mean[:, top_dims], aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_xlabel(f'Top-{n_show} Features'); axes[1].set_ylabel('Cluster')
        axes[1].set_title('Cluster × Feature Activation')

        axes[2].hist(exclusivity, bins=40, edgecolor='black', color='darkorange')
        axes[2].axvline(exclusivity.mean(), color='red', linestyle='--',
                        label=f'Mean={exclusivity.mean():.3f}')
        axes[2].set_xlabel('Exclusivity (max_c / sum_c)'); axes[2].set_ylabel('Count')
        axes[2].set_title('Per-Dim Cluster Exclusivity\n(1.0 = used by one cluster only)')
        axes[2].legend()

        plt.suptitle('5. Latent Dimension Clustering & Alignment', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, '05_dimension_clustering.png'), dpi=150, bbox_inches='tight')
        plt.close()
        np.savez(os.path.join(prt_dir, '05_cluster_stats.npz'),
                 labels=labels, cluster_sizes=cluster_sizes, silhouette=sil,
                 exclusivity=exclusivity, mean_exclusivity=float(exclusivity.mean()),
                 cluster_mean_acts=cluster_mean)
        print(f"    Silhouette={sil:.4f}  mean exclusivity={exclusivity.mean():.4f}")

    print(f"Partonomy sparsity analysis saved to {prt_dir}")


def analyze_weight_sparsity(model, save_dir):
    """Analyse the sparsity and structure of learned filter weights.

    Per taxonomic layer:
      a. Leaf filter diversity  – pairwise cosine similarity between leaf filters.
      b. Spatial concentration  – Gini coefficient of absolute weight magnitudes.
      c. Parent-child similarity – cosine similarity between a parent and each child
                                   filter, measures how much the tree structure
                                   forces filters to inherit from parents.
    """
    wt_dir = os.path.join(save_dir, 'weight_sparsity')
    os.makedirs(wt_dir, exist_ok=True)

    all_layer_data = []
    for i, layer in enumerate(model.encoder.conv_layers):
        if _is_any_taxon_layer(layer):
            all_layer_data.append((f'enc{i+1}', layer))
    for i, layer in enumerate(model.decoder.deconv_layers):
        if _is_any_taxon_layer(layer):
            all_layer_data.append((f'dec{i+1}', layer))

    if not all_layer_data:
        print("No taxonomic layers found; skipping weight sparsity analysis.")
        return

    def _gini(arr):
        arr = np.abs(arr).flatten()
        arr = np.sort(arr)
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * (index * arr).sum() - (n + 1) * arr.sum()) / (n * arr.sum())

    results = {}
    for layer_key, layer in all_layer_data:
        for sub_layer, h_idx in _get_sub_layers(layer):
            key = f"{layer_key}_h{h_idx:02d}"
            is_classic_deconv = isinstance(sub_layer, TaxonDeconv)
            is_resnet = isinstance(sub_layer, (TaxonResnetConv, TaxonResnetDeconv))
            is_resnet_deconv = isinstance(sub_layer, TaxonResnetDeconv)
            weights = sub_layer.get_hierarchy_weights()   # list, root → leaves
            n_levels = len(weights)

            # ── a. Leaf filter diversity ──────────────────────────────────────
            leaf_w = weights[-1].detach().cpu().numpy()   # leaves
            if is_resnet_deconv:
                # Resnet deconv: weight shape (in_ch, out_ch*2^i, k, k)
                # Transpose to (out_ch*2^i, in_ch, k, k) and treat each as a node
                leaf_w = leaf_w.transpose(1, 0, 2, 3)
                n_leaves = leaf_w.shape[0]
            elif is_classic_deconv:
                # (in_ch, N_leaves*out_ch, k, k) → (N_leaves, in_ch, k, k)
                n_leaves = 2 ** sub_layer.n_layers
                oc = sub_layer.out_channels
                leaf_w = leaf_w.transpose(1, 0, 2, 3)     # (N_leaves*oc, in_ch, k, k)
                leaf_w = leaf_w.reshape(n_leaves, oc,
                                        sub_layer.in_channels,
                                        leaf_w.shape[-2], leaf_w.shape[-1])
                leaf_w = leaf_w.mean(axis=1)               # (N_leaves, in_ch, k, k)
            else:
                n_leaves = leaf_w.shape[0]                 # (N_leaves, in_ch, k, k)

            leaf_flat = leaf_w.reshape(n_leaves, -1)       # (N_leaves, in_ch*k*k)
            norms = np.linalg.norm(leaf_flat, axis=1, keepdims=True) + 1e-8
            cos_mat = (leaf_flat / norms) @ (leaf_flat / norms).T  # (N, N)
            off_diag = cos_mat[np.triu_indices(n_leaves, k=1)]

            # ── b. Spatial concentration (Gini) per leaf filter ───────────────
            gini_vals = np.array([_gini(leaf_flat[i]) for i in range(n_leaves)])

            # ── c. Parent-child cosine similarity ────────────────────────────
            pc_sims = []
            for lv in range(1, n_levels):
                pw = weights[lv - 1].detach().cpu().numpy()
                cw = weights[lv].detach().cpu().numpy()
                if is_resnet_deconv:
                    # Resnet deconv: transpose to get nodes in first dim
                    pf = pw.transpose(1, 0, 2, 3)
                    cf = cw.transpose(1, 0, 2, 3)
                    n_p = pf.shape[0]
                    n_c = cf.shape[0]
                    pf = pf.reshape(n_p, -1)
                    cf = cf.reshape(n_c, -1)
                elif is_classic_deconv:
                    n_p = 2 ** (lv - 1)
                    n_c = 2 ** lv
                    oc = sub_layer.out_channels
                    ic = sub_layer.in_channels
                    def _reshape(w, n_nodes):
                        w = w.transpose(1, 0, 2, 3)
                        w = w.reshape(n_nodes, oc, ic, w.shape[-2], w.shape[-1])
                        return w.mean(axis=1).reshape(n_nodes, -1)
                    pf = _reshape(pw, n_p)
                    cf = _reshape(cw, n_c)
                else:
                    n_p = pw.shape[0]
                    n_c = cw.shape[0]
                    pf = pw.reshape(n_p, -1)
                    cf = cw.reshape(n_c, -1)
                p_norm = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
                c_norm = cf / (np.linalg.norm(cf, axis=1, keepdims=True) + 1e-8)
                for pi in range(n_p):
                    c0, c1 = pi * 2, pi * 2 + 1
                    if c1 < n_c:
                        pc_sims.append(float(p_norm[pi] @ c_norm[c0]))
                        pc_sims.append(float(p_norm[pi] @ c_norm[c1]))
            pc_sims = np.array(pc_sims)

            results[key] = dict(
                n_leaves=n_leaves, n_levels=n_levels,
                leaf_cosine_offdiag=off_diag, mean_leaf_cosine=float(off_diag.mean()),
                gini_per_filter=gini_vals,   mean_gini=float(gini_vals.mean()),
                parent_child_sims=pc_sims,
                mean_pc_sim=float(pc_sims.mean()) if len(pc_sims) else float('nan'),
            )
            print(f"    {key}: n_leaves={n_leaves}  leaf_cosine={off_diag.mean():.4f}  "
                  f"gini={gini_vals.mean():.4f}  pc_sim={results[key]['mean_pc_sim']:.4f}")

    keys = list(results.keys())
    n_k = len(keys)
    if n_k == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    cos_means = [results[k]['mean_leaf_cosine'] for k in keys]
    axes[0, 0].bar(range(n_k), cos_means, color='steelblue')
    axes[0, 0].set_xticks(range(n_k)); axes[0, 0].set_xticklabels(keys, rotation=40, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('Mean Cosine'); axes[0, 0].axhline(0, color='k', lw=0.5)
    axes[0, 0].set_title('Leaf Filter Diversity\n(lower = more independent)')

    gini_means = [results[k]['mean_gini'] for k in keys]
    axes[0, 1].bar(range(n_k), gini_means, color='darkorange')
    axes[0, 1].set_xticks(range(n_k)); axes[0, 1].set_xticklabels(keys, rotation=40, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Mean Gini'); axes[0, 1].set_title('Spatial Concentration\n(higher = more localised)')

    pc_means = [results[k]['mean_pc_sim'] for k in keys]
    axes[0, 2].bar(range(n_k), [v if not np.isnan(v) else 0 for v in pc_means], color='mediumpurple')
    axes[0, 2].set_xticks(range(n_k)); axes[0, 2].set_xticklabels(keys, rotation=40, ha='right', fontsize=8)
    axes[0, 2].set_ylabel('Mean Cosine'); axes[0, 2].set_title('Parent-Child Filter Similarity\n(lower = more differentiation)')

    # Per-value distributions for the first layer
    k0 = keys[0]
    axes[1, 0].hist(results[k0]['leaf_cosine_offdiag'], bins=30, edgecolor='black', color='steelblue')
    axes[1, 0].axvline(results[k0]['mean_leaf_cosine'], color='red', linestyle='--',
                       label=f"Mean={results[k0]['mean_leaf_cosine']:.3f}")
    axes[1, 0].set_xlabel('Cosine Similarity'); axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Leaf Pairwise Cosine Dist. ({k0})'); axes[1, 0].legend()

    axes[1, 1].hist(results[k0]['gini_per_filter'], bins=30, edgecolor='black', color='darkorange')
    axes[1, 1].axvline(results[k0]['mean_gini'], color='red', linestyle='--',
                       label=f"Mean={results[k0]['mean_gini']:.3f}")
    axes[1, 1].set_xlabel('Gini Coefficient'); axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Filter Spatial Concentration ({k0})'); axes[1, 1].legend()

    pcs = results[k0]['parent_child_sims']
    if len(pcs):
        axes[1, 2].hist(pcs, bins=30, edgecolor='black', color='mediumpurple')
        axes[1, 2].axvline(pcs.mean(), color='red', linestyle='--', label=f'Mean={pcs.mean():.3f}')
        axes[1, 2].set_xlabel('Parent-Child Cosine'); axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title(f'Parent-Child Similarity Dist. ({k0})'); axes[1, 2].legend()
    axes[1, 2].axis('on')

    plt.suptitle('Weight Sparsity & Filter Structure Analysis', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(wt_dir, 'weight_sparsity_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    for k, v in results.items():
        np.savez(os.path.join(wt_dir, f'{k}_weight_stats.npz'),
                 **{kk: vv for kk, vv in v.items() if isinstance(vv, (np.ndarray, float, int))})

    print(f"Weight sparsity analysis saved to {wt_dir}")


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
                
                # Generate single reconstruction; accept (recon, dkl) tuples
                reconstructed = model(img)
                if isinstance(reconstructed, (tuple, list)):
                    reconstructed = reconstructed[0]

                if not torch.is_tensor(reconstructed):
                    axes[1, i].text(0.5, 0.5, 'no-tensor', ha='center')
                else:
                    # match spatial size and channels
                    if reconstructed.shape[2:] != img.shape[2:]:
                        reconstructed = F.interpolate(reconstructed, size=img.shape[2:], mode='bilinear', align_corners=False)
                    if reconstructed.shape[1] == 1 and img.shape[1] == 3:
                        reconstructed = reconstructed.repeat(1, 3, 1, 1)
                    elif reconstructed.shape[1] != img.shape[1]:
                        reconstructed = reconstructed[:, :img.shape[1], ...]

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
                # KL layers return (tensor, kl_loss) tuples — unpack
                if isinstance(x, tuple):
                    x = x[0]
                
                # Check if this is any taxonomic layer (single or multi-hierarchy)
                if _is_any_taxon_layer(conv_layer):
                    ch_per_h = _channels_per_single_hierarchy(conv_layer)
                    sub_layers_enc = _get_sub_layers(conv_layer)
                    multi_enc = len(sub_layers_enc) > 1
                    for sub_layer, h_idx in sub_layers_enc:
                        if multi_enc:
                            h_name = f'Encoder Layer {i+1} H{h_idx:02d}'
                            h_tree_dir = os.path.join(img_dir, f'encoder_layer_{i+1}',
                                                      f'hierarchy_{h_idx:02d}')
                            h_subtree_dir = h_tree_dir
                        else:
                            h_name = f'Encoder Layer {i+1}'
                            h_tree_dir = img_dir
                            h_subtree_dir = os.path.join(img_dir, f'encoder_layer_{i+1}')
                        # Slice activation channels that belong to this hierarchy
                        act_slice = x[:, h_idx * ch_per_h:(h_idx + 1) * ch_per_h, :, :]
                        visualize_taxonomy_tree(sub_layer, h_name, h_tree_dir,
                                                max_depth=4, activations=act_slice)
                        visualize_all_subtree_hierarchies(
                            sub_layer, h_name, h_subtree_dir, activations=act_slice)
                else:
                    # Regular conv - use grid visualization
                    visualize_feature_maps(x, os.path.join(img_dir, f'encoder_layer_{i+1}.png'),
                                         f'Encoder Layer {i+1}', max_maps=16)
                
                # KL/Resnet layers output log-probabilities — skip ReLU
                if not _is_kl_or_resnet_layer(conv_layer):
                    x = F.leaky_relu(x, negative_slope=0.01)
                # Only pool when use_maxpool=True; strided layers handle their
                # own downsampling when use_maxpool=False.
                if model.encoder.use_maxpool and model.encoder.strides[i] > 1:
                    x = F.max_pool2d(x, kernel_size=model.encoder.strides[i],
                                     stride=model.encoder.strides[i])
            
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
                # KL layers return (tensor, kl_loss) tuples — unpack
                if isinstance(x, tuple):
                    x = x[0]
                
                # Check if this is any taxonomic layer (single or multi-hierarchy)
                if _is_any_taxon_layer(deconv_layer):
                    ch_per_h = _channels_per_single_hierarchy(deconv_layer)
                    sub_layers_dec = _get_sub_layers(deconv_layer)
                    multi_dec = len(sub_layers_dec) > 1
                    for sub_layer, h_idx in sub_layers_dec:
                        if multi_dec:
                            h_name = f'Decoder Layer {i+1} H{h_idx:02d}'
                            h_tree_dir = os.path.join(img_dir, f'decoder_layer_{i+1}',
                                                      f'hierarchy_{h_idx:02d}')
                            h_subtree_dir = h_tree_dir
                        else:
                            h_name = f'Decoder Layer {i+1}'
                            h_tree_dir = img_dir
                            h_subtree_dir = os.path.join(img_dir, f'decoder_layer_{i+1}')
                        # Slice activation channels that belong to this hierarchy
                        act_slice = x[:, h_idx * ch_per_h:(h_idx + 1) * ch_per_h, :, :]
                        visualize_taxonomy_tree(sub_layer, h_name, h_tree_dir,
                                                max_depth=4, activations=act_slice)
                        visualize_all_subtree_hierarchies(
                            sub_layer, h_name, h_subtree_dir, activations=act_slice)
                else:
                    # Regular deconv - use grid visualization
                    visualize_feature_maps(x, os.path.join(img_dir, f'decoder_layer_{i+1}.png'),
                                         f'Decoder Layer {i+1}', max_maps=16)
                
                # KL/Resnet layers output log-probabilities — skip ReLU
                if not _is_kl_or_resnet_layer(deconv_layer):
                    x = F.leaky_relu(x, negative_slope=0.01)
            
            # Apply batch norm before final conv if present (matches decoder forward)
            if model.decoder.pre_final_norm is not None:
                x = model.decoder.pre_final_norm(x)
            
            # Final Conv2D layer to RGB
            x = model.decoder.final_conv(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'decoder_final_conv.png'), 'Final Conv (RGB)', max_maps=3)
            
            # Final reconstruction
            reconstructed = model(img)
            if isinstance(reconstructed, (tuple, list)):
                reconstructed = reconstructed[0]
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
            # model may return (output, dkl) or similar; accept both
            if isinstance(reconstructed, (tuple, list)):
                reconstructed = reconstructed[0]

            # ensure tensor
            if not torch.is_tensor(reconstructed):
                print(f"Skipping batch {i}: model returned non-tensor {type(reconstructed)}")
                continue

            # match spatial size if needed
            if reconstructed.dim() == 4 and reconstructed.shape[2:] != images.shape[2:]:
                reconstructed = F.interpolate(reconstructed, size=images.shape[2:], mode='bilinear', align_corners=False)

            # match channels: allow grayscale->RGB repeat, or truncate extra channels
            if reconstructed.shape[1] != images.shape[1]:
                rec_ch = reconstructed.shape[1]
                img_ch = images.shape[1]
                if rec_ch == 1 and img_ch == 3:
                    reconstructed = reconstructed.repeat(1, 3, 1, 1)
                elif rec_ch >= img_ch:
                    reconstructed = reconstructed[:, :img_ch, ...]
                else:
                    print(f"Skipping batch {i}: channel mismatch reconstructed {rec_ch} vs images {img_ch}")
                    continue

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
        
        encoder_n_hierarchies = [layer.get('n_hierarchies', 1) for layer in encoder_layers]
        encoder_n_hierarchies = encoder_n_hierarchies if any(x != 1 for x in encoder_n_hierarchies) else None
        decoder_n_hierarchies = [layer.get('n_hierarchies', 1) for layer in decoder_layers]
        decoder_n_hierarchies = decoder_n_hierarchies if any(x != 1 for x in decoder_n_hierarchies) else None

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
        encoder_n_hierarchies = None

        decoder_kernel_sizes = model_config.get('decoder_kernel_sizes')
        decoder_strides = model_config.get('decoder_strides')
        decoder_n_layers = model_config.get('decoder_n_layers', None)
        decoder_n_filters = model_config.get('decoder_n_filters', None)
        decoder_layer_types = model_config.get('decoder_layer_types', None)
        decoder_paddings = model_config.get('decoder_paddings', None)
        decoder_output_paddings = model_config.get('decoder_output_paddings', None)
        decoder_n_hierarchies = None

    return {
        'encoder_kernel_sizes': encoder_kernel_sizes,
        'encoder_strides': encoder_strides,
        'encoder_n_layers': encoder_n_layers,
        'encoder_n_filters': encoder_n_filters,
        'encoder_layer_types': encoder_layer_types,
        'encoder_n_hierarchies': encoder_n_hierarchies,
        'decoder_kernel_sizes': decoder_kernel_sizes,
        'decoder_strides': decoder_strides,
        'decoder_n_layers': decoder_n_layers,
        'decoder_n_filters': decoder_n_filters,
        'decoder_layer_types': decoder_layer_types,
        'decoder_paddings': decoder_paddings,
        'decoder_output_paddings': decoder_output_paddings,
        'decoder_n_hierarchies': decoder_n_hierarchies
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
        encoder_n_hierarchies = layer_params.get('encoder_n_hierarchies', None)
        decoder_n_hierarchies = layer_params.get('decoder_n_hierarchies', None)
        
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
        encoder_n_hierarchies = None
        decoder_n_hierarchies = None
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
        encoder_n_hierarchies=encoder_n_hierarchies,
        decoder_n_hierarchies=decoder_n_hierarchies,
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
        sub_layers = _get_sub_layers(layer)
        multi = len(sub_layers) > 1
        if multi:
            print(f"  {layer_name}: {len(sub_layers)} independent hierarchies")
        for sub_layer, h_idx in sub_layers:
            if multi:
                h_name = f"{layer_name} H{h_idx:02d}"
                h_save_dir = os.path.join(layer_save_dir, f'hierarchy_{h_idx:02d}')
            else:
                h_name = layer_name
                h_save_dir = layer_save_dir
            visualize_taxonomy_tree(sub_layer, h_name, h_save_dir, max_depth=4)
            visualize_all_subtree_hierarchies(sub_layer, h_name, h_save_dir)
    
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
        sub_layers = _get_sub_layers(layer)
        multi = len(sub_layers) > 1
        if multi:
            print(f"  {layer_name}: {len(sub_layers)} independent hierarchies")
        for sub_layer, h_idx in sub_layers:
            if multi:
                h_name = f"{layer_name} H{h_idx:02d}"
                h_save_dir = os.path.join(layer_save_dir, f'hierarchy_{h_idx:02d}')
            else:
                h_name = layer_name
                h_save_dir = layer_save_dir
            visualize_taxonomy_tree(sub_layer, h_name, h_save_dir, max_depth=4)
            visualize_all_subtree_hierarchies(sub_layer, h_name, h_save_dir)
    
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

    # Analysis 9: Partonomy sparsity suite
    print("\n" + "=" * 60)
    print("9. Partonomy Sparsity Analysis")
    print("=" * 60)
    analyze_partonomy_sparsity(model, test_loader, device, save_dir)

    # Analysis 10: Weight sparsity
    print("\n" + "=" * 60)
    print("10. Weight Sparsity Analysis")
    print("=" * 60)
    analyze_weight_sparsity(model, save_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
