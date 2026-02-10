"""
Analysis script for CelebA-HQ Autoencoder.
Computes reconstruction metrics and saves sample reconstructions.
Performs comprehensive analysis including:
- Filter visualization at each hierarchy level (encoder Conv & decoder Deconv)
- Latent space sparsity analysis
- Reconstruction quality metrics
- Feature activation patterns
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CelebAHQTaxonAutoencoder
from src.model.taxon_layers import TaxonConv, TaxonDeconv
from src.utils.dataloader import CelebAHQLoader


def parse_layer_config(config_layers):
    """Parse layer configuration from JSON into model parameters."""
    if not config_layers:
        return None, None, None, None, None, None
    
    n_layers_list = []
    n_filters_list = []
    layer_types_list = []
    kernel_sizes_list = []
    strides_list = []
    paddings_list = []
    output_paddings_list = []
    
    for layer in config_layers:
        layer_type = layer.get('layer_type', 'taxonomic_conv')
        layer_types_list.append(layer_type)
        
        if 'n_layers' in layer:
            n_layers_list.append(layer['n_layers'])
        else:
            n_layers_list.append(None)
        
        if 'n_filters' in layer:
            n_filters_list.append(layer['n_filters'])
        else:
            n_filters_list.append(None)
        
        kernel_sizes_list.append(layer.get('kernel_size', 3))
        strides_list.append(layer.get('stride', 1))
        paddings_list.append(layer.get('padding', None))
        output_paddings_list.append(layer.get('output_padding', 0))
    
    # Clean up None lists
    n_layers_out = n_layers_list if any(x is not None for x in n_layers_list) else None
    n_filters_out = n_filters_list if any(x is not None for x in n_filters_list) else None
    paddings_out = paddings_list if any(x is not None for x in paddings_list) else None
    
    return (n_layers_out, n_filters_out, layer_types_list, 
            kernel_sizes_list, strides_list, paddings_out, output_paddings_list)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def load_model(checkpoint_path, device, config=None):
    """Load model from checkpoint. Uses config to determine if taxonomic or baseline."""
    import os
    
    # Verify checkpoint file exists and is readable
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check if file is actually a JSON config instead of a checkpoint
    if checkpoint_path.endswith('.json'):
        raise ValueError(f"Provided path appears to be a JSON config file, not a checkpoint: {checkpoint_path}\n"
                        f"Please provide a .pt checkpoint file (e.g., final_model.pt or checkpoint_epoch_X.pt)")
    
    # Check file content to detect JSON vs PyTorch checkpoint
    with open(checkpoint_path, 'rb') as f:
        first_bytes = f.read(10)
        if first_bytes.startswith(b'{') or first_bytes.startswith(b'{\n'):
            raise ValueError(f"File appears to be JSON format, not a PyTorch checkpoint: {checkpoint_path}\n"
                           f"First bytes: {first_bytes}\n"
                           f"Expected a .pt checkpoint file saved with torch.save()")
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path} ({file_size / 1024 / 1024:.1f} MB)")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Checkpoint path: {os.path.abspath(checkpoint_path)}")
        print(f"\nMake sure you're pointing to a .pt checkpoint file, not a .json config file.")
        print(f"Valid checkpoint files are saved by the training script with names like:")
        print(f"  - final_model.pt")
        print(f"  - checkpoint_epoch_X.pt")
        raise
    
    # Determine model type from config
    use_taxonomic = False
    if config:
        model_config = config.get('model', {})
        use_taxonomic = 'encoder_layers' in model_config or 'decoder_layers' in model_config
    
    # Parse layer configs
    model_config = config.get('model', {})
    encoder_layers = model_config.get('encoder_layers', [])
    decoder_layers = model_config.get('decoder_layers', [])
    
    (enc_n_layers, enc_n_filters, enc_layer_types, enc_kernel_sizes, 
        enc_strides, enc_paddings, enc_output_paddings) = parse_layer_config(encoder_layers)
    
    (dec_n_layers, dec_n_filters, dec_layer_types, dec_kernel_sizes, 
        dec_strides, dec_paddings, dec_output_paddings) = parse_layer_config(decoder_layers)
    
    # Get taxonomic-specific settings
    temperature = model_config.get('temperature', 1.0)
    use_maxpool = model_config.get('use_maxpool', True)
    random_init_alphas = model_config.get('random_init_alphas', False)
    alpha_init_distribution = model_config.get('alpha_init_distribution', 'uniform')
    alpha_init_range = model_config.get('alpha_init_range', None)
    alpha_init_seed = model_config.get('alpha_init_seed', None)
    output_activation = model_config.get('output_activation', 'sigmoid')
    
    model = CelebAHQTaxonAutoencoder(
        latent_dim=model_config.get('latent_dim', 256),
        temperature=temperature,
        encoder_kernel_sizes=enc_kernel_sizes,
        decoder_kernel_sizes=dec_kernel_sizes,
        encoder_strides=enc_strides,
        decoder_strides=dec_strides,
        encoder_n_layers=enc_n_layers,
        decoder_n_layers=dec_n_layers,
        encoder_n_filters=enc_n_filters,
        decoder_n_filters=dec_n_filters,
        encoder_layer_types=enc_layer_types,
        decoder_layer_types=dec_layer_types,
        decoder_paddings=dec_paddings,
        decoder_output_paddings=dec_output_paddings,
        use_maxpool=use_maxpool,
        random_init_alphas=random_init_alphas,
        alpha_init_distribution=alpha_init_distribution,
        alpha_init_range=alpha_init_range,
        alpha_init_seed=alpha_init_seed,
        output_activation=output_activation
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.6f}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"Val loss: {checkpoint['val_loss']:.6f}")
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
            # Average across input channels
            filter_img = w_norm[i].mean(axis=0)
            axes[i].imshow(filter_img, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'F{i}', fontsize=8)
        
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
            axes[i].set_title(f'F{i}', fontsize=8)
        
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
            parent_id = (node_id - 1) // 2
            
            # Position: evenly distribute across available width
            if num_nodes == 1:
                x_pos = 0.5
            else:
                x_pos = padding + (node_idx / (num_nodes - 1)) * available_width
            
            positions[node_id] = (x_pos, y_pos)
            node_labels[node_id] = f"N{node_id}"
            
            # Track image indices
            if image_type == "activations":
                # For activations: each node gets equal channel slices
                start_ch = cumulative_start
                end_ch = cumulative_start + per_node_channels
                node_image_indices[node_id] = (start_ch, end_ch)
                cumulative_start = end_ch
            else:
                # For filters: (level, filter_idx)
                node_image_indices[node_id] = (level - 1, node_idx)
            
            # Edge from parent
            edges.append((parent_id, node_id))
            
            # Alpha label (which child branch this is: 0 or 1)
            if level > 0:
                parent_node_in_level = parent_id - (2 ** (level - 1) - 1)
                child_branch = node_idx % 2
                
                if level - 1 < len(alpha_values):
                    alpha_array = alpha_values[level - 1]
                    if parent_node_in_level < alpha_array.shape[0]:
                        alpha_val = alpha_array[parent_node_in_level, 0]
                        if child_branch == 0:
                            edge_labels[(parent_id, node_id)] = f"α={alpha_val:.2f}"
                        else:
                            edge_labels[(parent_id, node_id)] = f"1-α={1-alpha_val:.2f}"
                    else:
                        edge_labels[(parent_id, node_id)] = ""
                else:
                    edge_labels[(parent_id, node_id)] = ""
        
        if image_type == "activations":
            cumulative_start = cumulative_start
        node_counter += num_nodes
    
    # Pre-render all image thumbnails to avoid expensive inline rendering
    image_thumbnails = {}
    image_sizes = {}
    print(f"  Pre-rendering {len(positions)} {image_type} thumbnails...")
    for node_id, (x, y) in positions.items():
        img_idx = node_image_indices[node_id]
        
        if image_type == "activations":
            start_ch, end_ch = img_idx
            if end_ch <= acts.shape[0]:
                act_slice = acts[start_ch:end_ch]
                if act_slice.shape[0] > 0:
                    img = act_slice.mean(axis=0)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                else:
                    img = np.zeros((8, 8))
            else:
                img = np.zeros((8, 8))
            image_thumbnails[node_id] = img
            image_sizes[node_id] = img.shape
        else:
            level, filter_idx = img_idx
            if level < len(hierarchy_weights):
                w_tensor = hierarchy_weights[level]
                w_np = w_tensor.detach().cpu().numpy()
                
                if filter_idx < w_np.shape[0]:
                    filter_img = w_np[filter_idx].mean(axis=0)
                    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-5)
                else:
                    filter_img = np.zeros((3, 3))
            else:
                filter_img = np.zeros((3, 3))
            image_thumbnails[node_id] = filter_img
            image_sizes[node_id] = filter_img.shape
            
    
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
        
        if label:
            ax.text(mid_x, mid_y, label, fontsize=8, ha='center', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7), zorder=2)
                
    
    # Draw nodes with image visualizations using OffsetImage for better control
    print(f"  Drawing {len(positions)} nodes with {image_type}...")
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    for node_id, (x, y) in positions.items():
        if node_id in image_thumbnails:
            img = image_thumbnails[node_id]
            imagebox = OffsetImage(img, zoom=base_zoom, cmap='viridis')
            ab = AnnotationBbox(imagebox, (x, y), frameon=True, pad=0.1,
                               bboxprops=dict(edgecolor='blue', linewidth=1.5, facecolor='white'))
            ax.add_artist(ab)
            
            # Add node label below the image
            ax.text(x, y - 0.04, node_labels[node_id], fontsize=9, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8), zorder=3)
        else:
            # Fallback circle if no image
            ax.plot(x, y, 'o', markersize=10, color='gray', zorder=3)
            ax.text(x, y, node_labels[node_id], fontsize=9, ha='center', va='center', zorder=4)
            
    
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
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Encoding")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            latent = model.encode(images)
            all_latents.append(latent.cpu().numpy())
    
    all_latents = np.concatenate(all_latents, axis=0)
    print(f"Collected {all_latents.shape[0]} latent vectors")
    
    # Handle spatial latent representations (flatten if needed)
    if all_latents.ndim > 2:
        print(f"  Spatial latent shape: {all_latents.shape}")
        original_shape = all_latents.shape
        all_latents = all_latents.reshape(all_latents.shape[0], -1)
        print(f"  Flattened to: {all_latents.shape}")
    
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
    
    # Mean activation per dimension (sample first 1000 dims if too many)
    dims_to_show = min(1000, len(mean_activation))
    axes[0, 0].bar(range(dims_to_show), mean_activation[:dims_to_show])
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Mean |Activation|')
    axes[0, 0].set_title(f'Mean Activation per Dimension (first {dims_to_show})')
    
    # Std activation per dimension
    axes[0, 1].bar(range(dims_to_show), std_activation[:dims_to_show])
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Std Activation')
    axes[0, 1].set_title(f'Activation Std per Dimension (first {dims_to_show})')
    
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
            data_iter = iter(data_loader)
            images, _ = next(data_iter)
        
        images = images[:num_images].to(device)
        
        with torch.no_grad():
            reconstructed = model(images)
        
        # Move to CPU
        images = images.cpu()
        reconstructed = reconstructed.cpu()
        
        # Create figure with 2 rows (original, reconstructed)
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        
        for i in range(num_images):
            # Original
            axes[0, i].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).clamp(0, 1))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)
        
        plt.suptitle(f'Reconstruction Set {set_idx + 1}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, f'reconstruction_set_{set_idx + 1}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Reconstructions saved to {recon_dir}")


def analyze_reconstruction_quality(model, data_loader, device, save_dir, num_batches=20):
    """Analyze reconstruction quality metrics."""
    
    model.eval()
    mse_losses = []
    mae_losses = []
    
    print(f"Computing reconstruction metrics on {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Analyzing")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            reconstructed = model(images)
            
            mse = torch.mean((reconstructed - images) ** 2, dim=[1, 2, 3])
            mae = torch.mean(torch.abs(reconstructed - images), dim=[1, 2, 3])
            
            mse_losses.extend(mse.cpu().numpy())
            mae_losses.extend(mae.cpu().numpy())
    
    mse_losses = np.array(mse_losses)
    mae_losses = np.array(mae_losses)
    
    # Compute PSNR (Peak Signal-to-Noise Ratio)
    # Assuming pixel values in [0, 1]
    psnr = 10 * np.log10(1.0 / (mse_losses + 1e-10))
    
    print(f"\nReconstruction Quality Metrics:")
    print(f"  MSE - Mean: {mse_losses.mean():.6f}, Std: {mse_losses.std():.6f}")
    print(f"  MAE - Mean: {mae_losses.mean():.6f}, Std: {mae_losses.std():.6f}")
    print(f"  PSNR - Mean: {psnr.mean():.2f} dB, Std: {psnr.std():.2f} dB")
    
    # Save metrics
    np.savez(
        os.path.join(save_dir, 'reconstruction_metrics.npz'),
        mse=mse_losses,
        mae=mae_losses,
        psnr=psnr
    )
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MSE distribution
    axes[0].hist(mse_losses, bins=50, edgecolor='black')
    axes[0].axvline(mse_losses.mean(), color='red', linestyle='--', 
                    label=f'Mean: {mse_losses.mean():.6f}')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MSE Distribution')
    axes[0].legend()
    
    # MAE distribution
    axes[1].hist(mae_losses, bins=50, edgecolor='black')
    axes[1].axvline(mae_losses.mean(), color='red', linestyle='--',
                    label=f'Mean: {mae_losses.mean():.6f}')
    axes[1].set_xlabel('MAE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('MAE Distribution')
    axes[1].legend()
    
    # PSNR distribution
    axes[2].hist(psnr, bins=50, edgecolor='black')
    axes[2].axvline(psnr.mean(), color='red', linestyle='--',
                    label=f'Mean: {psnr.mean():.2f} dB')
    axes[2].set_xlabel('PSNR (dB)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('PSNR Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Reconstruction quality analysis saved to {save_dir}")


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
            img_display = img.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
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
                
                x = torch.nn.functional.relu(x)
                if i < len(model.encoder.strides) and model.encoder.strides[i] > 1:
                    if model.encoder.use_maxpool:
                        x = torch.nn.functional.max_pool2d(x, model.encoder.strides[i])
                    else:
                        x = torch.nn.functional.avg_pool2d(x, model.encoder.strides[i])
            
            # Latent (may be spatial or flat, so flatten for visualization)
            latent = model.encode(img)
            
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
                
                x = torch.nn.functional.relu(x)
            
            # Final Conv2D layer to RGB
            x = model.decoder.final_conv(x)
            visualize_feature_maps(x, os.path.join(img_dir, 'decoder_final_conv.png'), 'Final Conv (RGB)', max_maps=3)
            
            # Final reconstruction
            reconstructed = model(img)
            recon_display = reconstructed.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
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
        fmap = maps[i].numpy()
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Ch{i}', fontsize=8)
    
    # Turn off extra axes
    for ax in axes[num_maps:]:
        ax.axis('off')
    
    plt.suptitle(f'{title} (showing {num_maps}/{maps.shape[0]} feature maps)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze CelebA-HQ Autoencoder')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file (must specify checkpoint_path in analysis section)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None)

    args = parser.parse_args()

    # Load config (now required)
    config = load_config(args.config)
    batch_size = config['data'].get('batch_size', 16)
    num_workers = config['data'].get('num_workers', 4)
    data_root = config['data']['data_root']
    image_size = config['data'].get('image_size', 256)
    val_split = config['data'].get('val_split', 0.05)
    experiment_name = config.get('experiment_name', None)
    save_dir_prefix = config.get('output', {}).get('analysis_save_dir', 'outputs/celebahq/analysis')
    
    # Analysis settings from config
    analysis_config = config.get('analysis', {})
    checkpoint_path = args.checkpoint or analysis_config.get('checkpoint_path')
    num_latent_batches = analysis_config.get('num_latent_batches', 50)
    num_reconstruction_batches = analysis_config.get('num_reconstruction_batches', 20)
    num_multiple_recon_images = analysis_config.get('num_multiple_recon_images', 8)
    num_reconstructions_per_image = analysis_config.get('num_reconstructions_per_image', 8)
    num_activation_images = analysis_config.get('num_activation_images', 3)

    if not checkpoint_path:
        raise ValueError("Must specify checkpoint_path in config file under 'analysis' section or provide --checkpoint argument")

    # CLI overrides
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.data_root is not None:
        data_root = args.data_root
    if args.num_workers is not None:
        num_workers = args.num_workers

    save_dir = save_dir_prefix if experiment_name else os.path.join(save_dir_prefix, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 80)
    print('CelebA-HQ Taxonomic Autoencoder Analysis')
    print('=' * 80)
    print(f'Device: {device}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Data root: {data_root}')
    print(f'Image size: {image_size}')
    print(f'Batch size: {batch_size}')
    print(f'Save dir: {save_dir}')
    print('=' * 80)

    # Load data
    print("\nLoading CelebA-HQ dataset...")
    loader = CelebAHQLoader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_split=val_split,
    )
    train_loader, val_loader = loader.get_loaders()
    eval_loader = val_loader if val_loader is not None else train_loader

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(checkpoint_path, device, config)

    # Analysis 1: Visualize encoder Conv filters
    print("\n" + "=" * 80)
    print("1. Visualizing Encoder TaxonConv Filters")
    print("=" * 80)
    for i in range(len(model.encoder.conv_layers)):
        layer_name = f'encoder_layer_{i+1}'
        print(f"\nProcessing {layer_name}...")
        visualize_taxonconv_filters(model, save_dir, layer_name=layer_name, n_cols=8)

    # Analysis 2: Visualize taxonomy trees for encoder
    print("\n" + "=" * 80)
    print("2. Visualizing Encoder Taxonomy Trees")
    print("=" * 80)
    encoder_layers = []
    for i, layer in enumerate(model.encoder.conv_layers):
        layer_name = f'encoder_layer_{i+1}'
        encoder_layers.append((layer, layer_name))
    
    for layer, layer_name in encoder_layers:
        print(f"\nProcessing {layer_name}...")
        layer_save_dir = os.path.join(save_dir, f"{layer_name}_filters")
        visualize_taxonomy_tree(layer, layer_name, layer_save_dir, max_depth=4)

    # Analysis 3: Visualize decoder Deconv filters
    print("\n" + "=" * 80)
    print("3. Visualizing Decoder TaxonDeconv Filters")
    print("=" * 80)
    for i in range(len(model.decoder.deconv_layers)):
        layer_name = f'decoder_layer_{i+1}'
        print(f"\nProcessing {layer_name}...")
        visualize_taxondeconv_filters(model, save_dir, layer_name=layer_name, n_cols=8)

    # Analysis 4: Visualize taxonomy trees for decoder
    print("\n" + "=" * 80)
    print("4. Visualizing Decoder Taxonomy Trees")
    print("=" * 80)
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
    print("\n" + "=" * 80)
    print("5. Analyzing Latent Space")
    print("=" * 80)
    analyze_latent_sparsity(model, eval_loader, device, save_dir, num_batches=num_latent_batches)

    # Analysis 6: Reconstruction quality metrics
    print("\n" + "=" * 80)
    print("6. Analyzing Reconstruction Quality")
    print("=" * 80)
    analyze_reconstruction_quality(model, eval_loader, device, save_dir, num_batches=num_reconstruction_batches)

    # Analysis 7: Multiple reconstructions
    print("\n" + "=" * 80)
    print("7. Visualizing Multiple Reconstructions")
    print("=" * 80)
    visualize_multiple_reconstructions(
        model, eval_loader, device, save_dir,
        num_images=num_multiple_recon_images,
        num_reconstructions=num_reconstructions_per_image
    )

    # Analysis 8: Layer activations for random images
    print("\n" + "=" * 80)
    print("8. Visualizing Layer Activations")
    print("=" * 80)
    visualize_layer_activations(model, eval_loader, device, save_dir, num_images=num_activation_images)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
