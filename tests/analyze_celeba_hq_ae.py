"""
Analysis script for CelebA-HQ Autoencoder.
Computes reconstruction metrics and saves sample reconstructions.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CelebAHQTaxonAutoencoder
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    return model, checkpoint


def visualize_reconstructions(model, data_loader, device, save_dir, num_images=8):
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        recons = model(images)
    images = images.cpu()
    recons = recons.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        axes[1, i].imshow(recons[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'analysis_reconstructions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(model, data_loader, device, num_batches=10):
    model.eval()
    mse_list = []
    psnr_list = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            recons = model(images)
            mse = torch.mean((recons - images) ** 2, dim=[1, 2, 3])
            mse_list.append(mse.cpu().numpy())
            psnr = 10 * torch.log10(torch.ones_like(mse) / mse.clamp(min=1e-10))
            psnr_list.append(psnr.cpu().numpy())

    mse_all = np.concatenate(mse_list) if mse_list else np.array([])
    psnr_all = np.concatenate(psnr_list) if psnr_list else np.array([])
    return mse_all, psnr_all


def main():
    parser = argparse.ArgumentParser(description='Analyze CelebA-HQ Autoencoder')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--num-batches', type=int, default=10, help='Batches to use for metrics')

    args = parser.parse_args()

    config = None
    if args.config:
        config = load_config(args.config)
        batch_size = config['data']['batch_size']
        num_workers = config['data'].get('num_workers', 4)
        data_root = config['data']['data_root']
        image_size = config['data'].get('image_size', 256)
        val_split = config['data'].get('val_split', 0.05)
        experiment_name = config.get('experiment_name', None)
        save_dir_prefix = config.get('output', {}).get('analysis_save_dir', 'outputs/celebahq/analysis')
    else:
        batch_size = 16
        num_workers = 4
        data_root = './data/celeba_hq'
        image_size = 256
        val_split = 0.0
        experiment_name = None
        save_dir_prefix = 'outputs/celebahq/analysis'

    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.data_root is not None:
        data_root = args.data_root
    if args.num_workers is not None:
        num_workers = args.num_workers

    save_dir = save_dir_prefix if experiment_name else os.path.join(save_dir_prefix, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('CelebA-HQ Autoencoder Analysis')
    print('=' * 60)
    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Data root: {data_root}')
    print(f'Image size: {image_size}')
    print(f'Batch size: {batch_size}')
    print(f'Save dir: {save_dir}')
    print('=' * 60)

    loader = CelebAHQLoader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_split=val_split,
    )
    train_loader, val_loader = loader.get_loaders()
    eval_loader = val_loader if val_loader is not None else train_loader

    model, checkpoint = load_model(args.checkpoint, device, config)

    visualize_reconstructions(model, eval_loader, device, save_dir)

    mse_all, psnr_all = compute_metrics(model, eval_loader, device, num_batches=args.num_batches)

    if mse_all.size > 0:
        print(f'Mean MSE: {mse_all.mean():.6f}')
        print(f'Median MSE: {np.median(mse_all):.6f}')
        print(f'Mean PSNR: {psnr_all.mean():.2f} dB')
    else:
        print('No samples evaluated (empty loader).')

    np.savez(
        os.path.join(save_dir, 'metrics.npz'),
        mse=mse_all,
        psnr=psnr_all,
    )
    print(f'Metrics saved to {os.path.join(save_dir, "metrics.npz")}')


if __name__ == '__main__':
    main()
