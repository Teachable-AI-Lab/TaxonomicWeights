"""
Training script for CelebA-HQ Autoencoder (256x256x3).
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def visualize_reconstructions(model, data_loader, device, save_dir, num_images=8, epoch=None):
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
    filename = f'reconstructions_epoch_{epoch}.png' if epoch is not None else 'reconstructions.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            optimizer.zero_grad()
            recons = model(images)
            loss = criterion(recons, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    images = images.to(device)
                    recons = model(images)
                    loss = criterion(recons, images)
                    val_running += loss.item() * images.size(0)
            epoch_val_loss = val_running / len(val_loader.dataset)
        else:
            epoch_val_loss = None
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - train MSE: {epoch_train_loss:.6f}" +
              (f", val MSE: {epoch_val_loss:.6f}" if epoch_val_loss is not None else ""))

        # Save reconstructions every epoch
        visualize_reconstructions(model, val_loader or train_loader, device, save_dir, epoch=epoch+1)

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    # Final save
    final_ckpt = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(final_ckpt, os.path.join(save_dir, 'final_model.pt'))

    # Plot curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train MSE')
    if any(v is not None for v in val_losses):
        plt.plot([v for v in val_losses if v is not None], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train CelebA-HQ Autoencoder')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None)

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        batch_size = config['data']['batch_size']
        num_workers = config['data'].get('num_workers', 4)
        data_root = config['data']['data_root']
        image_size = config['data'].get('image_size', 256)
        val_split = config['data'].get('val_split', 0.05)
        train_subset = config['data'].get('train_subset', None)
        val_subset = config['data'].get('val_subset', None)
        epochs = config['training'].get('epochs', 20)
        lr = config['training'].get('learning_rate', 1e-4)
        experiment_name = config.get('experiment_name', None)
        save_dir_prefix = config.get('output', {}).get('training_save_dir', 'outputs/celebahq/training')
        
        # Model configuration - check if taxonomic or baseline
        model_config = config.get('model', {})
        use_taxonomic = 'encoder_layers' in model_config or 'decoder_layers' in model_config
    else:
        batch_size = 16
        num_workers = 4
        data_root = './data/celeba_hq'
        image_size = 256
        val_split = 0.05
        train_subset = None
        val_subset = None
        epochs = 20
        lr = 1e-4
        experiment_name = None
        save_dir_prefix = 'outputs/celebahq/training'
        use_taxonomic = False
        model_config = {}

    # CLI overrides
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.epochs is not None:
        epochs = args.epochs
    if args.lr is not None:
        lr = args.lr
    if args.data_root is not None:
        data_root = args.data_root
    if args.num_workers is not None:
        num_workers = args.num_workers

    save_dir = save_dir_prefix if experiment_name else os.path.join(save_dir_prefix, datetime.now().strftime('%Y%m%d_%H%M%S'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('CelebA-HQ Autoencoder Training')
    print('=' * 60)
    print(f'Device: {device}')
    print(f'Model type: {"Taxonomic" if use_taxonomic else "Baseline"}')
    print(f'Batch size: {batch_size}')
    print(f'Epochs: {epochs}')
    print(f'Learning rate: {lr}')
    print(f'Data root: {data_root}')
    print(f'Image size: {image_size}')
    print(f'Val split: {val_split}')
    print(f'Train subset: {train_subset if train_subset else "Full dataset"}')
    print(f'Val subset: {val_subset if val_subset else "Full dataset"}')
    print(f'Save dir: {save_dir}')
    print('=' * 60)

    print("\nLoading CelebA-HQ dataset...")
    loader = CelebAHQLoader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_split=val_split,
        train_subset=train_subset,
        val_subset=val_subset,
    )
    train_loader, val_loader = loader.get_loaders()
    
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")
    else:
        print("Val samples: 0 (no validation split)")

    print("\nCreating CelebAHQTaxonAutoencoder...")
    # Parse encoder and decoder layers from config
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
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    print('\nStarting training...\n')
    model, train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_dir=save_dir,
    )

    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'Final train MSE: {train_losses[-1]:.6f}')
    if val_losses[-1] is not None:
        print(f'Final val MSE: {val_losses[-1]:.6f}')
    print(f'Outputs saved to: {save_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
