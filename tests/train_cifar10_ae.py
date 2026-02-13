"""
Training script for CIFAR-10 Taxonomic Autoencoder

Trains the CIFAR10TaxonAutoencoder and saves the model checkpoint.
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CIFAR10TaxonAutoencoder
from src.utils.dataloader import CIFAR10Loader


def train_autoencoder(
    model,
    train_loader,
    test_loader,
    epochs=50,
    lr=0.001,
    device='cuda',
    save_dir='outputs/cifar10/training',
    kl_weight=1.0
):
    """Train the autoencoder and save checkpoints.
    
    Args:
        kl_weight: Weight for KL divergence loss (default 1.0). Set to 0 to ignore KL.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    
    train_losses = []
    test_losses = []
    
    print(f"Training on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Handle both (reconstruction, kl) and just reconstruction returns
            result = model(images)
            if isinstance(result, tuple):
                reconstructed, kl = result
                recon_loss = criterion(reconstructed, images)
                kl_loss = kl if kl_weight > 0 else 0.0
                loss = recon_loss + kl_weight * kl_loss
                train_kl_loss += (kl.item() if hasattr(kl, 'item') else kl)
            else:
                reconstructed = result
                loss = criterion(reconstructed, images)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += criterion(reconstructed, images).item()
        
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                
                # Handle both (reconstruction, kl) and just reconstruction returns
                result = model(images)
                if isinstance(result, tuple):
                    reconstructed, kl = result
                    recon_loss = criterion(reconstructed, images)
                    kl_loss = kl if kl_weight > 0 else 0.0
                    loss = recon_loss + kl_weight * kl_loss
                    test_kl_loss += (kl.item() if hasattr(kl, 'item') else kl)
                else:
                    reconstructed = result
                    loss = criterion(reconstructed, images)
                
                test_loss += loss.item()
                test_recon_loss += criterion(reconstructed, images).item()
        
        test_loss /= len(test_loader)
        test_recon_loss /= len(test_loader)
        test_kl_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Print losses with KL if present
        if train_kl_loss > 0 or test_kl_loss > 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} (Recon: {train_recon_loss:.6f}, KL: {train_kl_loss:.6f}), "
                  f"Test Loss: {test_loss:.6f} (Recon: {test_recon_loss:.6f}, KL: {test_kl_loss:.6f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        # Generate reconstructions every epoch (or every 5 epochs for faster training)
        visualize_reconstructions(model, test_loader, device, save_dir, num_images=8, epoch=epoch+1)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_losses': train_losses,
                'test_losses': test_losses,
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining complete! Model saved to {save_dir}")
    return model, train_losses, test_losses


def visualize_reconstructions(model, test_loader, device, save_dir, num_images=8, epoch=None):
    """Visualize original vs reconstructed images.
    
    Args:
        epoch: If provided, saves with epoch-specific filename (e.g., 'reconstructions_epoch_5.png')
    """
    model.eval()
    
    # Get a batch
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        result = model(images)
        # Handle both (reconstruction, kl) and just reconstruction returns
        if isinstance(result, tuple):
            reconstructed = result[0]
        else:
            reconstructed = result
    
    # Unnormalize images (from [-1, 1] to [0, 1])
    images = images.cpu() * 0.5 + 0.5
    reconstructed = reconstructed.cpu() * 0.5 + 0.5
    
    # Plot
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
    
    plt.tight_layout()
    
    # Save with epoch-specific filename if epoch is provided
    if epoch is not None:
        filename = f'reconstructions_epoch_{epoch}.png'
    else:
        filename = 'reconstructions.png'
    
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Only print message for final reconstruction (without epoch number)
    if epoch is None:
        print(f"Reconstructions saved to {save_dir}/{filename}")


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
    parser = argparse.ArgumentParser(description='Train CIFAR-10 Taxonomic Autoencoder')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--latent-dim', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    
    args = parser.parse_args()
    
    # Load config if provided, otherwise use defaults
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
        
        # Training-specific parameters (optional)
        epochs = config.get('training', {}).get('epochs', 20)
        lr = config.get('training', {}).get('learning_rate', 0.001)
        kl_weight = config.get('training', {}).get('kl_weight', 1.0)
        
        # Use experiment_name from config, or fall back to training_save_dir
        experiment_name = config.get('experiment_name', None)
        if experiment_name:
            save_dir_prefix = f"outputs/cifar10/training/{experiment_name}"
        else:
            save_dir_prefix = config.get('output', {}).get('training_save_dir', 'outputs/cifar10/training')
    else:
        # Default configuration
        batch_size = 128
        epochs = 20
        lr = 0.001
        kl_weight = 1.0
        data_root = './data/cifar10'
        latent_dim = 256
        temperature = 1.0
        encoder_kernel_sizes = [5, 5, 5]
        decoder_kernel_sizes = [6, 6, 5]
        encoder_strides = [2, 2]
        decoder_strides = [2, 2, 1]
        encoder_n_layers = None
        decoder_n_layers = None
        decoder_paddings = None
        decoder_output_paddings = None
        use_maxpool = True
        random_init_alphas = False
        alpha_init_distribution = 'uniform'
        alpha_init_range = None
        alpha_init_seed = None
        save_dir_prefix = 'outputs/cifar10/training'
    
    # Command line args override config
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.epochs is not None:
        epochs = args.epochs
    if args.lr is not None:
        lr = args.lr
    if args.latent_dim is not None:
        latent_dim = args.latent_dim
    if args.temperature is not None:
        temperature = args.temperature
    
    # Use experiment name if provided in config, otherwise use timestamp
    if args.config and 'experiment_name' in config:
        save_dir = save_dir_prefix
    else:
        save_dir = f'{save_dir_prefix}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("CIFAR-10 Taxonomic Autoencoder Training")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Latent dim: {latent_dim}")
    print(f"Temperature: {temperature}")
    print(f"Learning rate: {lr}")
    print(f"Encoder kernel sizes: {encoder_kernel_sizes}")
    print(f"Decoder kernel sizes: {decoder_kernel_sizes if decoder_kernel_sizes else '[4, 4, 3] (default)'}")
    print(f"Encoder strides: {encoder_strides}")
    print(f"Decoder strides: {decoder_strides if decoder_strides else '[2, 2, 1] (default)'}")
    print(f"Use max pooling: {use_maxpool}")
    print(f"Random alpha init: {random_init_alphas} (dist={alpha_init_distribution}, range={alpha_init_range}, seed={alpha_init_seed})")
    print(f"Data directory: {data_root}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    loader = CIFAR10Loader(batch_size=batch_size, root=data_root)
    train_loader, test_loader = loader.get_loaders()
    print(f"Train samples: {len(loader.trainset)}")
    print(f"Test samples: {len(loader.testset)}")
    
    # Create model
    print("\nCreating CIFAR10TaxonAutoencoder...")
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
    
    # Train
    print("\nStarting training...\n")
    model, train_losses, test_losses = train_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_dir=save_dir,
        kl_weight=kl_weight
    )
    
    # Visualize reconstructions
    print("\nGenerating reconstruction visualizations...")
    visualize_reconstructions(model, test_loader, device, save_dir)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final test loss: {test_losses[-1]:.6f}")
    print(f"All outputs saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
