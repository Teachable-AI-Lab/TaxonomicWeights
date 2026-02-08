"""
Training script for CIFAR-10 Taxonomic Autoencoder

Trains the CIFAR10TaxonAutoencoder and saves the model checkpoint.
"""

import os
import sys
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
    save_dir='outputs/training'
):
    """Train the autoencoder and save checkpoints."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    print(f"Training on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
        
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


def visualize_reconstructions(model, test_loader, device, save_dir, num_images=8):
    """Visualize original vs reconstructed images."""
    model.eval()
    
    # Get a batch
    images, _ = next(iter(test_loader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
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
    plt.savefig(os.path.join(save_dir, 'reconstructions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstructions saved to {save_dir}/reconstructions.png")


def main():
    # Configuration
    batch_size = 128
    epochs = 20
    latent_dim = 256
    temperature = 1.0
    lr = 0.001
    data_root = './data/cifar10'
    save_dir = f'outputs/training/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Architecture parameters
    encoder_kernel_sizes = [5, 5, 5]  # int or list [k1, k2, k3] for each TaxonConv layer
    decoder_kernel_sizes = [6, 6, 5]  # None = default [4, 4, 3], or list [k1, k2, k3]
    encoder_strides = [2, 2]  # int or list [s1, s2] for downsampling between layers
    decoder_strides = [2, 2, 1]  # None = default [2, 2, 1], or list [s1, s2, s3]
    use_maxpool = True  # True = max pooling, False = average pooling
    
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
        use_maxpool=use_maxpool
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
        save_dir=save_dir
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
