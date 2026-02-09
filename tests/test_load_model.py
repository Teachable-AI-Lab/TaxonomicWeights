"""
Test script to load and verify CIFAR-10 autoencoder model
Tests both taxonomic and regular layer configurations
"""

import os
import sys
import json
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.taxon_ae import CIFAR10TaxonAutoencoder


def parse_layer_config(config):
    """Parse layer-by-layer config format."""
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
                k = layer['kernel_size']
                decoder_paddings.append(k // 2)
        
        # Auto-infer decoder output_paddings if not specified
        decoder_output_paddings = []
        for layer in decoder_layers:
            if 'output_padding' in layer:
                decoder_output_paddings.append(layer['output_padding'])
            else:
                decoder_output_paddings.append(1 if layer['stride'] > 1 else 0)
    else:
        raise ValueError("Only layer-by-layer format supported in this test")
    
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


def test_model_loading(config_path):
    """Load and test a model from config."""
    print("=" * 60)
    print(f"Testing model from: {config_path}")
    print("=" * 60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\nConfiguration: {config['description']}")
    
    # Parse parameters
    latent_dim = config['model']['latent_dim']
    temperature = config['model']['temperature']
    use_maxpool = config['model']['use_maxpool']
    
    layer_params = parse_layer_config(config)
    
    print(f"\nEncoder layers:")
    for i, layer_type in enumerate(layer_params['encoder_layer_types']):
        print(f"  Layer {i+1}: type={layer_type}", end="")
        if layer_type == 'taxonomic':
            print(f", n_layers={layer_params['encoder_n_layers'][i]}", end="")
        else:
            print(f", n_filters={layer_params['encoder_n_filters'][i]}", end="")
        print(f", kernel={layer_params['encoder_kernel_sizes'][i]}, stride={layer_params['encoder_strides'][i]}")
    
    print(f"\nDecoder layers:")
    for i, layer_type in enumerate(layer_params['decoder_layer_types']):
        print(f"  Layer {i+1}: type={layer_type}", end="")
        if layer_type == 'taxonomic':
            print(f", n_layers={layer_params['decoder_n_layers'][i]}", end="")
        else:
            print(f", n_filters={layer_params['decoder_n_filters'][i]}", end="")
        print(f", kernel={layer_params['decoder_kernel_sizes'][i]}, stride={layer_params['decoder_strides'][i]}")
    
    # Create model
    print(f"\nCreating model...")
    model = CIFAR10TaxonAutoencoder(
        latent_dim=latent_dim,
        temperature=temperature,
        encoder_kernel_sizes=layer_params['encoder_kernel_sizes'],
        decoder_kernel_sizes=layer_params['decoder_kernel_sizes'],
        encoder_strides=layer_params['encoder_strides'],
        decoder_strides=layer_params['decoder_strides'],
        encoder_n_layers=layer_params['encoder_n_layers'],
        decoder_n_layers=layer_params['decoder_n_layers'],
        encoder_n_filters=layer_params['encoder_n_filters'],
        decoder_n_filters=layer_params['decoder_n_filters'],
        encoder_layer_types=layer_params['encoder_layer_types'],
        decoder_layer_types=layer_params['decoder_layer_types'],
        decoder_paddings=layer_params['decoder_paddings'],
        decoder_output_paddings=layer_params['decoder_output_paddings'],
        use_maxpool=use_maxpool
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    
    with torch.no_grad():
        # Test encoder
        latent = model.encode(dummy_input)
        print(f"  Encoder output shape: {latent.shape}")
        
        # Verify encoder output is spatial (4D tensor: B, C, H, W)
        assert latent.ndim == 4, f"Encoder should output 4D tensor, got {latent.ndim}D"
        
        # Test decoder
        reconstructed = model.decode(latent)
        print(f"  Decoder output shape: {reconstructed.shape} (expected: [{batch_size}, 3, 32, 32])")
        
        # Test full forward
        output = model(dummy_input)
        print(f"  Full model output shape: {output.shape} (expected: [{batch_size}, 3, 32, 32])")
    
    # Verify shapes
    assert latent.shape[0] == batch_size, f"Batch size mismatch: {latent.shape[0]}"
    assert output.shape == (batch_size, 3, 32, 32), f"Output shape mismatch: {output.shape}"
    
    print(f"\n✓ Model loaded and tested successfully!")
    print("=" * 60)
    return model


if __name__ == "__main__":
    # Test both configs
    configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    
    # Test taxonomic config
    taxonomic_config = os.path.join(configs_dir, 'cifar10_standard.json')
    if os.path.exists(taxonomic_config):
        test_model_loading(taxonomic_config)
        print("\n")
    
    # Test regular config
    regular_config = os.path.join(configs_dir, 'cifar10_standard_regular.json')
    if os.path.exists(regular_config):
        test_model_loading(regular_config)
    else:
        print(f"Config not found: {regular_config}")
