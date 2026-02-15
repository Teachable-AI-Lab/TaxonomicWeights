import torch
import random
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path

class TargetRemappedSubset(Dataset):
    """
    Wraps a Subset and remaps its targets via a given dict.
    """
    def __init__(self, subset: Subset, class_map: dict):
        self.subset = subset
        self.class_map = class_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, orig_t = self.subset[idx]
        new_t = self.class_map[orig_t]
        return img, new_t


class CIFAR10Loader:
    """
    DataLoader wrapper for CIFAR-10 dataset.
    """
    def __init__(self, batch_size=128, root='./data'):
        self.batch_size = batch_size
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.trainset = datasets.CIFAR10(root=self.root, train=True, download=True, transform=self.transform)
        self.testset = datasets.CIFAR10(root=self.root, train=False, download=True, transform=self.transform)
        
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
    
    def get_loaders(self):
        """Returns train and test loaders."""
        return self.train_loader, self.test_loader


class FashionMNISTLoader:
    """
    DataLoader wrapper for Fashion MNIST dataset.
    """
    def __init__(self, batch_size=64, root='./data'):
        self.batch_size = batch_size
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.trainset = datasets.FashionMNIST(root=self.root, train=True, download=True, transform=self.transform)
        self.testset = datasets.FashionMNIST(root=self.root, train=False, download=True, transform=self.transform)
        
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
    
    def get_loaders(self):
        """Returns train and test loaders."""
        return self.train_loader, self.test_loader


class ImageNetLoader:
    """
    DataLoader wrapper for ImageNet-1K dataset.
    """
    def __init__(self, data_dir, batch_size=256, num_workers=8):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Normalization constants for ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        # Training transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Load datasets
        self.train_dataset = datasets.ImageNet(root=self.data_dir, split='train', transform=self.train_transform)
        self.val_dataset = datasets.ImageNet(root=self.data_dir, split='val', transform=self.val_transform)
        
        # Create loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_loaders(self):
        """Returns train and validation loaders."""
        return self.train_loader, self.val_loader

    def make_random_subset(
        self,
        num_classes: int = 10,
        images_per_class: int = 10000,
        seed: int = 101
    ):
        """
        Creates a random subset of ImageNet with specified number of classes and images per class.
        
        Args:
            num_classes: Number of classes to sample
            images_per_class: Number of images per class
            seed: Random seed for reproducibility
        
        Returns:
            remapped_subset: A Dataset of size num_classes*images_per_class
                            whose targets run 0 .. num_classes-1
            selected_classes: List of original class-indices (len=num_classes)
        """
        random.seed(seed)
        
        # Build mapping: original class_idx -> list of sample-indices
        class_to_indices = {}
        for idx, (_, cls) in enumerate(self.train_dataset.samples):
            class_to_indices.setdefault(cls, []).append(idx)
        
        # Choose classes
        all_classes = list(class_to_indices.keys())
        selected_classes = random.sample(all_classes, num_classes)
        
        # Make a mapping from original -> new index
        class_map = {orig: new for new, orig in enumerate(selected_classes)}
        
        # Sample indices (with replacement if needed)
        subset_inds = []
        for orig in selected_classes:
            inds = class_to_indices[orig]
            if len(inds) >= images_per_class:
                chosen = random.sample(inds, images_per_class)
            else:
                chosen = inds[:] + random.choices(inds, k=images_per_class - len(inds))
            subset_inds.extend(chosen)
        
        random.shuffle(subset_inds)
        
        base_subset = Subset(self.train_dataset, subset_inds)
        remapped = TargetRemappedSubset(base_subset, class_map)
        return remapped, selected_classes

    def create_subset_loader(
        self,
        num_classes: int = 10,
        images_per_class: int = 1000,
        seed: int = 5454,
        batch_size: int = None
    ):
        """
        Creates a DataLoader for a random subset of ImageNet.
        
        Args:
            num_classes: Number of classes to sample
            images_per_class: Number of images per class
            seed: Random seed for reproducibility
            batch_size: Batch size for the loader (uses default if None)
        
        Returns:
            subset_loader: DataLoader for the subset
            selected_classes: List of original class-indices
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        subset, selected_classes = self.make_random_subset(
            num_classes=num_classes,
            images_per_class=images_per_class,
            seed=seed
        )
        
        subset_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return subset_loader, selected_classes


class CelebAHQLoader:
    """
    DataLoader wrapper for CelebA-HQ stored in an ImageFolder-style directory.

    Directory layout options:
    1) data_root/train and data_root/val (preferred)
    2) data_root only: a random val_split is carved out.
    3) If data_root doesn't exist: Falls back to datasets.CelebA (30k subset)
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 256,
        val_split: float = 0.05,
        seed: int = 42,
        pin_memory: bool = True,
        train_subset: int = None,
        val_subset: int = None,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.val_subset = val_subset

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Scales to [0, 1]
        ])

        train_dir = self.data_root / "train"
        val_dir = self.data_root / "val"

        # Check if data exists at data_root
        data_exists = False
        if train_dir.exists() and any(train_dir.iterdir()):
            data_exists = True
        elif self.data_root.exists() and any(self.data_root.iterdir()):
            # Check if it's a valid ImageFolder structure
            try:
                test_dataset = datasets.ImageFolder(root=str(self.data_root), transform=None)
                if len(test_dataset) > 0:
                    data_exists = True
            except:
                data_exists = False

        if not data_exists:
            # Fallback to CelebA dataset
            print("=" * 70)
            print("WARNING: CelebA-HQ data not found at:", self.data_root)
            print("Falling back to torchvision's CelebA dataset (30k subset)")
            print("This is a temporary solution - please obtain CelebA-HQ for full quality")
            print("=" * 70)
            
            # Download and use CelebA
            celeba_root = self.data_root.parent / "celeba_fallback"
            celeba_root.mkdir(parents=True, exist_ok=True)
            
            full_celeba = datasets.CelebA(
                root=str(celeba_root),
                split='train',
                transform=self.transform,
                download=True
            )
            
            # Create a 30k subset
            subset_size = min(30000, len(full_celeba))
            indices = list(range(subset_size))
            random.seed(seed)
            random.shuffle(indices)
            
            full_subset = Subset(full_celeba, indices)
            
            if val_split > 0:
                val_size = int(subset_size * val_split)
                train_size = subset_size - val_size
                self.trainset, self.valset = torch.utils.data.random_split(
                    full_subset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(seed)
                )
            else:
                self.trainset = full_subset
                self.valset = None
            
        elif train_dir.exists():
            self.trainset = datasets.ImageFolder(root=str(train_dir), transform=self.transform)
            if val_dir.exists():
                self.valset = datasets.ImageFolder(root=str(val_dir), transform=self.transform)
            else:
                self.valset = None
        else:
            # Single directory: split into train/val
            full_dataset = datasets.ImageFolder(root=str(self.data_root), transform=self.transform)
            if val_split > 0:
                val_size = int(len(full_dataset) * val_split)
                train_size = len(full_dataset) - val_size
                self.trainset, self.valset = torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(seed)
                )
            else:
                self.trainset = full_dataset
                self.valset = None
        
        # Apply subsets if specified
        if self.train_subset is not None and self.train_subset < len(self.trainset):
            indices = list(range(len(self.trainset)))
            random.seed(seed)
            random.shuffle(indices)
            self.trainset = Subset(self.trainset, indices[:self.train_subset])
            print(f"Using train subset: {self.train_subset} samples")
        
        if self.valset is not None and self.val_subset is not None and self.val_subset < len(self.valset):
            indices = list(range(len(self.valset)))
            random.seed(seed)
            random.shuffle(indices)
            self.valset = Subset(self.valset, indices[:self.val_subset])
            print(f"Using val subset: {self.val_subset} samples")

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

        self.val_loader = None
        if self.valset is not None:
            self.val_loader = DataLoader(
                self.valset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )

    def get_loaders(self):
        return self.train_loader, self.val_loader
