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
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_subset = train_subset
        self.val_subset = val_subset

        # Use caller-supplied transform if given, otherwise sensible default.
        if transform is not None:
            self.transform = transform
        else:
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


# ---------------------------------------------------------------------------
# LLM Activation Dataset — for training linear SAEs
# ---------------------------------------------------------------------------

class LLMActivationDataset(Dataset):
    """
    Dataset of LLM residual-stream activations extracted by hooking into a
    transformer model.  Supports two modes:

    1. **Pre-extracted** (recommended for large-scale training):
       Provide ``activation_dir`` pointing to a directory of ``.pt`` files,
       each containing a ``[N, d_model]`` tensor of activations.

    2. **On-the-fly extraction** (convenient for small-scale / debugging):
       Provide ``model_name``, ``layer``, ``hook_point``, and
       ``tokenizer_name``.  Activations are collected by running text
       through the model and hooking the specified point.

    Each ``__getitem__`` returns ``(activation_vector, 0)`` — the second
    element is a dummy label for API compatibility with image loaders.
    """

    def __init__(
        self,
        activation_dir: str = None,
        model_name: str = "gpt2",
        layer: int = 6,
        hook_point: str = "resid_post",
        tokenizer_name: str = None,
        text_dataset: str = "openwebtext",
        text_dataset_split: str = "train",
        max_tokens: int = 1_000_000,
        context_length: int = 128,
        d_model: int = None,
        device: str = "cpu",
        seed: int = 42,
        cache_dir: str = None,
    ):
        """
        Args:
            activation_dir: Path to pre-extracted activations (``.pt`` files).
                            If provided, all other model/text args are ignored.
            model_name:     HuggingFace model id for on-the-fly extraction.
            layer:          Transformer layer index to hook.
            hook_point:     Hook point type (``"resid_post"``, ``"resid_pre"``,
                            ``"mlp_out"``, ``"attn_out"``).
            tokenizer_name: Tokenizer id (defaults to ``model_name``).
            text_dataset:   HuggingFace dataset id for text data.
            text_dataset_split: Split to use.
            max_tokens:     Max tokens to extract activations from.
            context_length: Sequence length per forward pass.
            d_model:        Model hidden dimension (auto-detected if None).
            device:         Device for model inference during extraction.
            seed:           Random seed for text shuffling.
            cache_dir:      Directory to cache extracted activations. Defaults
                            to ``./cache/activations``. Set to ``""`` to disable.
        """
        super().__init__()
        self.seed = seed

        if activation_dir is not None:
            self._load_preextracted(activation_dir)
        else:
            # Build cache path from extraction parameters
            resolved_tokenizer = tokenizer_name or model_name
            cache_hit = False
            if cache_dir != "":
                cache_path = self._cache_path(
                    cache_dir=cache_dir,
                    model_name=model_name,
                    layer=layer,
                    hook_point=hook_point,
                    text_dataset=text_dataset,
                    text_dataset_split=text_dataset_split,
                    max_tokens=max_tokens,
                    context_length=context_length,
                )
                if cache_path.exists():
                    print(f"Loading cached activations from {cache_path}")
                    self.activations = torch.load(
                        str(cache_path), map_location="cpu", weights_only=True
                    )
                    self.d_model = self.activations.shape[1]
                    print(f"Loaded {len(self.activations)} cached activation vectors "
                          f"(d_model={self.d_model})")
                    cache_hit = True

            if not cache_hit:
                self._extract_activations(
                    model_name=model_name,
                    layer=layer,
                    hook_point=hook_point,
                    tokenizer_name=resolved_tokenizer,
                    text_dataset=text_dataset,
                    text_dataset_split=text_dataset_split,
                    max_tokens=max_tokens,
                    context_length=context_length,
                    d_model=d_model,
                    device=device,
                )
                # Save to cache
                if cache_dir != "":
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.activations, str(cache_path))
                    print(f"Cached {len(self.activations)} activation vectors to {cache_path}")

    @staticmethod
    def _cache_path(
        cache_dir: str | None,
        model_name: str,
        layer: int,
        hook_point: str,
        text_dataset: str,
        text_dataset_split: str,
        max_tokens: int,
        context_length: int,
    ) -> Path:
        """Deterministic cache file path based on extraction parameters."""
        base = Path(cache_dir) if cache_dir else Path("./cache/activations")
        safe_model = model_name.replace("/", "_")
        safe_dataset = text_dataset.replace("/", "_")
        filename = (
            f"{safe_model}_L{layer}_{hook_point}"
            f"_{safe_dataset}_{text_dataset_split}"
            f"_tok{max_tokens}_ctx{context_length}.pt"
        )
        return base / filename

    def _load_preextracted(self, activation_dir: str) -> None:
        """Load pre-extracted activation tensors from a directory of .pt files."""
        act_dir = Path(activation_dir)
        if not act_dir.exists():
            raise FileNotFoundError(f"Activation directory not found: {act_dir}")

        pt_files = sorted(act_dir.glob("*.pt"))
        if not pt_files:
            # Try .npy files as fallback
            npy_files = sorted(act_dir.glob("*.npy"))
            if npy_files:
                chunks = [torch.from_numpy(np.load(str(f))).float() for f in npy_files]
            else:
                raise FileNotFoundError(
                    f"No .pt or .npy files found in {act_dir}"
                )
        else:
            chunks = [torch.load(str(f), map_location="cpu", weights_only=True) for f in pt_files]

        self.activations = torch.cat(chunks, dim=0)  # [N, d_model]
        if self.activations.ndim == 3:
            # [N, seq_len, d_model] → flatten to [N*seq_len, d_model]
            self.activations = self.activations.view(-1, self.activations.shape[-1])
        self.d_model = self.activations.shape[1]
        print(f"Loaded {len(self.activations)} activation vectors "
              f"(d_model={self.d_model}) from {activation_dir}")

    @torch.no_grad()
    def _extract_activations(
        self,
        model_name: str,
        layer: int,
        hook_point: str,
        tokenizer_name: str,
        text_dataset: str,
        text_dataset_split: str,
        max_tokens: int,
        context_length: int,
        d_model: int,
        device: str,
    ) -> None:
        """Extract activations on-the-fly from a HuggingFace model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "On-the-fly activation extraction requires "
                "`pip install transformers datasets`."
            )

        print(f"Loading model {model_name} for activation extraction...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device).eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Auto-detect d_model
        if d_model is None:
            d_model = model.config.hidden_size
        self.d_model = d_model

        # Load text data
        print(f"Loading text dataset {text_dataset}...")
        ds = load_dataset(text_dataset, split=text_dataset_split, streaming=True)

        # Hook setup
        collected = []
        hook_handle = None

        def _hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            collected.append(act.detach().cpu())

        # Attach hook to the right layer
        target_layer = model.transformer.h[layer] if hasattr(model, 'transformer') else model.model.layers[layer]
        if hook_point == "resid_post":
            hook_handle = target_layer.register_forward_hook(_hook_fn)
        elif hook_point == "mlp_out":
            hook_handle = target_layer.mlp.register_forward_hook(_hook_fn)
        elif hook_point == "attn_out":
            hook_handle = target_layer.attn.register_forward_hook(_hook_fn)
        else:
            hook_handle = target_layer.register_forward_hook(_hook_fn)

        total_tokens = 0
        token_buffer = []

        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            while len(token_buffer) >= context_length:
                chunk = token_buffer[:context_length]
                token_buffer = token_buffer[context_length:]

                input_ids = torch.tensor([chunk], device=device)
                model(input_ids)

                total_tokens += context_length
                if total_tokens >= max_tokens:
                    break

            if total_tokens >= max_tokens:
                break

        hook_handle.remove()
        del model

        if not collected:
            raise RuntimeError("No activations collected. Check model/layer/hook_point.")

        self.activations = torch.cat(collected, dim=0).view(-1, d_model)
        print(f"Extracted {len(self.activations)} activation vectors "
              f"(d_model={self.d_model}) from {model_name} layer {layer}")

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], 0  # dummy label for API compat


class LLMActivationLoader:
    """
    DataLoader wrapper for LLM activation data.

    Mirrors the API of :class:`CelebAHQLoader` — call ``.get_loaders()`` to get
    ``(train_loader, val_loader)``.

    Args:
        activation_dir: Path to pre-extracted activations.
        model_name:     HuggingFace model for on-the-fly extraction.
        layer:          Transformer layer to hook.
        hook_point:     Hook type (``"resid_post"`` etc.).
        tokenizer_name: HuggingFace tokenizer id.
        text_dataset:   HuggingFace dataset id.
        text_dataset_split: Dataset split to use.
        max_tokens:     Maximum tokens to extract.
        context_length: Tokens per forward pass.
        d_model:        Model hidden dim (auto-detected if None).
        batch_size:     Batch size for loaders.
        num_workers:    DataLoader workers.
        val_split:      Fraction held out for validation.
        seed:           Random seed.
        pin_memory:     Pin memory for CUDA.
        device:         Device for on-the-fly extraction.
        cache_dir:      Directory to cache extracted activations.
    """

    def __init__(
        self,
        activation_dir: str = None,
        model_name: str = "gpt2",
        layer: int = 6,
        hook_point: str = "resid_post",
        tokenizer_name: str = None,
        text_dataset: str = "openwebtext",
        text_dataset_split: str = "train",
        max_tokens: int = 1_000_000,
        context_length: int = 128,
        d_model: int = None,
        batch_size: int = 4096,
        num_workers: int = 4,
        val_split: float = 0.05,
        seed: int = 42,
        pin_memory: bool = True,
        device: str = "cpu",
        cache_dir: str = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        dataset = LLMActivationDataset(
            activation_dir=activation_dir,
            model_name=model_name,
            layer=layer,
            hook_point=hook_point,
            tokenizer_name=tokenizer_name,
            text_dataset=text_dataset,
            text_dataset_split=text_dataset_split,
            max_tokens=max_tokens,
            context_length=context_length,
            d_model=d_model,
            device=device,
            seed=seed,
            cache_dir=cache_dir,
        )

        self.d_model = dataset.d_model

        # Split into train/val
        n = len(dataset)
        if val_split > 0:
            val_size = int(n * val_split)
            train_size = n - val_size
            self.trainset, self.valset = torch.utils.data.random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            self.trainset = dataset
            self.valset = None

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
