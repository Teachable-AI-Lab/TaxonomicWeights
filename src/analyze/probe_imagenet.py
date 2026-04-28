#!/usr/bin/env python3
"""probe_imagenet.py — Linear and KNN probing of Tiny-ImageNet model latents.

Mirror of probe_celeba_hq.py, adapted for 200-class integer labels rather than
40 binary CelebA attributes.

Metrics
-------
* Linear probe: top-1 and top-5 accuracy (multiclass LogisticRegression)
* KNN: top-1 and top-5 accuracy
* Sparsity analysis: L0, dead-feature fraction, selectivity, Jaccard similarity
* Taxon vs. non-taxon sparsity-matched comparisons

Outputs (in ``--save-dir``)
---------------------------
* probe_imagenet_results.csv
* probe_imagenet_linear.png
* probe_imagenet_knn.png
* probe_imagenet_sparsity.png
* probe_imagenet_taxon_scatter.png
* probe_imagenet_taxon_matched.png
* probe_imagenet_taxon_class_compare.png  — per-class accuracy heatmap, matched pairs
* probe_imagenet_class_detail.png         — per-class accuracy heatmap across all models
* probe_imagenet_monosemanticity.png      — L0 vs monosemanticity score scatter
* tsne/probe_imagenet_tsne_<name>.png     — per-model t-SNE grid (binary per-class panels)
* tsne/probe_imagenet_tsne_compare.png    — cross-model t-SNE for headline classes
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.compare.compare_imagenet import (
    load_model,
    discover_runs,
    _model_type,
    _short_name,
)
from src.utils.dataloader import TinyImageNetLoader


# ─── sklearn lazy imports ─────────────────────────────────────────────────────

def _import_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    return LogisticRegression, KNeighborsClassifier, StandardScaler, accuracy_score


# ─── Tiny-ImageNet class names ────────────────────────────────────────────────

# Ordered synset IDs matching label indices 0..199 in zh-plus/tiny-imagenet.
_TINY_IMAGENET_WNIDS: List[str] = [
    "n01443537", "n01629819", "n01641577", "n01644900", "n01698640",
    "n01742172", "n01768244", "n01770393", "n01774384", "n01774750",
    "n01784675", "n01882714", "n01910747", "n01917289", "n01944390",
    "n01950731", "n01983481", "n01984695", "n02002724", "n02056570",
    "n02058221", "n02074367", "n02094433", "n02099601", "n02099712",
    "n02106662", "n02113799", "n02123045", "n02123394", "n02124075",
    "n02125311", "n02129165", "n02132136", "n02165456", "n02226429",
    "n02231487", "n02233338", "n02236044", "n02268443", "n02279972",
    "n02281406", "n02321529", "n02364673", "n02395406", "n02403003",
    "n02410509", "n02415577", "n02423022", "n02437312", "n02480495",
    "n02481823", "n02486410", "n02504458", "n02509815", "n02666347",
    "n02669723", "n02699494", "n02769748", "n02788148", "n02791270",
    "n02793495", "n02795169", "n02802426", "n02808440", "n02814533",
    "n02814860", "n02815834", "n02823428", "n02837789", "n02841315",
    "n02843684", "n02883205", "n02892201", "n02909870", "n02917067",
    "n02927161", "n02948072", "n02950826", "n02963159", "n02977058",
    "n02988304", "n03014705", "n03026506", "n03042490", "n03085013",
    "n03089624", "n03100240", "n03126707", "n03160309", "n03179701",
    "n03201208", "n03255030", "n03355925", "n03373237", "n03388043",
    "n03393912", "n03400231", "n03404251", "n03424325", "n03444034",
    "n03447447", "n03544143", "n03584254", "n03599486", "n03617480",
    "n03637318", "n03649909", "n03662601", "n03670208", "n03706229",
    "n03733131", "n03763968", "n03770439", "n03796401", "n03814639",
    "n03837869", "n03838899", "n03854065", "n03891332", "n03902125",
    "n03930313", "n03937543", "n03970156", "n03977966", "n03980874",
    "n03983396", "n03992509", "n04008634", "n04023962", "n04070727",
    "n04074963", "n04099969", "n04118538", "n04133789", "n04146614",
    "n04149813", "n04179913", "n04251144", "n04254777", "n04259630",
    "n04265275", "n04275548", "n04285008", "n04311004", "n04328186",
    "n04356056", "n04366367", "n04371430", "n04376876", "n04398044",
    "n04399382", "n04417672", "n04456115", "n04465666", "n04486054",
    "n04487081", "n04501370", "n04507155", "n04532106", "n04532670",
    "n04540053", "n04560804", "n04562935", "n04596742", "n04598010",
    "n06596364", "n07056680", "n07583066", "n07614500", "n07615774",
    "n07646821", "n07647870", "n07657664", "n07695742", "n07711569",
    "n07715103", "n07720875", "n07749582", "n07753592", "n07768694",
    "n07871810", "n07873807", "n07875152", "n07920052", "n07975909",
    "n08496334", "n08620881", "n08742578", "n09193705", "n09246464",
    "n09256479", "n09332890", "n09428293", "n12267677", "n12520864",
    "n13001041", "n13652335", "n13652994", "n13719102", "n14991210",
]

# Short human-readable names for each synset (from standard Tiny-ImageNet words.txt).
_WNID_SHORT_NAMES: Dict[str, str] = {
    "n01443537": "goldfish",        "n01629819": "fire salamander",
    "n01641577": "bullfrog",         "n01644900": "tailed frog",
    "n01698640": "alligator",        "n01742172": "boa constrictor",
    "n01768244": "trilobite",        "n01770393": "scorpion",
    "n01774384": "garden spider",    "n01774750": "black widow",
    "n01784675": "tarantula",        "n01882714": "koala",
    "n01910747": "jellyfish",        "n01917289": "brain coral",
    "n01944390": "snail",            "n01950731": "slug",
    "n01983481": "lobster",          "n01984695": "spiny lobster",
    "n02002724": "black stork",      "n02056570": "king penguin",
    "n02058221": "albatross",        "n02074367": "dugong",
    "n02094433": "Yorkshire terrier","n02099601": "golden retriever",
    "n02099712": "Labrador retriever","n02106662": "German shepherd",
    "n02113799": "standard poodle", "n02123045": "tabby cat",
    "n02123394": "Persian cat",      "n02124075": "Egyptian cat",
    "n02125311": "cougar",           "n02129165": "lion",
    "n02132136": "brown bear",       "n02165456": "ladybug",
    "n02226429": "grasshopper",      "n02231487": "walking stick",
    "n02233338": "cockroach",        "n02236044": "mantis",
    "n02268443": "dragonfly",        "n02279972": "monarch butterfly",
    "n02281406": "sulphur butterfly","n02321529": "sea cucumber",
    "n02364673": "guinea pig",       "n02395406": "hog",
    "n02403003": "ox",               "n02410509": "bison",
    "n02415577": "bighorn sheep",    "n02423022": "gazelle",
    "n02437312": "Arabian camel",    "n02480495": "orangutan",
    "n02481823": "chimpanzee",       "n02486410": "baboon",
    "n02504458": "elephant",         "n02509815": "red panda",
    "n02666347": "abacus",           "n02669723": "academic gown",
    "n02699494": "altar",            "n02769748": "backpack",
    "n02788148": "bannister",        "n02791270": "barbershop",
    "n02793495": "barn",             "n02795169": "barrel",
    "n02802426": "basketball",       "n02808440": "bathtub",
    "n02814533": "beach wagon",      "n02814860": "beacon",
    "n02815834": "beaker",           "n02823428": "beer bottle",
    "n02837789": "bikini",           "n02841315": "binoculars",
    "n02843684": "birdhouse",        "n02883205": "bow tie",
    "n02892201": "brass plaque",     "n02909870": "broom",
    "n02917067": "bullet train",     "n02927161": "butcher shop",
    "n02948072": "candle",           "n02950826": "cannon",
    "n02963159": "cardigan",         "n02977058": "ATM",
    "n02988304": "CD player",        "n03014705": "chest",
    "n03026506": "Xmas stocking",    "n03042490": "cliff dwelling",
    "n03085013": "keyboard",         "n03089624": "candy store",
    "n03100240": "convertible",      "n03126707": "crane",
    "n03160309": "dam",              "n03179701": "desk",
    "n03201208": "dining table",     "n03255030": "dumbbell",
    "n03355925": "flagpole",         "n03373237": "fly",
    "n03388043": "freight car",      "n03393912": "frying pan",
    "n03400231": "fur coat",         "n03404251": "garbage truck",
    "n03424325": "go-kart",          "n03444034": "gondola",
    "n03447447": "grille",           "n03544143": "hourglass",
    "n03584254": "iPod",             "n03599486": "iron",
    "n03617480": "jeans",            "n03637318": "lamp",
    "n03649909": "laptop",           "n03662601": "lemon",
    "n03670208": "lifeboat",         "n03706229": "compass",
    "n03733131": "maypole",          "n03763968": "military uniform",
    "n03770439": "miniskirt",        "n03796401": "moving van",
    "n03814639": "mushroom",         "n03837869": "nail",
    "n03838899": "neck brace",       "n03854065": "obelisk",
    "n03891332": "orange",           "n03902125": "organ",
    "n03930313": "picket fence",     "n03937543": "pill bottle",
    "n03970156": "plunger",          "n03977966": "police van",
    "n03980874": "poncho",           "n03983396": "pool table",
    "n03992509": "pot",              "n04008634": "projectile",
    "n04023962": "punching bag",     "n04070727": "refrigerator",
    "n04074963": "remote control",   "n04099969": "rocking chair",
    "n04118538": "rubber eraser",    "n04133789": "running shoe",
    "n04146614": "school bus",       "n04149813": "scoreboard",
    "n04179913": "sewing machine",   "n04251144": "snorkel",
    "n04254777": "sock",             "n04259630": "sombrero",
    "n04265275": "space heater",     "n04275548": "spider web",
    "n04285008": "sports car",       "n04311004": "steel arch bridge",
    "n04328186": "stopwatch",        "n04356056": "sunglasses",
    "n04366367": "suspension bridge","n04371430": "swimming trunks",
    "n04376876": "syringe",          "n04398044": "table lamp",
    "n04399382": "tank",             "n04417672": "teddy bear",
    "n04456115": "teapot",           "n04465666": "toaster",
    "n04486054": "tractor",          "n04487081": "trailer truck",
    "n04501370": "triumphal arch",   "n04507155": "trolleybus",
    "n04532106": "umbrella",         "n04532670": "upright piano",
    "n04540053": "vase",             "n04560804": "volcano",
    "n04562935": "volleyball",       "n04596742": "water jug",
    "n04598010": "water tower",      "n06596364": "comic book",
    "n07056680": "pretzel",          "n07583066": "guacamole",
    "n07614500": "ice cream",        "n07615774": "ice lolly",
    "n07646821": "orange",           "n07647870": "lemon",
    "n07657664": "fig",              "n07695742": "pretzel",
    "n07711569": "mashed potato",    "n07715103": "cauliflower",
    "n07720875": "bell pepper",      "n07749582": "lemon",
    "n07753592": "banana",           "n07768694": "pineapple",
    "n07871810": "meat loaf",        "n07873807": "pizza",
    "n07875152": "potpie",           "n07920052": "espresso",
    "n07975909": "mushroom",         "n08496334": "cliff",
    "n08620881": "valley",           "n08742578": "alp",
    "n09193705": "mountain",         "n09246464": "cliff",
    "n09256479": "coral reef",       "n09332890": "lakeside",
    "n09428293": "seashore",         "n12267677": "daisy",
    "n12520864": "ear of corn",      "n13001041": "bolete",
    "n13652335": "bolete",           "n13652994": "ear",
    "n13719102": "coral fungus",     "n14991210": "stone wall",
}


def _build_class_names(data_root: str) -> List[str]:
    """Return 200 readable class names for Tiny-ImageNet, indexed by label 0..199.

    Tries to confirm synset order from the cached dataset_info.json; falls back
    to the hardcoded ``_TINY_IMAGENET_WNIDS`` list if the file is unavailable.
    """
    import json as _json
    import glob as _glob
    wnids: Optional[List[str]] = None
    try:
        pattern = str(
            Path(data_root) / "zh-plus___tiny-imagenet" / "**" / "dataset_info.json"
        )
        matches = _glob.glob(pattern, recursive=True)
        if matches:
            with open(matches[0]) as _f:
                _info = _json.load(_f)
            wnids = _info["features"]["label"]["names"]
    except Exception:
        pass
    if not wnids or len(wnids) != 200:
        wnids = _TINY_IMAGENET_WNIDS
    return [_WNID_SHORT_NAMES.get(w, w) for w in wnids]


# ─── data loading ─────────────────────────────────────────────────────────────

def _load_tiny_imagenet_val(
    cache_dir: str,
    image_size: int,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
):
    """Return a DataLoader over the Tiny-ImageNet validation split.

    Yields ``(img_tensor, class_label_int)`` tuples where class labels are
    integers in [0, 199].

    If *max_samples* is set, a subset of samples is returned via a
    random Subset sampler.
    """
    loader_cls = TinyImageNetLoader(
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size,
        cache_dir=cache_dir,
    )
    _, val_loader = loader_cls.get_loaders()

    if max_samples is not None and max_samples < len(loader_cls.valset):
        from torch.utils.data import Subset, DataLoader
        rng = np.random.RandomState(42)
        indices = rng.choice(len(loader_cls.valset), max_samples, replace=False).tolist()
        subset = Subset(loader_cls.valset, indices)
        val_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    return val_loader


# ─── latent extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(
    model: torch.nn.Module,
    run_type: str,
    loader,
    device: torch.device,
    max_batches: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract encoder latents and class labels from *loader*.

    Returns
    -------
    latents : np.ndarray, shape (N, C)  — pooled encoder activations
    labels  : np.ndarray, shape (N,)    — integer class labels 0–199
    """
    model.eval()
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured["z"] = out.detach()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            first = out[0]
            if isinstance(first, torch.Tensor):
                captured["z"] = first.detach()
            elif isinstance(first, (list, tuple)) and len(first) > 0:
                # intermediate_topk_sae: encoder returns (List[Tensor], info)
                # Use the last (deepest) stage latent as the representation.
                last = first[-1]
                if isinstance(last, torch.Tensor):
                    captured["z"] = last.detach()

    hook_handle = model.encoder.register_forward_hook(_hook)

    all_latents: List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []

    try:
        for batch_idx, (imgs, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)

            try:
                model(imgs)
            except Exception:
                pass  # hook already fired before any error

            if "z" not in captured:
                continue

            z = captured.pop("z")          # (B, C, H, W) or (B, C)
            if z.dim() == 4:
                z = F.adaptive_avg_pool2d(z, 1).flatten(1)   # → (B, C)
            elif z.dim() == 3:
                z = z.mean(dim=1)           # sequence → avg

            all_latents.append(z.cpu().float().numpy())
            lbl = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_labels.append(lbl.astype(np.int64))

    finally:
        hook_handle.remove()

    if not all_latents:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(all_latents, axis=0), np.concatenate(all_labels, axis=0)


# ─── linear probing ───────────────────────────────────────────────────────────

def _top5_accuracy(decision_func: np.ndarray, y_true: np.ndarray) -> float:
    """Top-5 accuracy from decision function scores (N, num_classes)."""
    top5 = np.argsort(decision_func, axis=1)[:, -5:]
    return float(np.mean([y in t5 for y, t5 in zip(y_true, top5)]))


def run_linear_probe(
    latents: np.ndarray,
    labels: np.ndarray,
    n_train_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, object]:
    """Train a multiclass LogisticRegression probe on Tiny-ImageNet labels.

    Returns dict with keys:
      ``top1_accuracy``    : float — mean top-1 accuracy on held-out set
      ``top5_accuracy``    : float — mean top-5 accuracy on held-out set
      ``per_class_accuracy``: (200,) per-class accuracy
    """
    LogisticRegression, _, StandardScaler, accuracy_score = _import_sklearn()

    N = len(latents)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(N * n_train_frac)
    tr, te = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(latents[tr])
    X_te = scaler.transform(latents[te])
    y_tr, y_te = labels[tr], labels[te]

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    dec = clf.decision_function(X_te)

    top1 = float(accuracy_score(y_te, y_pred))
    top5 = _top5_accuracy(dec, y_te)

    # Per-class accuracy
    classes = sorted(set(y_te.tolist()))
    per_class = np.full(200, float("nan"))
    for c in classes:
        mask = y_te == c
        if mask.sum() >= 1:
            per_class[c] = float(accuracy_score(y_te[mask], y_pred[mask]))

    return {
        "top1_accuracy":     top1,
        "top5_accuracy":     top5,
        "per_class_accuracy": per_class,
    }


# ─── KNN classification ───────────────────────────────────────────────────────

def run_knn(
    latents: np.ndarray,
    labels: np.ndarray,
    k_values: Tuple[int, ...] = (1, 5, 20),
    n_train_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    """KNN probing for Tiny-ImageNet class labels.

    Returns dict with keys ``k{k}_top1`` and ``k{k}_top5`` for each k.
    """
    _, KNeighborsClassifier, StandardScaler, accuracy_score = _import_sklearn()

    N = len(latents)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(N * n_train_frac)
    tr, te = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(latents[tr])
    X_te = scaler.transform(latents[te])
    y_tr, y_te = labels[tr], labels[te]

    results: Dict[str, float] = {}
    for k in k_values:
        # k_neighbors must be at least 5 for top-5
        n_neighbors = max(k, 5)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=4, metric="cosine")
        knn.fit(X_tr, y_tr)

        y_pred = knn.predict(X_te)
        top1 = float(accuracy_score(y_te, y_pred))

        # Top-5: get all n_neighbors predictions per sample
        neigh_labels = knn.kneighbors(X_te, n_neighbors=min(5, n_neighbors), return_distance=False)
        top5_correct = [y in row for y, row in
                        zip(y_te, knn.classes_[neigh_labels])]
        top5 = float(np.mean(top5_correct))

        results[f"k{k}_top1"] = top1
        results[f"k{k}_top5"] = top5

    return results


# ─── sparsity stats ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_sparsity_stats(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 1e-6,
    max_batches: int = 60,
) -> Dict[str, float]:
    """Compute L0 norm, dead feature fraction, and selectivity on the val set."""
    model.eval()
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured["z"] = out.detach()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            first = out[0]
            if isinstance(first, torch.Tensor):
                captured["z"] = first.detach()
            elif isinstance(first, (list, tuple)) and len(first) > 0:
                last = first[-1]
                if isinstance(last, torch.Tensor):
                    captured["z"] = last.detach()

    hook_handle = model.encoder.register_forward_hook(_hook)
    all_acts: List[np.ndarray] = []

    try:
        for batch_idx, (imgs, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            try:
                model(imgs)
            except Exception:
                pass
            if "z" not in captured:
                continue
            z = captured.pop("z")
            if z.dim() == 4:
                z = z.reshape(z.shape[0], -1)
            z = z.abs().cpu().float().numpy()
            all_acts.append(z)
    finally:
        hook_handle.remove()

    if not all_acts:
        return {k: float("nan") for k in
                ("mean_l0", "dead_frac", "l0_abs", "selectivity", "mean_jaccard")}

    acts = np.concatenate(all_acts, axis=0)   # (N, C)
    N, C = acts.shape
    binary = (acts > threshold).astype(np.float32)

    mean_l0 = binary.mean()
    l0_abs  = binary.sum(axis=1).mean()
    dead    = (acts.max(axis=0) <= threshold)
    dead_frac = dead.mean()

    ch_mean = acts.mean(axis=0)
    global_mean = ch_mean.mean() + 1e-8
    selectivity = ch_mean.std() / global_mean

    n_sub = min(512, N)
    rng = np.random.RandomState(0)
    sub_idx = rng.choice(N, n_sub, replace=False)
    B = binary[sub_idx]
    inter = (B @ B.T)
    row_sum = B.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - inter + 1e-8
    jac = inter / union
    mask = ~np.eye(n_sub, dtype=bool)
    mean_jaccard = float(jac[mask].mean())

    return {
        "mean_l0":      float(mean_l0),
        "dead_frac":    float(dead_frac),
        "l0_abs":       float(l0_abs),
        "selectivity":  float(selectivity),
        "mean_jaccard": float(mean_jaccard),
    }


def compute_monosemanticity(
    latents: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 200,
    top_k: int = 50,
    min_active_frac: float = 0.005,
) -> float:
    """Compute a monosemanticity score averaged over active features.

    For each feature *i* that fires on at least ``min_active_frac`` of samples:

    1. Collect the indices of the top-K images most strongly activating feature i.
    2. For each class j compute the *lift*:
       ``lift_ij = freq(class_j | top-K) / (base_freq(class_j) + eps)``
    3. The per-feature score is ``max_j(lift_ij)``.

    The model-level score is the mean over all qualifying features.
    A perfectly monosemantic feature that fires only for class j yields
    lift ≈ n_classes; a random feature yields lift ≈ 1.
    """
    N, C = latents.shape
    k = min(top_k, N)

    # One-hot encode labels → (N, n_classes)
    labels_oh = np.zeros((N, n_classes), dtype=np.float32)
    for i, c in enumerate(labels):
        if 0 <= int(c) < n_classes:
            labels_oh[i, int(c)] = 1.0

    base_freq = labels_oh.mean(axis=0) + 1e-6  # (n_classes,)

    ch_max = latents.max(axis=0)               # (C,)
    thresh = np.quantile(latents[latents > 0], min_active_frac) if (latents > 0).any() else 0.0
    active_mask = ch_max > thresh

    scores = []
    for i in range(C):
        if not active_mask[i]:
            continue
        top_idx  = np.argpartition(latents[:, i], -k)[-k:]
        freq_top = labels_oh[top_idx].mean(axis=0)   # (n_classes,)
        lift     = freq_top / base_freq
        scores.append(float(lift.max()))

    return float(np.mean(scores)) if scores else float("nan")


# ─── plotting ─────────────────────────────────────────────────────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#000080", "#1f3a93", "#3b5998", "#5b7bbf", "#082567",
    "#003f5c", "#2f4b7c", "#1a3a5c", "#bc6c25", "#dda15e",
]


def _bar(ax, vals, labels, colours, title, ylabel, lower_better=False):
    valid = [(v, l, c) for v, l, c in zip(vals, labels, colours)
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    vs, ls, cs = zip(*valid)
    bars = ax.bar(range(len(vs)), vs, color=cs, edgecolor="black", linewidth=0.4)
    best = int(np.argmin(vs) if lower_better else np.argmax(vs))
    bars[best].set_edgecolor("red")
    bars[best].set_linewidth(2.0)
    ax.set_xticks(range(len(ls)))
    ax.set_xticklabels(ls, rotation=40, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_linear_results(all_results, save_dir: Path):
    names   = [r["short"] for r in all_results]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(all_results))]

    top1 = [r["linear"]["top1_accuracy"] if r.get("linear") else float("nan")
            for r in all_results]
    top5 = [r["linear"]["top5_accuracy"] if r.get("linear") else float("nan")
            for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 0.9), 5))
    _bar(axes[0], top1, names, colours, "Top-1 Accuracy (linear probe)", "Accuracy")
    _bar(axes[1], top5, names, colours, "Top-5 Accuracy (linear probe)", "Accuracy")
    plt.suptitle("Linear Probing — Tiny-ImageNet (200 classes)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_imagenet_linear.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_knn_results(all_results, save_dir: Path):
    names   = [r["short"] for r in all_results]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(all_results))]

    all_k_keys = set()
    for r in all_results:
        if r.get("knn"):
            all_k_keys.update(r["knn"].keys())
    top1_keys = sorted([k for k in all_k_keys if k.endswith("_top1")])
    top5_keys = sorted([k for k in all_k_keys if k.endswith("_top5")])

    n_panels = len(top1_keys) + len(top5_keys)
    if n_panels == 0:
        return

    n_cols = max(1, min(n_panels, 4))
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(10, len(names) * 0.9) * n_cols / 2, 4 * n_rows))
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    panel_keys = top1_keys + top5_keys
    for ax, key in zip(axes_flat, panel_keys):
        label_type = "Top-1" if key.endswith("_top1") else "Top-5"
        k = key.split("_")[0]
        vals = [r["knn"][key] if r.get("knn") and key in r["knn"]
                else float("nan") for r in all_results]
        _bar(ax, vals, names, colours, f"KNN {label_type} Acc ({k})", "Accuracy")

    for ax in axes_flat[len(panel_keys):]:
        ax.set_visible(False)

    plt.suptitle("KNN Probing — Tiny-ImageNet", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_imagenet_knn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_sparsity_results(all_results, save_dir: Path):
    names   = [r["short"] for r in all_results]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(all_results))]
    sp      = [r.get("sparsity", {}) for r in all_results]

    fig, axes = plt.subplots(1, 5, figsize=(max(20, len(names) * 1.2), 5))
    _bar(axes[0], [s.get("mean_l0")      for s in sp], names, colours,
         "Mean L0 (frac active)",      "Fraction",    lower_better=False)
    _bar(axes[1], [s.get("dead_frac")    for s in sp], names, colours,
         "Dead Feature Fraction",      "Fraction",    lower_better=True)
    _bar(axes[2], [s.get("l0_abs")       for s in sp], names, colours,
         "Mean Active Channels",       "Count",       lower_better=False)
    _bar(axes[3], [s.get("selectivity")  for s in sp], names, colours,
         "Feature Selectivity Index",  "Selectivity", lower_better=False)
    _bar(axes[4], [s.get("mean_jaccard") for s in sp], names, colours,
         "Mean Jaccard (binarised)",   "Jaccard",     lower_better=True)
    plt.suptitle("Sparse Interpretability — Tiny-ImageNet", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_imagenet_sparsity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_class_heatmap(
    all_results: List[dict],
    save_dir: Path,
    class_names: Optional[List[str]] = None,
):
    """Heatmap: rows = models, columns = 200 Tiny-ImageNet classes, value = per-class accuracy."""
    rows = [r for r in all_results if r.get("linear")]
    if not rows:
        return
    names  = [r["short"] for r in rows]
    matrix = np.stack([r["linear"]["per_class_accuracy"] for r in rows], axis=0)  # (M, 200)
    fig, ax = plt.subplots(figsize=(max(20, 200 * 0.12), max(4, len(rows) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    tick_ids = list(range(0, 200, 10))
    ax.set_xticks(tick_ids)
    if class_names:
        tick_labels = [f"{c}:{class_names[c][:12]}" for c in tick_ids]
    else:
        tick_labels = [str(c) for c in tick_ids]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Class (Tiny-ImageNet 0–199)", fontsize=9)
    ax.set_title("Per-Class Linear Probe Accuracy (rows=models, cols=classes)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    plt.tight_layout()
    fig.savefig(save_dir / "probe_imagenet_class_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: probe_imagenet_class_detail.png")


# ─── taxon vs. non-taxon sparsity-matched comparisons ─────────────────────────

_TAXON_TYPES = {
    "taxon", "multi_taxon",
    "topk_taxon", "topk_multi_taxon",
    "bias_taxon", "bias_multi_taxon",
    "bottleneck_taxon", "bottleneck_multi_taxon",
    "bottleneck_topk_taxon", "bottleneck_topk_multi_taxon",
}

_TAXON_COLOR = "#1f77b4"
_BASE_COLOR  = "#d62728"


def _is_taxon(r: dict) -> bool:
    return r.get("type", "") in _TAXON_TYPES


def _sparsity_l0(r: dict) -> Optional[float]:
    sp = r.get("sparsity", {})
    v = sp.get("l0_abs")
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def _linear_top1(r: dict) -> Optional[float]:
    lin = r.get("linear")
    if lin is None:
        return None
    v = lin.get("top1_accuracy")
    return float(v) if v is not None else None


def find_sparsity_matched_pairs(
    all_results: List[dict],
    max_l0_ratio: float = 3.0,
) -> List[Tuple[dict, dict]]:
    taxon_rs    = [r for r in all_results if _is_taxon(r) and _sparsity_l0(r) is not None]
    baseline_rs = [r for r in all_results if not _is_taxon(r) and _sparsity_l0(r) is not None]

    if not taxon_rs or not baseline_rs:
        return []

    pairs: List[Tuple[dict, dict]] = []
    for tr in taxon_rs:
        l0_t = _sparsity_l0(tr)
        best, best_dist = None, float("inf")
        for br in baseline_rs:
            l0_b = _sparsity_l0(br)
            dist = abs(np.log(l0_t + 1) - np.log(l0_b + 1))
            if dist < best_dist:
                best_dist, best = dist, br
        if best is not None:
            l0_b = _sparsity_l0(best)
            ratio = max(l0_t, l0_b) / (min(l0_t, l0_b) + 1e-8)
            if ratio <= max_l0_ratio:
                pairs.append((tr, best))

    return pairs


def plot_taxon_scatter(all_results: List[dict], save_dir: Path):
    """Scatter: L0 (mean active channels) vs performance/sparsity metrics."""
    taxon_r    = [r for r in all_results if _is_taxon(r)]
    baseline_r = [r for r in all_results if not _is_taxon(r)]

    def _collect(results, metric_fn):
        xs, ys, labels = [], [], []
        for r in results:
            l0 = _sparsity_l0(r)
            y  = metric_fn(r)
            if l0 is not None and y is not None and not np.isnan(float(y)):
                xs.append(l0)
                ys.append(float(y))
                labels.append(r["short"])
        return xs, ys, labels

    metrics = [
        ("Linear Top-1 Accuracy",
         lambda r: _linear_top1(r)),
        ("KNN k=5 Top-1 Acc",
         lambda r: r["knn"].get("k5_top1") if r.get("knn") else None),
        ("Feature Selectivity",
         lambda r: r.get("sparsity", {}).get("selectivity")),
        ("Dead Feature Fraction",
         lambda r: r.get("sparsity", {}).get("dead_frac")),
    ]

    active_metrics = []
    for title, fn in metrics:
        tx, ty, _ = _collect(taxon_r, fn)
        bx, by, _ = _collect(baseline_r, fn)
        if tx or bx:
            active_metrics.append((title, fn))

    if not active_metrics:
        print("  No data for taxon scatter plot, skipping.")
        return

    n = len(active_metrics)

    def _draw_panel(ax, tx, ty, tlabels, bx, by, blabels, title, x_cap=None):
        def _cx(x):
            return min(x, x_cap) if x_cap is not None else x
        ax.scatter([_cx(x) for x in bx], by, marker="X", s=90,
                   color=_BASE_COLOR, alpha=0.8, label="Non-taxon",
                   zorder=3, edgecolors="black", linewidths=0.4)
        ax.scatter([_cx(x) for x in tx], ty, marker="o", s=90,
                   color=_TAXON_COLOR, alpha=0.85, label="Taxon",
                   zorder=4, edgecolors="black", linewidths=0.4)
        for x, y, lbl in zip(tx, ty, tlabels):
            prefix = "\u2192" if (x_cap is not None and x > x_cap) else ""
            ax.annotate(f"{prefix}{lbl}", (_cx(x), y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=_TAXON_COLOR, alpha=0.85)
        for x, y, lbl in zip(bx, by, blabels):
            prefix = "\u2192" if (x_cap is not None and x > x_cap) else ""
            ax.annotate(f"{prefix}{lbl}", (_cx(x), y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=_BASE_COLOR, alpha=0.7)
        if tx:
            med = float(np.median(tx))
            ax.axvline(_cx(med), color=_TAXON_COLOR, linewidth=0.8, linestyle="--",
                       alpha=0.45, label=f"Taxon median L0={med:.0f}")
        if x_cap is not None:
            ax.set_xlim(-0.02 * x_cap, x_cap * 1.05)
            ax.axvline(x_cap, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlabel("Mean Active Features (L0 abs)", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    # Shared x_cap: 90th percentile of all L0 values
    all_xs_flat = []
    for _, fn in active_metrics:
        tx, _, _ = _collect(taxon_r, fn)
        bx, _, _ = _collect(baseline_r, fn)
        all_xs_flat.extend(tx + bx)
    x_cap = float(np.percentile(all_xs_flat, 90)) if all_xs_flat else None

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (title, fn) in enumerate(active_metrics):
        tx, ty, tlabels = _collect(taxon_r, fn)
        bx, by, blabels = _collect(baseline_r, fn)
        _draw_panel(axes[0, col], tx, ty, tlabels, bx, by, blabels, title)
        axes[0, col].set_title(f"{title} (full)", fontsize=10)
        _draw_panel(axes[1, col], tx, ty, tlabels, bx, by, blabels, title, x_cap=x_cap)
        axes[1, col].set_title(f"{title} (zoomed, L0 \u2264 {x_cap:.0f})", fontsize=10)

    fig.suptitle("Taxonomic vs Non-Taxonomic: Performance vs Sparsity (Tiny-ImageNet)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_imagenet_taxon_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_matched_pairs(all_results: List[dict], save_dir: Path):
    """Side-by-side bar chart: each taxon model vs its nearest-L0 non-taxon model."""
    pairs = find_sparsity_matched_pairs(all_results)
    if not pairs:
        print("  No sparsity-matched pairs found — skipping matched-pair plot.")
        return

    metric_defs = [
        ("Linear Top-1 Acc",
         lambda r: _linear_top1(r)),
        ("KNN k=5 Top-1 Acc",
         lambda r: r["knn"].get("k5_top1") if r.get("knn") else None),
        ("Feature Selectivity",
         lambda r: r.get("sparsity", {}).get("selectivity")),
        ("Dead Feature Fraction",
         lambda r: r.get("sparsity", {}).get("dead_frac")),
    ]

    active_defs = []
    for label, fn in metric_defs:
        tv = [fn(t) for t, _ in pairs]
        bv = [fn(b) for _, b in pairs]
        if any(v is not None for v in tv + bv):
            active_defs.append((label, fn))

    if not active_defs:
        print("  No metrics available for matched-pair plot, skipping.")
        return

    n_pairs   = len(pairs)
    n_metrics = len(active_defs)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(12, n_pairs * 1.8), 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    x_pos = np.arange(n_pairs)
    width = 0.35

    for ax, (label, fn) in zip(axes, active_defs):
        tv = [fn(t) for t, _ in pairs]
        bv = [fn(b) for _, b in pairs]
        pair_labels = [f"{t['short']}\nvs\n{b['short']}" for t, b in pairs]
        l0_annots   = [
            f"L0 {_sparsity_l0(t):.0f} / {_sparsity_l0(b):.0f}"
            for t, b in pairs
        ]

        def _safe(v):
            return float(v) if v is not None and not np.isnan(float(v)) else 0.0

        tv_safe = [_safe(v) for v in tv]
        bv_safe = [_safe(v) for v in bv]

        ax.bar(x_pos - width / 2, tv_safe, width,
               color=_TAXON_COLOR, label="Taxon",
               alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x_pos + width / 2, bv_safe, width,
               color=_BASE_COLOR, label="Non-taxon (matched)",
               alpha=0.85, edgecolor="black", linewidth=0.5)

        for i, (tv_i, bv_i) in enumerate(zip(tv_safe, bv_safe)):
            delta = tv_i - bv_i
            if abs(delta) > 1e-6:
                sign = "+" if delta > 0 else ""
                ax.text(x_pos[i] - width / 2, tv_i + 0.002,
                        f"{sign}{delta:.3f}", ha="center", va="bottom",
                        fontsize=6, color=_TAXON_COLOR, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(pair_labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"{label} — Taxon vs Sparsity-Matched Baseline", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ymin = ax.get_ylim()[0]
        for i, annot in enumerate(l0_annots):
            ax.annotate(annot, xy=(x_pos[i], ymin),
                        xytext=(0, -24), textcoords="offset points",
                        ha="center", fontsize=6, color="gray")

    fig.suptitle("Taxon vs Sparsity-Matched Non-Taxon Baselines (Tiny-ImageNet)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_imagenet_taxon_matched.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_taxon_class_compare(
    all_results: List[dict],
    save_dir: Path,
    class_names: Optional[List[str]] = None,
):
    """Per-class accuracy heatmap interleaving each taxon model with its matched baseline.

    Rows alternate: [Taxon] model then [Base] matched non-taxon.  Dashed horizontal
    lines separate pairs.  Analogous to ``plot_taxon_attr_compare`` in celeba.
    """
    pairs = find_sparsity_matched_pairs(all_results)
    valid_pairs = [(t, b) for t, b in pairs if t.get("linear")]
    if not valid_pairs:
        print("  No matched pairs with linear results — skipping class compare plot.")
        return

    row_models: List[dict] = []
    row_labels:  List[str]  = []
    row_colors:  List[str]  = []

    for t, b in valid_pairs:
        row_models.append(t)
        row_labels.append(f"[Taxon] {t['short']}")
        row_colors.append(_TAXON_COLOR)
        if b.get("linear"):
            row_models.append(b)
            row_labels.append(f"[Base]  {b['short']}")
            row_colors.append(_BASE_COLOR)

    if not row_models:
        return

    matrix = np.stack([r["linear"]["per_class_accuracy"] for r in row_models], axis=0)

    fig, ax = plt.subplots(
        figsize=(max(20, 200 * 0.12), max(4, len(row_models) * 0.55))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    tick_ids = list(range(0, 200, 10))
    ax.set_xticks(tick_ids)
    if class_names:
        tick_labels = [f"{c}:{class_names[c][:12]}" for c in tick_ids]
    else:
        tick_labels = [str(c) for c in tick_ids]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Class (Tiny-ImageNet 0–199)", fontsize=9)

    for ytick, color in zip(ax.get_yticklabels(), row_colors):
        ytick.set_color(color)

    # Dashed separator lines between pairs
    cursor = -0.5
    for t, b in valid_pairs:
        cursor += 1.0  # taxon row
        if b.get("linear"):
            cursor += 1.0  # baseline row
        ax.axhline(cursor, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_title(
        "Per-Class Linear Probe Accuracy — Taxon (blue) vs Sparsity-Matched Baseline (red)\n"
        "Pairs separated by dashed lines",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    plt.tight_layout()
    out = save_dir / "probe_imagenet_taxon_class_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_monosemanticity_scatter(all_results: List[dict], save_dir: Path):
    """Scatter: mean active channels (L0) vs monosemanticity score.

    Each point is one model coloured by model type.  A score of ≈1 means the
    top-K images for every feature match the global class distribution (random).
    Higher scores indicate features that fire preferentially for one class.
    """
    type_order = sorted({r["type"] for r in all_results if r.get("type")})
    colour_map = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(type_order)}

    xs, ys, colours, labels = [], [], [], []
    for r in all_results:
        l0   = _sparsity_l0(r)
        mono = r.get("sparsity", {}).get("monosemanticity")
        if l0 is None or mono is None or np.isnan(mono):
            continue
        xs.append(l0)
        ys.append(mono)
        colours.append(colour_map.get(r.get("type", ""), "#888888"))
        labels.append(r["short"])

    if not xs:
        print("  No monosemanticity data, skipping scatter.")
        return

    x_cap = float(np.percentile(xs, 90)) if xs else None

    def _draw_mono_panel(ax, x_cap=None):
        def _cx(x):
            return min(x, x_cap) if x_cap is not None else x
        ax.scatter([_cx(x) for x in xs], ys, c=colours, s=60, edgecolors="none", alpha=0.85, zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            prefix = "\u2192" if (x_cap is not None and x > x_cap) else ""
            ax.annotate(f"{prefix}{lbl}", (_cx(x), y), fontsize=5.5, ha="left", va="bottom",
                        xytext=(3, 2), textcoords="offset points", color="#333333")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.6, label="random (score=1)")
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map[t],
                       markersize=7, label=t)
            for t in type_order if t in colour_map
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  title="Model type", title_fontsize=7, framealpha=0.8)
        if x_cap is not None:
            ax.set_xlim(-0.02 * x_cap, x_cap * 1.05)
            ax.axvline(x_cap, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlabel("Mean Active Features (L0 abs)", fontsize=10)
        ax.set_ylabel("Mean Monosemanticity Score\n(max class lift, top-50 images)", fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.5)

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(9, 12))
    _draw_mono_panel(ax_full)
    ax_full.set_title("Monosemanticity vs. Sparsity \u2014 Tiny-ImageNet (full)", fontsize=12, fontweight="bold")
    _draw_mono_panel(ax_zoom, x_cap=x_cap)
    zoom_title = (f"Monosemanticity vs. Sparsity \u2014 Tiny-ImageNet (zoomed, L0 \u2264 {x_cap:.0f})"
                  if x_cap else "Monosemanticity vs. Sparsity \u2014 Tiny-ImageNet (zoomed)")
    ax_zoom.set_title(zoom_title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = save_dir / "probe_imagenet_monosemanticity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── t-SNE visualisation ─────────────────────────────────────────────────────

# Model types that receive t-SNE plots.
_TSNE_TYPES = {
    "bottleneck_topk_taxon",
    "bottleneck_topk_multi_taxon",
    "bottleneck_taxon",
    "bottleneck_multi_taxon",
    "topk_taxon",
    "topk_multi_taxon",
    "matryoshka_batch_topk_sae",
    "matryoshka_intermediate_topk_sae",
    "topk_sae",
    "intermediate_topk_sae",
}

# Number of top-frequent classes shown per panel in the per-model t-SNE grid.
_TSNE_N_CLASSES = 20

# Fixed headline class IDs used as columns in the cross-model comparison figure.
_TSNE_HEADLINE_CLASSES = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198]


def _run_tsne(
    latents: np.ndarray,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    pca_components: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run (optional PCA →) t-SNE on latents.

    Returns
    -------
    emb     : (N, 2) 2-D t-SNE embedding
    idx_sub : (N,)   indices of the subsample used (into original latents array)
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    N = latents.shape[0]
    n = min(max_samples, N)
    idx_sub = rng.choice(N, n, replace=False)
    X = latents[idx_sub].copy()

    # Standardise
    X = StandardScaler().fit_transform(X)

    # PCA first if dimensionality is high
    if X.shape[1] > pca_components:
        n_comp = min(pca_components, X.shape[0] - 1, X.shape[1])
        X = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    perp = min(perplexity, n / 4)
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)

    return emb.astype(np.float32), idx_sub


def _tsne_scatter_ax(
    ax: "plt.Axes",
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    size: float = 4.0,
    alpha: float = 0.6,
):
    """Draw a two-class coloured scatter on *ax* (1=this class / 0=other)."""
    pos = labels == 1
    neg = ~pos
    ax.scatter(emb[neg, 0], emb[neg, 1], s=size, c="#aec7e8", alpha=alpha,
               linewidths=0, rasterized=True, label="other")
    ax.scatter(emb[pos, 0], emb[pos, 1], s=size, c="#1f77b4", alpha=alpha,
               linewidths=0, rasterized=True, label="this class")
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_tsne_latents(
    result: dict,
    tsne_dir: Path,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    n_classes: int = _TSNE_N_CLASSES,
    class_names: Optional[List[str]] = None,
):
    """Per-model t-SNE grid: one panel per top-N class coloured by binary membership.

    The embedding is computed once and shared across all panels so spatial
    structure can be compared across classes directly.  Analogous to the
    per-attribute grid in probe_celeba_hq.
    """
    latents = result.get("_latents")
    labels  = result.get("_labels")
    if latents is None or labels is None or latents.shape[0] < 50:
        print(f"  [t-SNE] No cached latents for {result['short']}, skipping.")
        return

    print(f"  [t-SNE] Running for {result['short']} ({latents.shape[0]} samples)...")
    try:
        emb, idx = _run_tsne(latents, max_samples=max_samples,
                             perplexity=perplexity, n_iter=n_iter)
    except Exception as e:
        print(f"  [t-SNE] Failed: {e}")
        return

    labels_sub = labels[idx]  # (N_sub,)

    # Pick the n_classes most frequent in the subsample
    from collections import Counter
    top_classes = [c for c, _ in Counter(labels_sub.tolist()).most_common(n_classes)]

    ncols = 5
    nrows = (len(top_classes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes_flat = np.array(axes).flatten()

    for i, c in enumerate(top_classes):
        binary = (labels_sub == c).astype(np.int32)
        lbl = class_names[c] if (class_names and c < len(class_names)) else f"class {c}"
        _tsne_scatter_ax(axes_flat[i], emb, binary, lbl)

    for ax in axes_flat[len(top_classes):]:
        ax.set_visible(False)

    model_label = f"{result['short']}  ({result['type']})"
    fig.suptitle(
        f"t-SNE of Sparse Latent — {model_label}\n"
        f"Blue = class present | Grey = other   (n={emb.shape[0]}, top-{n_classes} classes)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    safe = result["short"].replace("/", "_").replace(" ", "_").replace(",", "")
    out = tsne_dir / f"probe_imagenet_tsne_{safe}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE] Saved: {out.name}")


def plot_tsne_comparison(
    all_results: List[dict],
    tsne_dir: Path,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    class_names: Optional[List[str]] = None,
):
    """Cross-model t-SNE comparison for headline classes.

    Rows = models; columns = ``_TSNE_HEADLINE_CLASSES``.
    Each model's embedding is computed independently (same perplexity/init),
    so inter-model spatial alignment is only qualitative.
    """
    tsne_results = [
        r for r in all_results
        if r.get("type") in _TSNE_TYPES
        and r.get("_latents") is not None
        and r["_latents"].shape[0] >= 50
    ]
    if not tsne_results:
        print("  [t-SNE compare] No eligible models — skipping.")
        return

    n_models = len(tsne_results)
    n_cols   = len(_TSNE_HEADLINE_CLASSES)
    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(n_cols * 2.0, n_models * 2.0))
    if n_models == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, result in enumerate(tsne_results):
        latents = result["_latents"]
        labels  = result["_labels"]
        print(f"  [t-SNE compare] {result['short']} ({latents.shape[0]} samples)...")
        try:
            emb, idx = _run_tsne(latents, max_samples=max_samples,
                                 perplexity=perplexity, n_iter=n_iter)
        except Exception as e:
            print(f"    Failed: {e}")
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue

        labels_sub = labels[idx]

        for col_idx, c in enumerate(_TSNE_HEADLINE_CLASSES):
            ax = axes[row_idx, col_idx]
            binary = (labels_sub == c).astype(np.int32)
            _tsne_scatter_ax(ax, emb, binary, "")
            if row_idx == 0:
                col_lbl = class_names[c] if (class_names and c < len(class_names)) else f"cls {c}"
                ax.set_title(col_lbl, fontsize=7, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(result["short"], fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=60)

    fig.suptitle(
        "t-SNE Latent Space Comparison — Tiny-ImageNet\n"
        "Rows: models | Cols: headline classes   (blue = present)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])

    out = tsne_dir / "probe_imagenet_tsne_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE compare] Saved: {out.name}")


# ─── save CSV ─────────────────────────────────────────────────────────────────

def save_csv(all_results, save_dir: Path):
    rows = []
    for r in all_results:
        row = {
            "name":              r["name"],
            "short":             r["short"],
            "type":              r["type"],
            "top1_accuracy":     r["linear"]["top1_accuracy"] if r.get("linear") else "",
            "top5_accuracy":     r["linear"]["top5_accuracy"] if r.get("linear") else "",
            "mean_knn1_top1":    r["knn"].get("k1_top1", "") if r.get("knn") else "",
            "mean_knn5_top1":    r["knn"].get("k5_top1", "") if r.get("knn") else "",
            "mean_knn20_top1":   r["knn"].get("k20_top1", "") if r.get("knn") else "",
            "mean_knn5_top5":    r["knn"].get("k5_top5", "") if r.get("knn") else "",
            "mean_l0":           r.get("sparsity", {}).get("mean_l0",    ""),
            "dead_frac":         r.get("sparsity", {}).get("dead_frac",  ""),
            "l0_abs":            r.get("sparsity", {}).get("l0_abs",     ""),
            "selectivity":       r.get("sparsity", {}).get("selectivity",""),
            "mean_jaccard":      r.get("sparsity", {}).get("mean_jaccard",""),
            "monosemanticity":   r.get("sparsity", {}).get("monosemanticity",""),
        }
        rows.append(row)

    if not rows:
        return

    out_path = save_dir / "probe_imagenet_results.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


# ─── latent cache ────────────────────────────────────────────────────────────

def _cache_path(run_path: Path, cache_dir: Path) -> Path:
    return cache_dir / run_path.name / "latents.npz"


def _load_cached_latents(path: Path):
    if path.exists():
        d = np.load(path)
        return d["latents"], d["labels"]
    return None, None


def _save_cached_latents(path: Path, latents: np.ndarray, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, latents=latents, labels=labels)


# ─── arg parsing ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Tiny-ImageNet model latents")
    p.add_argument("--outputs-dir", type=str, default="./outputs/imagenet",
                   help="Root outputs directory to scan for trained models")
    p.add_argument("--data-root", type=str, default="./data/tiny_imagenet",
                   help="Tiny-ImageNet data cache directory (passed as cache_dir to TinyImageNetLoader)")
    p.add_argument("--save-dir", type=str, default="./outputs/analysis_imagenet",
                   help="Directory to write analysis outputs")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=10000,
                   help="Max val images to use for probing")
    p.add_argument("--max-batches-sparsity", type=int, default=60,
                   help="Max batches for sparsity analysis")
    p.add_argument("--n-train-frac", type=float, default=0.8,
                   help="Fraction of samples used for probe training")
    p.add_argument("--knn-k", type=int, nargs="+", default=[1, 5, 20],
                   help="k values for KNN probing")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore cached latents and recompute from scratch")
    p.add_argument("--ablations", type=str, default=None, metavar="DIR",
                   help="Path to ablations outputs directory to include alongside "
                        "main runs (e.g. ./outputs/imagenet/ablations)")
    p.add_argument("--skip-sparsity", action="store_true",
                   help="Skip sparsity analysis (faster)")
    p.add_argument("--skip-knn", action="store_true",
                   help="Skip KNN probing")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE visualisation")
    p.add_argument("--tsne-max-samples", type=int, default=2000,
                   help="Max samples for t-SNE (keep ≤3000 for speed)")
    p.add_argument("--tsne-perplexity", type=float, default=30.0,
                   help="t-SNE perplexity")
    p.add_argument("--tsne-n-iter", type=int, default=1000,
                   help="t-SNE number of iterations")
    p.add_argument("--model-filter", type=str, default="",
                   help="Only process runs whose name contains this substring")
    return p.parse_args()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = save_dir / "latent_cache"

    print("=" * 80)
    print("Tiny-ImageNet Probing Analysis")
    print("=" * 80)
    print(f"  device      = {device}")
    print(f"  outputs_dir = {outputs_dir}")
    print(f"  save_dir    = {save_dir}")
    if args.ablations:
        print(f"  ablations   = {args.ablations}")
    print(f"  max_samples = {args.max_samples}")
    print("=" * 80)

    # ── discover runs ──────────────────────────────────────────────────────────
    runs = discover_runs(outputs_dir, include_ablations=False)
    if args.ablations:
        abl_dir = Path(args.ablations)
        if abl_dir.exists():
            abl_runs = discover_runs(abl_dir, include_ablations=True)
            existing = {r["path"] for r in runs}
            new_abl = [r for r in abl_runs if r["path"] not in existing]
            runs.extend(new_abl)
            print(f"  + {len(new_abl)} ablation run(s) from {abl_dir}")
        else:
            print(f"  WARNING: --ablations directory not found: {abl_dir}")
    if args.model_filter:
        runs = [r for r in runs if args.model_filter.lower() in r["name"].lower()]
    print(f"\nFound {len(runs)} runs to analyze:")
    for r in runs:
        print(f"  [{r['type']:35s}] {r['name']}")

    if not runs:
        print("No runs found. Exiting.")
        return

    # ── load val dataset ───────────────────────────────────────────────────────
    print("\n[Step 1] Loading Tiny-ImageNet val set...")
    try:
        val_loader = _load_tiny_imagenet_val(
            cache_dir=args.data_root,
            image_size=args.image_size,
            batch_size=64,
            max_samples=args.max_samples,
        )
        print(f"  Val loader ready (max_samples={args.max_samples})")
    except Exception as e:
        print(f"  WARNING: Could not load Tiny-ImageNet val set: {e}")
        print("  Linear probing and KNN will be skipped.")
        val_loader = None

    print("\n[Step 2] Building Tiny-ImageNet class name lookup...")
    class_names = _build_class_names(args.data_root)
    print(f"  {len(class_names)} classes (e.g. 0='{class_names[0]}', 99='{class_names[99]}', 199='{class_names[199]}')")

    # ── process each run ────────────────────────────────────────────────────────
    all_results = []

    for run_idx, run in enumerate(runs):
        print(f"\n[{run_idx + 1}/{len(runs)}] {run['name']}")
        result = {
            "name":     run["name"],
            "short":    run["short"],
            "type":     run["type"],
            "path":     str(run["path"]),
            "linear":   None,
            "knn":      None,
            "sparsity": {},
        }

        # Load model
        try:
            model, _ = load_model(run, device)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            all_results.append(result)
            continue

        model.eval()
        latents, labels = None, None

        # ── extract / load cached latents ──────────────────────────────────────
        if val_loader is not None:
            cpath = _cache_path(run["path"], cache_dir)
            if not args.force_recompute:
                latents, labels = _load_cached_latents(cpath)
                if latents is not None:
                    print(f"  Loaded cached latents: shape={latents.shape}")

            if latents is None:
                print(f"  Extracting latents (max {args.max_samples} samples)...")
                latents, labels = extract_latents(model, run["type"], val_loader, device)
                if latents.shape[0] > 0:
                    _save_cached_latents(cpath, latents, labels)
                    print(f"  Extracted: latents={latents.shape}, labels={labels.shape}")
                else:
                    print("  WARNING: No latents extracted, skipping.")
                    latents = None

            if latents is not None and latents.shape[0] >= 50:
                # ── stash latents for t-SNE (only eligible model types) ────────────
                if run["type"] in _TSNE_TYPES and not args.skip_tsne:
                    result["_latents"] = latents
                    result["_labels"]  = labels

                # ── linear probing ──────────────────────────────────────────
                print("  Running linear probe...")
                try:
                    result["linear"] = run_linear_probe(
                        latents, labels, n_train_frac=args.n_train_frac)
                    t1 = result["linear"]["top1_accuracy"]
                    t5 = result["linear"]["top5_accuracy"]
                    print(f"  Linear probe: top1={t1:.4f}, top5={t5:.4f}")
                except Exception as e:
                    print(f"  WARNING linear probe failed: {e}")

                # ── KNN ────────────────────────────────────────────────────────
                if not args.skip_knn:
                    print("  Running KNN...")
                    try:
                        result["knn"] = run_knn(
                            latents, labels,
                            k_values=tuple(args.knn_k),
                            n_train_frac=args.n_train_frac,
                        )
                        k5t1 = result["knn"].get("k5_top1", float("nan"))
                        print(f"  KNN k=5 top-1: {k5t1:.4f}")
                    except Exception as e:
                        print(f"  WARNING KNN failed: {e}")

        # ── sparsity stats ──────────────────────────────────────────────────────
        if not args.skip_sparsity and val_loader is not None:
            print("  Computing sparsity stats...")
            try:
                result["sparsity"] = compute_sparsity_stats(
                    model, val_loader, device,
                    max_batches=args.max_batches_sparsity)
                sp = result["sparsity"]
                print(f"  Sparsity: L0={sp['mean_l0']:.4f}, dead={sp['dead_frac']:.4f}, "
                      f"selectivity={sp['selectivity']:.3f}, jaccard={sp['mean_jaccard']:.4f}")
            except Exception as e:
                print(f"  WARNING sparsity failed: {e}")

        # ── monosemanticity (requires latents + labels) ───────────────────────
        if latents is not None and labels is not None and latents.shape[0] >= 50:
            try:
                mono = compute_monosemanticity(latents, labels)
                if not result.get("sparsity"):
                    result["sparsity"] = {}
                result["sparsity"]["monosemanticity"] = mono
                print(f"  Monosemanticity: {mono:.3f}")
            except Exception as e:
                print(f"  WARNING monosemanticity failed: {e}")

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_results.append(result)

    if not all_results:
        print("No results to save. Exiting.")
        return

    # ── save outputs ─────────────────────────────────────────────────────────────
    print("\n[Saving results]")
    save_csv(all_results, save_dir)

    has_linear = any(r.get("linear") for r in all_results)
    has_knn    = any(r.get("knn") for r in all_results)
    has_sp     = any(r.get("sparsity") for r in all_results)

    if has_linear:
        plot_linear_results(all_results, save_dir)
        plot_class_heatmap(all_results, save_dir, class_names=class_names)
    if has_knn:
        plot_knn_results(all_results, save_dir)
    if has_sp:
        plot_sparsity_results(all_results, save_dir)

    has_mono = any(r.get("sparsity", {}).get("monosemanticity") is not None
                   for r in all_results)
    if has_mono:
        plot_monosemanticity_scatter(all_results, save_dir)

    has_taxon    = any(_is_taxon(r) for r in all_results)
    has_baseline = any(not _is_taxon(r) for r in all_results)
    if has_taxon and has_baseline:
        print("\n[Taxon vs. Baseline Comparisons]")
        if has_sp:
            plot_taxon_scatter(all_results, save_dir)
        plot_matched_pairs(all_results, save_dir)
        if has_linear:
            plot_taxon_class_compare(all_results, save_dir, class_names=class_names)

    # ── t-SNE visualisations ────────────────────────────────────────────────
    if not args.skip_tsne:
        tsne_eligible = [r for r in all_results if r.get("_latents") is not None]
        if tsne_eligible:
            tsne_dir = save_dir / "tsne"
            tsne_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[t-SNE] {len(tsne_eligible)} model(s) eligible, saving to {tsne_dir}")
            tsne_kw = dict(
                max_samples=args.tsne_max_samples,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_n_iter,
            )
            for r in tsne_eligible:
                plot_tsne_latents(r, tsne_dir, **tsne_kw, class_names=class_names)
            if len(tsne_eligible) > 1:
                plot_tsne_comparison(all_results, tsne_dir, **tsne_kw, class_names=class_names)
        else:
            print("\n[t-SNE] No eligible models with latents — skipping.")

    print(f"\nAll outputs saved to: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
