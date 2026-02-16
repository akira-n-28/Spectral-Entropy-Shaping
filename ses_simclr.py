#!/usr/bin/env python3
"""
SES — Experiment 23: SimCLR Self-Supervised Learning (2x T4 16GB)
=================================================================
First application of SES to self-supervised learning.
SimCLR contrastive pretraining on CIFAR-100 with ResNet-18 backbone,
then linear probing to evaluate learned representations.

GPU 0 (2 configs):
  1. baseline      — SimCLR without SES
  2. ses_b07       — SimCLR + SES β=0.7 (all layers)

GPU 1 (2 configs):
  3. ses_b05       — SimCLR + SES β=0.5
  4. ses_b07_l3    — SimCLR + SES β=0.7 (last-3 layers only)

Pretraining: 200 epochs, NT-Xent loss, temperature τ=0.5,
ResNet-18 backbone + 2-layer MLP projection head (128-d).
Linear probe: 100 epochs, SGD lr=0.3, on frozen features.

Usage:
  python ses_simclr.py                    # Launches both workers
  python ses_simclr.py --worker gpu0      # Internal: GPU0 worker
  python ses_simclr.py --worker gpu1      # Internal: GPU1 worker
"""

import os
import sys
import time
import json
import math
import copy
import argparse
import warnings
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    OUTPUT_DIR = Path("/kaggle/working/outputs_simclr") if Path("/kaggle").exists() else Path("./outputs_simclr")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    # Pretraining
    PRETRAIN_EPOCHS = 200
    PRETRAIN_BATCH = 512
    PRETRAIN_LR = 0.3  # scaled: 0.3 * batch_size / 256
    PRETRAIN_WD = 1e-4
    TEMPERATURE = 0.5
    PROJ_DIM = 128

    # Linear probe
    PROBE_EPOCHS = 100
    PROBE_BATCH = 256
    PROBE_LR = 0.3
    PROBE_WD = 0.0

    NUM_WORKERS = 2
    LAMBDA_SES = 0.01
    SEED = 42


CFG = Config()
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def gpu_tag():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    return f"[GPU{vis}]"


# ══════════════════════════════════════════════════════════════════════════════
# SES CORE
# ══════════════════════════════════════════════════════════════════════════════

def spectral_entropy(H, eps=1e-12):
    B, d = H.shape
    if B < 2:
        return torch.tensor(0.0, device=H.device), torch.tensor(1.0, device=H.device)
    H_c = H - H.mean(dim=0, keepdim=True)
    cov = (H_c.T @ H_c) / (B - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=eps)
    p = eigvals / eigvals.sum()
    entropy = -(p * p.log()).sum()
    erank = entropy.exp()
    return entropy, erank


def ses_regularizer(activations, beta=0.7, eps=1e-12):
    """SES loss — always computed in float32."""
    anchor_device = activations[0].device
    reg = torch.tensor(0.0, device=anchor_device)
    for H in activations:
        if H.device != anchor_device:
            H = H.to(anchor_device)
        if H.dim() == 4:       # CNN: [B, C, H, W] -> GAP
            H = H.mean(dim=[2, 3])
        elif H.dim() > 2:
            H = H.flatten(1)
        H = H.float()
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# RESNET-18 BACKBONE (CIFAR-adapted)
# ══════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone without final FC — outputs 512-d features."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    def _make_layer(self, in_p, out_p, n_blocks, stride):
        layers = [BasicBlock(in_p, out_p, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_p, out_p, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)  # [B, 512]


# ══════════════════════════════════════════════════════════════════════════════
# SIMCLR MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SimCLRModel(nn.Module):
    """ResNet-18 backbone + MLP projection head for SimCLR."""
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = ResNet18Backbone()
        feat_dim = self.backbone.feat_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)       # [B, 512] — representations
        z = self.projector(h)       # [B, proj_dim] — projections
        return h, F.normalize(z, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# NT-XENT LOSS
# ══════════════════════════════════════════════════════════════════════════════

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: [B, D] L2-normalized projections of two augmented views.
        Returns scalar NT-Xent loss.
        """
        B = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_i = torch.arange(B, device=z.device)
        pos_j = pos_i + B
        labels = torch.cat([pos_j, pos_i], dim=0)  # [2B]

        loss = F.cross_entropy(sim, labels)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTOR — hooks on ResNet backbone layers
# ══════════════════════════════════════════════════════════════════════════════

class BackboneCollector:
    """Hook on ResNet backbone layers for SES during pretraining."""
    def __init__(self, model, hook_mode="all"):
        self.activations = []
        self.hooks = []
        self.enabled = True
        backbone = model.backbone

        all_layers = [
            ("layer1", backbone.layer1),
            ("layer2", backbone.layer2),
            ("layer3", backbone.layer3),
            ("layer4", backbone.layer4),
        ]

        if hook_mode == "last_3":
            target_layers = all_layers[1:]  # layer2, layer3, layer4
        elif hook_mode == "all":
            target_layers = all_layers
        else:
            raise ValueError(f"Unknown hook_mode: {hook_mode}")

        for name, layer in target_layers:
            for block in layer:
                self.hooks.append(block.register_forward_hook(self._hook_fn))

        indices = [name for name, _ in target_layers]
        print(f"  [BackboneCollector] {len(self.hooks)} hooks ({hook_mode}): {indices}")

    def _hook_fn(self, module, input, output):
        if self.enabled:
            if output.dim() == 4:
                self.activations.append(output.mean(dim=[2, 3]))
            else:
                self.activations.append(output.view(output.size(0), -1))

    def clear(self):
        self.activations = []
    def disable(self):
        self.enabled = False; self.activations = []
    def enable(self):
        self.enabled = True
    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks = []


# ══════════════════════════════════════════════════════════════════════════════
# DATA — SimCLR augmentations
# ══════════════════════════════════════════════════════════════════════════════

class SimCLRAugmentation:
    """Produces two augmented views of the same image."""
    def __init__(self, img_size=32):
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLRDataset(Dataset):
    """Wraps a torchvision dataset with SimCLR dual augmentation."""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        x_i, x_j = self.transform(img)
        return x_i, x_j, label


def get_pretrain_loader():
    root = str(CFG.DATA_DIR / "cifar100")
    base = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    augment = SimCLRAugmentation(img_size=32)
    dataset = SimCLRDataset(base, augment)
    loader = DataLoader(dataset, batch_size=CFG.PRETRAIN_BATCH, shuffle=True,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    print(f"  [SimCLR Pretrain] {len(dataset)} images, batch {CFG.PRETRAIN_BATCH}")
    return loader


def get_probe_loaders():
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = str(CFG.DATA_DIR / "cifar100")
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=CFG.PROBE_BATCH, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=CFG.PROBE_BATCH, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULERS
# ══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ══════════════════════════════════════════════════════════════════════════════
# SIMCLR PRETRAINING
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_simclr(model, collector, device, pretrain_loader,
                    label, beta=0.7, lambda_ses=0.0):
    tag = gpu_tag()
    # LARS-like: SGD with high momentum, scaled LR
    lr = CFG.PRETRAIN_LR * CFG.PRETRAIN_BATCH / 256
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=CFG.PRETRAIN_WD)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=10,
                                       total_epochs=CFG.PRETRAIN_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    criterion = NTXentLoss(temperature=CFG.TEMPERATURE)

    history = defaultdict(list)
    epoch_times = []

    pbar = tqdm(range(CFG.PRETRAIN_EPOCHS), desc=f"{tag} PT {label}", unit="ep", leave=True)
    for epoch in pbar:
        t0 = time.time()
        lr = scheduler.step(epoch)
        model.train()
        loss_m = AverageMeter()

        for x_i, x_j, _ in pretrain_loader:
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            if collector:
                collector.clear()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                contrastive_loss = criterion(z_i, z_j)

            reg_loss = torch.tensor(0.0, device=device)
            if collector and lambda_ses > 0 and collector.activations:
                reg_loss = lambda_ses * ses_regularizer(collector.activations, beta=beta)

            total_loss = contrastive_loss.float() + reg_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_m.update(total_loss.item(), x_i.size(0))

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        history["pretrain_loss"].append(loss_m.avg)
        history["lr"].append(lr)

        pbar.set_postfix_str(f"L {loss_m.avg:.4f} lr {lr:.5f} {epoch_time:.1f}s")

    if collector:
        collector.remove()

    print(f"{tag} Pretraining {label} DONE: Final Loss {history['pretrain_loss'][-1]:.4f} "
          f"({np.mean(epoch_times):.1f}s/ep)")

    return {
        "pretrain_loss_history": history["pretrain_loss"],
        "avg_pretrain_time": float(np.mean(epoch_times)),
        "vram_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(backbone, loader, device):
    """Extract frozen features from backbone for linear probing."""
    backbone.eval()
    all_feats, all_labels = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            feats = backbone(X)
        all_feats.append(feats.float().cpu())
        all_labels.append(y)
    return torch.cat(all_feats, 0), torch.cat(all_labels, 0)


def linear_probe(backbone, device, label):
    """Train linear classifier on frozen features."""
    tag = gpu_tag()
    print(f"\n{tag} Linear Probe: {label}")

    train_loader, test_loader = get_probe_loaders()

    # Extract features
    print(f"{tag}   Extracting features...")
    train_feats, train_labels = extract_features(backbone, train_loader, device)
    test_feats, test_labels = extract_features(backbone, test_loader, device)
    print(f"{tag}   Train: {train_feats.shape}, Test: {test_feats.shape}")

    # Create feature dataloaders
    train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    test_ds = torch.utils.data.TensorDataset(test_feats, test_labels)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Linear classifier
    feat_dim = train_feats.shape[1]
    classifier = nn.Linear(feat_dim, 100).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=CFG.PROBE_LR,
                          momentum=0.9, weight_decay=CFG.PROBE_WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.PROBE_EPOCHS)

    history = defaultdict(list)
    best_acc = 0.0

    pbar = tqdm(range(CFG.PROBE_EPOCHS), desc=f"{tag} LP {label}", unit="ep", leave=True)
    for epoch in pbar:
        classifier.train()
        correct = total = 0
        for feats, labels in train_dl:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_acc = 100.0 * correct / total

        # Eval
        classifier.eval()
        correct = total = 0
        for feats, labels in test_dl:
            feats, labels = feats.to(device), labels.to(device)
            logits = classifier(feats)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        test_acc = 100.0 * correct / total

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

        pbar.set_postfix_str(f"Te {test_acc:.1f}%")

    print(f"{tag} Probe {label}: Best {best_acc:.2f}%")
    return {
        "probe_best_acc": best_acc,
        "probe_history": dict(history),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CONFIG RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_config(device, label, beta=0.7, hook_mode="all", use_ses=False):
    tag = gpu_tag()
    set_seed(CFG.SEED)

    model = SimCLRModel(proj_dim=CFG.PROJ_DIM).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [SimCLR ResNet-18] {nparams:.1f}M params, proj_dim={CFG.PROJ_DIM}")

    collector = None
    lambda_ses = 0.0
    if use_ses:
        collector = BackboneCollector(model, hook_mode=hook_mode)
        lambda_ses = CFG.LAMBDA_SES

    pretrain_loader = get_pretrain_loader()

    # Phase 1: Pretraining
    pretrain_results = pretrain_simclr(
        model, collector, device, pretrain_loader,
        label=label, beta=beta, lambda_ses=lambda_ses)

    # Phase 2: Linear Probe
    probe_results = linear_probe(model.backbone, device, label)

    # Merge
    result = {
        **pretrain_results,
        **probe_results,
        "beta": beta,
        "hook_mode": hook_mode,
        "use_ses": use_ses,
        "label": label,
    }

    del model, collector
    torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def experiment_gpu0(device):
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} SimCLR SES — GPU 0: Baseline + SES β=0.7")
    print(f"{tag} {'=' * 55}")

    results = {}

    configs = [
        ("baseline",  0.7, "all",  False, "Baseline (no SES)"),
        ("ses_b07",   0.7, "all",  True,  "SES β=0.7 All"),
    ]

    for key, beta, hook_mode, use_ses, label in configs:
        print(f"\n{tag} --- {label} ---")
        results[key] = run_single_config(device, label, beta=beta,
                                          hook_mode=hook_mode, use_ses=use_ses)

    return results


def experiment_gpu1(device):
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} SimCLR SES — GPU 1: SES β=0.5 + SES β=0.7 Last-3")
    print(f"{tag} {'=' * 55}")

    results = {}

    configs = [
        ("ses_b05",     0.5, "all",    True, "SES β=0.5 All"),
        ("ses_b07_l3",  0.7, "last_3", True, "SES β=0.7 Last-3"),
    ]

    for key, beta, hook_mode, use_ses, label in configs:
        print(f"\n{tag} --- {label} ---")
        results[key] = run_single_config(device, label, beta=beta,
                                          hook_mode=hook_mode, use_ses=use_ses)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_results(all_results, save_dir):
    order = ["baseline", "ses_b07", "ses_b05", "ses_b07_l3"]
    labels = ["Baseline", "SES β=0.7\nAll", "SES β=0.5\nAll", "SES β=0.7\nLast-3"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]

    present = [k for k in order if k in all_results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    # ═══════ Main bar chart: probe accuracy ═══════
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Exp 23: SES on SimCLR Self-Supervised Learning (CIFAR-100, ResNet-18)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Linear probe accuracy
    ax = axes[0]
    vals = [all_results[k]["probe_best_acc"] for k in present]
    bars = ax.bar(range(len(present)), vals, color=present_colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present_labels, fontsize=9)
    ax.set_ylabel("Linear Probe Acc (%)")
    ax.set_title("Linear Probe Accuracy (100 epochs)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Pretrain loss curves
    ax = axes[1]
    for i, key in enumerate(present):
        loss_hist = all_results[key]["pretrain_loss_history"]
        ax.plot(loss_hist, color=present_colors[i], lw=2,
                label=present_labels[i].replace('\n', ' '))
    ax.set_xlabel("Pretrain Epoch")
    ax.set_ylabel("NT-Xent Loss")
    ax.set_title("Pretraining Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Probe learning curves
    ax = axes[2]
    for i, key in enumerate(present):
        if "probe_history" in all_results[key]:
            h = all_results[key]["probe_history"]
            ax.plot(h["test_acc"], color=present_colors[i], lw=2,
                    label=f"{present_labels[i].replace(chr(10), ' ')} "
                          f"({all_results[key]['probe_best_acc']:.2f}%)")
    ax.set_xlabel("Probe Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Linear Probe Curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "23_simclr_ses.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {fname}")

    # ═══════ Delta vs baseline ═══════
    if "baseline" in all_results:
        base_acc = all_results["baseline"]["probe_best_acc"]
        ses_keys = [k for k in present if k != "baseline"]
        ses_labels = [present_labels[present.index(k)].replace('\n', ' ') for k in ses_keys]
        ses_colors = [present_colors[present.index(k)] for k in ses_keys]

        fig, ax = plt.subplots(figsize=(10, 5))
        deltas = [all_results[k]["probe_best_acc"] - base_acc for k in ses_keys]
        bars = ax.bar(range(len(ses_keys)), deltas, color=ses_colors, alpha=0.85)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.05 if val >= 0 else -0.15),
                    f"{val:+.2f}", ha="center", fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(ses_keys)))
        ax.set_xticklabels(ses_labels, fontsize=10)
        ax.set_ylabel("Δ Linear Probe Acc vs Baseline (pp)")
        ax.set_title(f"Exp 23: SES Improvement over SimCLR Baseline ({base_acc:.2f}%)",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fname = "23b_simclr_delta.png"
        plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def worker_main(experiment_name):
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    tag = gpu_tag()
    print(f"{tag} Worker '{experiment_name}' started — {name} ({vram:.1f} GB)")

    if experiment_name == "gpu0":
        results = experiment_gpu0(device)
        out_path = CFG.OUTPUT_DIR / "gpu0_simclr_results.json"
    elif experiment_name == "gpu1":
        results = experiment_gpu1(device)
        out_path = CFG.OUTPUT_DIR / "gpu1_simclr_results.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Experiment 23: SimCLR Self-Supervised Learning          ║")
    print("║   GPU 0: Baseline + SES β=0.7                                  ║")
    print("║   GPU 1: SES β=0.5 + SES β=0.7 Last-3                         ║")
    print("║   ResNet-18, CIFAR-100, 200ep pretrain + 100ep probe            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"\n[INFO] {n_gpus} GPU(s) detected:")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # Download data
    print("\n[DATA] Pre-downloading CIFAR-100...")
    root = str(CFG.DATA_DIR / "cifar100")
    torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    print("[DATA] CIFAR-100 ready.")

    script_path = os.path.abspath(__file__)

    if n_gpus >= 2:
        print(f"\n[PARALLEL] Launching 2 workers...")
        env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

        p0 = subprocess.Popen(
            [sys.executable, script_path, "--worker", "gpu0"],
            env=env0, stdout=sys.stdout, stderr=sys.stderr)
        p1 = subprocess.Popen(
            [sys.executable, script_path, "--worker", "gpu1"],
            env=env1, stdout=sys.stdout, stderr=sys.stderr)

        ret0 = p0.wait()
        ret1 = p1.wait()
        print(f"\n[PARALLEL] Workers finished (exit codes: gpu0={ret0}, gpu1={ret1})")
    else:
        print(f"\n[SEQUENTIAL] Only {n_gpus} GPU — running sequentially...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        worker_main("gpu0")
        worker_main("gpu1")

    # ── Load and merge results ──────────────────────────────
    gpu0_path = CFG.OUTPUT_DIR / "gpu0_simclr_results.json"
    gpu1_path = CFG.OUTPUT_DIR / "gpu1_simclr_results.json"

    all_results = {}
    for path in [gpu0_path, gpu1_path]:
        if path.exists():
            with open(path) as f:
                all_results.update(json.load(f))

    if all_results:
        # Save summary (strip large histories)
        summary = {}
        for k, v in all_results.items():
            summary[k] = {kk: vv for kk, vv in v.items()
                          if kk not in ("pretrain_loss_history", "probe_history")}
        summary_path = CFG.OUTPUT_DIR / "simclr_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save full
        full_path = CFG.OUTPUT_DIR / "simclr_full_results.json"
        with open(full_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Plot
        plot_all_results(all_results, CFG.OUTPUT_DIR)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 23: SIMCLR SES COMPLETE — {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    if all_results:
        print(f"\n┌─ SimCLR + SES (CIFAR-100, ResNet-18, τ={CFG.TEMPERATURE}) ──────────┐")
        order = ["baseline", "ses_b07", "ses_b05", "ses_b07_l3"]
        nice = {"baseline": "Baseline", "ses_b07": "SES β=0.7 All",
                "ses_b05": "SES β=0.5 All", "ses_b07_l3": "SES β=0.7 L3"}

        base_acc = all_results.get("baseline", {}).get("probe_best_acc", 0)
        for key in order:
            if key in all_results:
                r = all_results[key]
                name = nice[key]
                acc = r["probe_best_acc"]
                delta = acc - base_acc if base_acc > 0 else 0
                delta_str = f"({delta:+.2f}pp)" if key != "baseline" else ""
                print(f"│  {name:<20} Probe Acc: {acc:.2f}% {delta_str}  "
                      f"PT Loss: {r['pretrain_loss_history'][-1]:.4f}  "
                      f"{r['avg_pretrain_time']:.1f}s/ep")

        print(f"└──────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Exp 23: SimCLR SSL")
    parser.add_argument("--worker", choices=["gpu0", "gpu1"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
