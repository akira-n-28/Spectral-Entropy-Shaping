#!/usr/bin/env python3
"""
SES — ViT Experiments on Kaggle (2x T4 16GB)
=============================================
Tests SES on Vision Transformers (non-convolutional architectures).
Mixed Precision FP16 + SES regularizer in FP32.
True parallelism via subprocess (not threading).

GPU 0: ViT-Small/4 on CIFAR-100 (Baseline + SES)
GPU 1: ViT-Tiny/4  on CIFAR-100 (Baseline + SES)

Usage:
  python ses_vit_experiment.py                      # Launches both workers
  python ses_vit_experiment.py --worker vit_small   # Internal: ViT-Small on assigned GPU
  python ses_vit_experiment.py --worker vit_tiny    # Internal: ViT-Tiny on assigned GPU
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
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    OUTPUT_DIR = Path("/kaggle/working/outputs_vit") if Path("/kaggle").exists() else Path("./outputs_vit")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    EPOCHS = 50
    BATCH_SIZE = 512
    NUM_WORKERS = 2

    # AdamW (standard for ViTs)
    LR = 1e-3
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5

    # SES
    LAMBDA_SES = 0.01
    BETA = 0.7
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
# SES CORE (identical to CNN experiments)
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
    """SES loss — always computed in float32 regardless of autocast."""
    anchor_device = activations[0].device
    reg = torch.tensor(0.0, device=anchor_device)
    for H in activations:
        if H.device != anchor_device:
            H = H.to(anchor_device)
        if H.dim() > 2:
            H = H.mean(dim=[2, 3]) if H.dim() == 4 else H.flatten(1)
        H = H.float()  # force float32 for eigendecomposition
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# VISION TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)               # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        # Stochastic depth (DropPath)
        self.drop_path_rate = drop_path

    def _drop_path(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob

    def forward(self, x):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self._drop_path(attn_out)
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for CIFAR-sized images (32x32).

    Configurations:
      ViT-Tiny/4:  embed_dim=192, depth=12, heads=3  (~5.5M params)
      ViT-Small/4: embed_dim=384, depth=12, heads=6  (~22M params)
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, dpr[i])
            for i in range(depth)
        ])

        # Final norm + classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)            # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token for classification


def make_vit_small(num_classes=100):
    model = VisionTransformer(
        img_size=32, patch_size=4, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        drop_rate=0.0, drop_path_rate=0.1)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [ViT-Small/4] {nparams:.1f}M params, embed_dim=384, depth=12, heads=6")
    return model


def make_vit_tiny(num_classes=100):
    model = VisionTransformer(
        img_size=32, patch_size=4, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
        drop_rate=0.0, drop_path_rate=0.1)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [ViT-Tiny/4] {nparams:.1f}M params, embed_dim=192, depth=12, heads=3")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# VIT ACTIVATION COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class ViTCollector:
    """
    Hooks on TransformerBlock outputs for SES.
    Pools over sequence dimension: [B, seq_len, embed_dim] → [B, embed_dim].
    """

    def __init__(self, model, hook_last_n=None):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []

        blocks = list(model.blocks)
        n_blocks = len(blocks)

        if hook_last_n is not None and hook_last_n < n_blocks:
            start_idx = n_blocks - hook_last_n
            target_blocks = blocks[start_idx:]
            block_indices = range(start_idx, n_blocks)
        else:
            target_blocks = blocks
            block_indices = range(n_blocks)

        for idx, block in zip(block_indices, target_blocks):
            self.hooks.append(block.register_forward_hook(self._hook_fn))
            self.hook_names.append(f"block_{idx}")

        print(f"  [ViT Hooks] {len(self.hooks)} blocks: "
              f"{', '.join(self.hook_names[:4])}{'...' if len(self.hook_names) > 4 else ''}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            # output shape: [B, seq_len, embed_dim]
            # Mean pool over sequence dimension (patches + CLS)
            pooled = output.mean(dim=1)  # [B, embed_dim]
            self.activations.append(pooled)

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
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_cifar100():
    root = str(CFG.DATA_DIR / "cifar100")
    torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    print("[DATA] CIFAR-100 ready.")


def get_loaders(batch_size=None):
    if batch_size is None:
        batch_size = CFG.BATCH_SIZE
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    root = str(CFG.DATA_DIR / "cifar100")
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# CORRUPTION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def apply_corruption(images, corruption_type, severity=3):
    if corruption_type == "gaussian_noise":
        scale = [0.04, 0.06, 0.08, 0.10, 0.12][severity - 1]
        return images + torch.randn_like(images) * scale
    elif corruption_type == "gaussian_blur":
        k = [3, 3, 5, 5, 7][severity - 1]
        pad = k // 2; c = images.shape[1]
        kernel = torch.ones(c, 1, k, k, device=images.device) / (k * k)
        return F.conv2d(images, kernel, padding=pad, groups=c)
    elif corruption_type == "contrast":
        factor = [0.8, 0.6, 0.4, 0.3, 0.2][severity - 1]
        mean = images.mean(dim=[2, 3], keepdim=True)
        return (images - mean) * factor + mean
    elif corruption_type == "brightness":
        delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        return images + delta
    elif corruption_type == "shot_noise":
        scale = [500, 250, 100, 75, 50][severity - 1]
        return torch.poisson(images.clamp(0) * scale) / scale
    return images


@torch.no_grad()
def evaluate_robustness(model, test_loader, device, severities=[1, 3, 5]):
    model.eval()
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    results = {}
    for corr in corruptions:
        for sev in severities:
            correct = total = 0
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                X_c = apply_corruption(X, corr, sev)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    correct += (model(X_c).argmax(1) == y).sum().item()
                total += y.size(0)
            results[f"{corr}_s{sev}"] = 100.0 * correct / total
    results["mean_corruption_acc"] = float(np.mean(
        [v for k, v in results.items() if k != "mean_corruption_acc"]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# COSINE LR WITH WARMUP
# ══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler:
    """Linear warmup + cosine decay."""

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
# TRAINING — Mixed Precision FP16 (T4 Tensor Cores)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.7):
    """FP16 autocast for forward, FP32 for SES."""
    model.train()
    loss_m = AverageMeter()
    correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if collector:
            collector.clear()

        optimizer.zero_grad(set_to_none=True)

        # ── Forward in FP16 ──
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(X)
            task_loss = F.cross_entropy(output, y)

        # ── SES regularizer in FP32 (outside autocast) ──
        reg_loss = torch.tensor(0.0, device=device)
        if collector and lambda_ses > 0 and collector.activations:
            reg_loss = lambda_ses * ses_regularizer(collector.activations, beta=beta)

        total_loss = task_loss.float() + reg_loss

        # ── Backward with gradient scaling ──
        scaler.scale(total_loss).backward()
        # Gradient clipping (standard for ViTs)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = X.size(0)
        loss_m.update(total_loss.item(), bs)
        correct += (output.argmax(1) == y).sum().item()
        total += bs

    return {"loss": loss_m.avg, "acc": 100.0 * correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m = AverageMeter()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(X)
            loss_m.update(F.cross_entropy(output, y).item(), X.size(0))
        correct += (output.argmax(1) == y).sum().item()
        total += X.size(0)
    return {"loss": loss_m.avg, "acc": 100.0 * correct / total}


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_config(model, collector, device, train_loader, test_loader,
                      label, use_ses=False):
    """Train one (model, config) pair. Returns result dict with history."""
    tag = gpu_tag()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR,
                             weight_decay=CFG.WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(optimizer, CFG.WARMUP_EPOCHS, CFG.EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    history = defaultdict(list)
    best_test_acc = 0.0
    best_state = None
    epoch_times = []

    pbar = tqdm(range(CFG.EPOCHS), desc=f"{tag} {label}", unit="ep", leave=True)
    for epoch in pbar:
        t0 = time.time()
        lr = scheduler.step(epoch)

        train_res = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                     collector=collector,
                                     lambda_ses=CFG.LAMBDA_SES if use_ses else 0.0,
                                     beta=CFG.BETA)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        if collector:
            collector.disable()
        test_res = evaluate(model, test_loader, device)
        if collector:
            collector.enable()

        history["train_acc"].append(train_res["acc"])
        history["test_acc"].append(test_res["acc"])

        if test_res["acc"] > best_test_acc:
            best_test_acc = test_res["acc"]
            best_state = copy.deepcopy(model.state_dict())

        gap = train_res["acc"] - test_res["acc"]
        vram = torch.cuda.max_memory_allocated(device) / 1e9
        pbar.set_postfix_str(f"Te {test_res['acc']:.1f}% Gap {gap:.1f} lr {lr:.5f} {epoch_time:.1f}s {vram:.1f}GB")

    if collector:
        collector.remove()
    model.load_state_dict(best_state)

    print(f"{tag} {label}: evaluating robustness...")
    rob = evaluate_robustness(model, test_loader, device)
    gap = history["train_acc"][-1] - history["test_acc"][-1]

    result = {
        "best_acc": best_test_acc,
        "final_gap": gap,
        "rob_mean": rob["mean_corruption_acc"],
        "robustness_detail": {k: v for k, v in rob.items() if k != "mean_corruption_acc"},
        "avg_epoch_time": float(np.mean(epoch_times)),
        "vram_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "history": dict(history),
    }
    print(f"{tag} {label} DONE: Acc {best_test_acc:.2f}% | Gap {gap:.2f}pp | "
          f"Rob {rob['mean_corruption_acc']:.2f}% | {np.mean(epoch_times):.1f}s/ep")

    del optimizer, scheduler, scaler
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: ViT-Small/4 on CIFAR-100
# ══════════════════════════════════════════════════════════════════════════════

def experiment_vit_small(device):
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT-Small/4 on CIFAR-100 (Baseline + SES)")
    print(f"{tag} AdamW lr={CFG.LR}, cosine + warmup, FP16, SES in FP32")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()
    results = {}

    for use_ses, label in [(False, "ViTS-Baseline"), (True, "ViTS-SES")]:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = ViTCollector(model) if use_ses else None

        key = "ses" if use_ses else "baseline"
        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: ViT-Tiny/4 on CIFAR-100
# ══════════════════════════════════════════════════════════════════════════════

def experiment_vit_tiny(device):
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT-Tiny/4 on CIFAR-100 (Baseline + SES)")
    print(f"{tag} AdamW lr={CFG.LR}, cosine + warmup, FP16, SES in FP32")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()
    results = {}

    for use_ses, label in [(False, "ViTT-Baseline"), (True, "ViTT-SES")]:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_tiny(num_classes=100).to(device)
        collector = ViTCollector(model) if use_ses else None

        key = "ses" if use_ses else "baseline"
        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_vit_results(results, title_suffix, prefix, save_dir):
    methods = ["Baseline", "SES"]
    keys = ["baseline", "ses"]
    colors = ["#377eb8", "#e41a1c"]

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title_suffix} on CIFAR-100 (FP16, AdamW)",
                 fontsize=14, fontweight="bold")
    for idx, (title, ylabel, fn) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in keys]
        bars = ax.bar(range(2), vals, color=colors, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(2)); ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_bars.png", dpi=150, bbox_inches="tight")
    plt.close(); print(f"[PLOT] {prefix}_bars.png")

    # Training curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color, label in zip(keys, colors, methods):
        h = results[key]["history"]
        ax.plot(h["train_acc"], color=color, ls="--", alpha=0.4)
        ax.plot(h["test_acc"], color=color, lw=2,
                label=f"{label} ({results[key]['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title_suffix} Training Curves (dashed=train, solid=test)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(); print(f"[PLOT] {prefix}_curves.png")

    # Per-corruption
    fig, ax = plt.subplots(figsize=(16, 6))
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    severities = [1, 3, 5]
    xlabels = [f"{c.replace('_',' ')}\ns{s}" for c in corruptions for s in severities]
    x = np.arange(len(xlabels)); w = 0.35
    for i, (key, color, label) in enumerate(zip(keys, colors, methods)):
        rob = results[key]["robustness_detail"]
        vals = [rob[f"{c}_s{s}"] for c in corruptions for s in severities]
        ax.bar(x + (i - 0.5) * w, vals, w, color=color, alpha=0.85, label=label)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.legend()
    ax.set_title(f"{title_suffix}: Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close(); print(f"[PLOT] {prefix}_percorruption.png")


def plot_comparison(small_results, tiny_results, save_dir):
    """Combined plot comparing both ViT variants."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("SES on Vision Transformers: ViT-Tiny vs ViT-Small (CIFAR-100)",
                 fontsize=14, fontweight="bold")

    all_labels = ["ViT-T Base", "ViT-T SES", "ViT-S Base", "ViT-S SES"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00"]
    all_results = [tiny_results["baseline"], tiny_results["ses"],
                   small_results["baseline"], small_results["ses"]]

    for idx, (title, ylabel, fn) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(r) for r in all_results]
        bars = ax.bar(range(4), vals, color=colors, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(all_labels, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "18_vit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 18_vit_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — SUBPROCESS ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def worker_main(experiment_name):
    """Entry point for subprocess workers."""
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    tag = gpu_tag()
    print(f"{tag} Worker '{experiment_name}' started — {name} ({vram:.1f} GB)")

    if experiment_name == "vit_small":
        results = experiment_vit_small(device)
        out_path = CFG.OUTPUT_DIR / "vit_small_results.json"
    elif experiment_name == "vit_tiny":
        results = experiment_vit_tiny(device)
        out_path = CFG.OUTPUT_DIR / "vit_tiny_results.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    """Main process: download data, launch workers, combine results, plot."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — ViT Experiments on Kaggle (Dual T4)                    ║")
    print("║   GPU 0: ViT-Small/4  ·  GPU 1: ViT-Tiny/4                   ║")
    print("║   Mixed Precision FP16  ·  SES regularizer in FP32             ║")
    print("║   AdamW + Cosine LR + Linear Warmup                           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    # Detect GPUs
    n_gpus = torch.cuda.device_count()
    print(f"\n[INFO] {n_gpus} GPU(s) detected:")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # Pre-download data
    print("\n[DATA] Pre-downloading CIFAR-100...")
    download_cifar100()

    script_path = os.path.abspath(__file__)

    if n_gpus >= 2:
        print(f"\n[PARALLEL] Launching 2 workers...")
        env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

        p0 = subprocess.Popen(
            [sys.executable, script_path, "--worker", "vit_small"],
            env=env0, stdout=sys.stdout, stderr=sys.stderr)
        p1 = subprocess.Popen(
            [sys.executable, script_path, "--worker", "vit_tiny"],
            env=env1, stdout=sys.stdout, stderr=sys.stderr)

        ret0 = p0.wait()
        ret1 = p1.wait()
        print(f"\n[PARALLEL] Workers finished (exit codes: vit_small={ret0}, vit_tiny={ret1})")
    else:
        print(f"\n[SEQUENTIAL] Only {n_gpus} GPU — running sequentially...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        worker_main("vit_small")
        worker_main("vit_tiny")

    # ── Load results ─────────────────────────────────────
    small_path = CFG.OUTPUT_DIR / "vit_small_results.json"
    tiny_path = CFG.OUTPUT_DIR / "vit_tiny_results.json"

    small_results = None
    tiny_results = None

    if small_path.exists():
        with open(small_path) as f:
            small_results = json.load(f)
        plot_vit_results(small_results, "ViT-Small/4", "18a_vit_small", CFG.OUTPUT_DIR)

    if tiny_path.exists():
        with open(tiny_path) as f:
            tiny_results = json.load(f)
        plot_vit_results(tiny_results, "ViT-Tiny/4", "18b_vit_tiny", CFG.OUTPUT_DIR)

    if small_results and tiny_results:
        plot_comparison(small_results, tiny_results, CFG.OUTPUT_DIR)

    # ── Summary JSON (without history) ───────────────────
    def strip_history(d):
        return {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                for k, v in d.items()}

    summary = {}
    if small_results:
        summary["vit_small_4"] = strip_history(small_results)
    if tiny_results:
        summary["vit_tiny_4"] = strip_history(tiny_results)

    summary_path = CFG.OUTPUT_DIR / "vit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"VIT EXPERIMENTS COMPLETE — {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"{'=' * 70}")

    for name, res in [("ViT-Small/4", small_results), ("ViT-Tiny/4", tiny_results)]:
        if res is None:
            continue
        print(f"\n┌─ {name} (CIFAR-100, FP16, AdamW) {'─' * (40 - len(name))}┐")
        for key, label in [("baseline", "Baseline"), ("ses", "SES")]:
            r = res[key]
            print(f"│  {label:<12} Acc: {r['best_acc']:.2f}%  Gap: {r['final_gap']:+.2f}pp  "
                  f"Rob: {r['rob_mean']:.2f}%  VRAM: {r['vram_peak_gb']:.1f}GB")
        d_acc = res["ses"]["best_acc"] - res["baseline"]["best_acc"]
        d_rob = res["ses"]["rob_mean"] - res["baseline"]["rob_mean"]
        print(f"│  Δ SES:      Acc {d_acc:+.2f}pp   Rob {d_rob:+.2f}pp")
        print(f"└{'─' * 62}┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")
    print(f"Summary: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES ViT Experiments")
    parser.add_argument("--worker", choices=["vit_small", "vit_tiny"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
