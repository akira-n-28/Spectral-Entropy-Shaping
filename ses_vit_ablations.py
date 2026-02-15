#!/usr/bin/env python3
"""
SES — Experiment 17: ViT Ablations (2x T4 16GB)
=================================================
β sweep + layer hooking ablation on ViT-Small/4.

Combined with Exp 14 (β=0.7, all 12 blocks), this gives:
  β sweep:        {0.3, 0.5, 0.7*, 0.9}       (* from Exp 14)
  Layer ablation:  {first-4, last-4, all-12*}    (* from Exp 14)

GPU 0: Baseline (no SES), β=0.5, Last-4 blocks
GPU 1: β=0.3, β=0.9, First-4 blocks

All on CIFAR-100, ViT-Small/4, AdamW, FP16, batch 512, 50 epochs, seed 42.

Usage:
  python ses_vit_ablations.py                    # Launches both workers
  python ses_vit_ablations.py --worker gpu0      # Internal: GPU0 worker
  python ses_vit_ablations.py --worker gpu1      # Internal: GPU1 worker
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
    OUTPUT_DIR = Path("/kaggle/working/outputs_vit_ablations") if Path("/kaggle").exists() else Path("./outputs_vit_ablations")
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
    """SES loss — always computed in float32 regardless of autocast."""
    anchor_device = activations[0].device
    reg = torch.tensor(0.0, device=anchor_device)
    for H in activations:
        if H.device != anchor_device:
            H = H.to(anchor_device)
        if H.dim() > 2:
            H = H.mean(dim=[2, 3]) if H.dim() == 4 else H.flatten(1)
        H = H.float()
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# VISION TRANSFORMER (identical to ses_vit_experiment.py)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
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
        self.drop_path_rate = drop_path

    def _drop_path(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self._drop_path(attn_out)
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
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
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def make_vit_small(num_classes=100):
    model = VisionTransformer(
        img_size=32, patch_size=4, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        drop_rate=0.0, drop_path_rate=0.1)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [ViT-Small/4] {nparams:.1f}M params, embed_dim=384, depth=12, heads=6")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# VIT ACTIVATION COLLECTOR (with flexible block selection)
# ══════════════════════════════════════════════════════════════════════════════

class ViTCollector:
    """
    Hooks on TransformerBlock outputs for SES.
    Pools over sequence dimension: [B, seq_len, embed_dim] -> [B, embed_dim].

    hook_mode:
        "all"      -> all 12 blocks (default)
        "first_4"  -> blocks 0-3
        "last_4"   -> blocks 8-11
        "middle_4" -> blocks 4-7
    """

    def __init__(self, model, hook_mode="all"):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []

        blocks = list(model.blocks)
        n_blocks = len(blocks)

        if hook_mode == "first_4":
            indices = list(range(min(4, n_blocks)))
        elif hook_mode == "last_4":
            indices = list(range(max(0, n_blocks - 4), n_blocks))
        elif hook_mode == "middle_4":
            mid = n_blocks // 2
            indices = list(range(max(0, mid - 2), min(n_blocks, mid + 2)))
        else:  # "all"
            indices = list(range(n_blocks))

        for idx in indices:
            self.hooks.append(blocks[idx].register_forward_hook(self._hook_fn))
            self.hook_names.append(f"block_{idx}")

        print(f"  [ViT Hooks] {len(self.hooks)} blocks ({hook_mode}): "
              f"{', '.join(self.hook_names)}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            pooled = output.mean(dim=1)  # [B, seq_len, D] -> [B, D]
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
# TRAINING — Mixed Precision FP16 with gradient clipping
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.7):
    model.train()
    loss_m = AverageMeter()
    correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if collector:
            collector.clear()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(X)
            task_loss = F.cross_entropy(output, y)

        reg_loss = torch.tensor(0.0, device=device)
        if collector and lambda_ses > 0 and collector.activations:
            reg_loss = lambda_ses * ses_regularizer(collector.activations, beta=beta)

        total_loss = task_loss.float() + reg_loss

        scaler.scale(total_loss).backward()
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
                      label, use_ses=False, beta=0.7):
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
                                     beta=beta)
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
        pbar.set_postfix_str(f"Te {test_res['acc']:.1f}% β={beta:.1f} Gap {gap:.1f} {epoch_time:.1f}s")

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
        "beta": beta,
        "history": dict(history),
    }
    print(f"{tag} {label} DONE: Acc {best_test_acc:.2f}% | Gap {gap:.2f}pp | "
          f"Rob {rob['mean_corruption_acc']:.2f}% | {np.mean(epoch_times):.1f}s/ep")

    del optimizer, scheduler, scaler
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def experiment_gpu0(device):
    """GPU 0: Baseline, β=0.5, Last-4 blocks."""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT Ablations — GPU 0")
    print(f"{tag} Baseline + β=0.5 + Last-4 blocks")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        # (key, use_ses, beta, hook_mode, label)
        ("baseline",   False, 0.7,  "all",    "Baseline (no SES)"),
        ("beta_0.5",   True,  0.5,  "all",    "SES β=0.5 (all 12)"),
        ("last_4",     True,  0.7,  "last_4", "SES β=0.7 (last 4)"),
    ]

    results = {}
    for key, use_ses, beta, hook_mode, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = ViTCollector(model, hook_mode=hook_mode) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, beta=beta)

        del model, collector
        torch.cuda.empty_cache()

    return results


def experiment_gpu1(device):
    """GPU 1: β=0.3, β=0.9, First-4 blocks."""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT Ablations — GPU 1")
    print(f"{tag} β=0.3 + β=0.9 + First-4 blocks")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("beta_0.3",   True, 0.3,  "all",      "SES β=0.3 (all 12)"),
        ("beta_0.9",   True, 0.9,  "all",      "SES β=0.9 (all 12)"),
        ("first_4",    True, 0.7,  "first_4",  "SES β=0.7 (first 4)"),
    ]

    results = {}
    for key, use_ses, beta, hook_mode, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = ViTCollector(model, hook_mode=hook_mode) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, beta=beta)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_beta_sweep(results, save_dir):
    """Plot β sweep: baseline, β={0.3, 0.5, 0.7*, 0.9}.
    * β=0.7 with all-12 hooks is not in this experiment (from Exp 14).
    """
    # β sweep configs present in results
    order = ["baseline", "beta_0.3", "beta_0.5", "beta_0.9"]
    labels = ["Baseline\n(no SES)", "β=0.3\n(all 12)", "β=0.5\n(all 12)", "β=0.9\n(all 12)"]
    colors = ["#377eb8", "#984ea3", "#4daf4a", "#ff7f00"]

    present = [k for k in order if k in results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exp 17a: ViT-Small/4 β Sweep (CIFAR-100, FP16, AdamW)",
                 fontsize=14, fontweight="bold")

    for idx, (title, ylabel, fn) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in present]
        bars = ax.bar(range(len(present)), vals, color=present_colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            offset = 0.15 if val >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present_labels, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "21a_vit_beta_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 21a_vit_beta_sweep.png")


def plot_layer_ablation(results, save_dir):
    """Plot layer hooking ablation: baseline, first-4, last-4.
    All-12 is from Exp 14 (not repeated here).
    """
    order = ["baseline", "first_4", "last_4"]
    labels = ["Baseline\n(no SES)", "First 4\n(blocks 0-3)", "Last 4\n(blocks 8-11)"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a"]

    present = [k for k in order if k in results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exp 17b: ViT-Small/4 Layer Hooking Ablation (CIFAR-100, β=0.7)",
                 fontsize=14, fontweight="bold")

    for idx, (title, ylabel, fn) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in present]
        bars = ax.bar(range(len(present)), vals, color=present_colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            offset = 0.15 if val >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present_labels, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "21b_vit_layer_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 21b_vit_layer_ablation.png")


def plot_all_configs(results, save_dir):
    """Plot all 6 configs together + accuracy curves."""
    order = ["baseline", "beta_0.3", "beta_0.5", "beta_0.9", "first_4", "last_4"]
    labels = ["Baseline", "β=0.3", "β=0.5", "β=0.9", "First-4", "Last-4"]
    colors = ["#377eb8", "#984ea3", "#4daf4a", "#ff7f00", "#e41a1c", "#a65628"]

    present = [k for k in order if k in results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    # ═══════ Bar charts ═══════
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Exp 17: ViT-Small/4 Ablations — All Configs (CIFAR-100, FP16)",
                 fontsize=14, fontweight="bold")

    for idx, (title, ylabel, fn) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in present]
        bars = ax.bar(range(len(present)), vals, color=present_colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            offset = 0.15 if val >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "21_vit_ablations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 21_vit_ablations.png")

    # ═══════ Accuracy curves ═══════
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    for key in present:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)]
        ax.plot(h["test_acc"], color=c, lw=2,
                label=f"{lbl} ({results[key]['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy Curves")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key in present:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)]
        ax.plot(h["train_acc"], color=c, ls="--", alpha=0.5)
        ax.plot(h["test_acc"], color=c, lw=2, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train (dashed) vs Test (solid)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "21c_vit_ablations_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 21c_vit_ablations_curves.png")

    # ═══════ Per-corruption ═══════
    fig, ax = plt.subplots(figsize=(20, 6))
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    severities = [1, 3, 5]
    xlabels = [f"{c.replace('_',' ')}\ns{s}" for c in corruptions for s in severities]
    x = np.arange(len(xlabels))
    n = len(present)
    w = 0.8 / n

    for i, key in enumerate(present):
        rob = results[key]["robustness_detail"]
        vals = [rob[f"{c}_s{s}"] for c in corruptions for s in severities]
        ax.bar(x + (i - n/2 + 0.5) * w, vals, w,
               color=present_colors[i], alpha=0.85,
               label=present_labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("ViT-Small/4 Ablations: Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "21d_vit_ablations_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 21d_vit_ablations_percorruption.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — SUBPROCESS ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def worker_main(experiment_name):
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    tag = gpu_tag()
    print(f"{tag} Worker '{experiment_name}' started — {name} ({vram:.1f} GB)")

    if experiment_name == "gpu0":
        results = experiment_gpu0(device)
        out_path = CFG.OUTPUT_DIR / "gpu0_vit_ablations.json"
    elif experiment_name == "gpu1":
        results = experiment_gpu1(device)
        out_path = CFG.OUTPUT_DIR / "gpu1_vit_ablations.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Experiment 17: ViT-Small/4 Ablations                   ║")
    print("║   GPU 0: Baseline + β=0.5 + Last-4 blocks                     ║")
    print("║   GPU 1: β=0.3 + β=0.9 + First-4 blocks                      ║")
    print("║   CIFAR-100, AdamW, FP16, batch 512, 50 epochs                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"\n[INFO] {n_gpus} GPU(s) detected:")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    print("\n[DATA] Pre-downloading CIFAR-100...")
    download_cifar100()

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
    gpu0_path = CFG.OUTPUT_DIR / "gpu0_vit_ablations.json"
    gpu1_path = CFG.OUTPUT_DIR / "gpu1_vit_ablations.json"

    all_results = {}
    for path in [gpu0_path, gpu1_path]:
        if path.exists():
            with open(path) as f:
                all_results.update(json.load(f))

    if all_results:
        combined_path = CFG.OUTPUT_DIR / "vit_ablations_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        summary = {}
        for k, v in all_results.items():
            summary[k] = {kk: vv for kk, vv in v.items() if kk != "history"}
        summary_path = CFG.OUTPUT_DIR / "vit_ablations_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        plot_beta_sweep(all_results, CFG.OUTPUT_DIR)
        plot_layer_ablation(all_results, CFG.OUTPUT_DIR)
        plot_all_configs(all_results, CFG.OUTPUT_DIR)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 17: VIT ABLATIONS COMPLETE — {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    if all_results:
        print(f"\n┌─ VIT-SMALL/4 ABLATIONS (CIFAR-100, FP16, AdamW) ────────────────┐")

        nice_names = {
            "baseline": "Baseline (no SES)",
            "beta_0.3": "SES β=0.3 (all 12)",
            "beta_0.5": "SES β=0.5 (all 12)",
            "beta_0.9": "SES β=0.9 (all 12)",
            "first_4":  "SES β=0.7 (first 4)",
            "last_4":   "SES β=0.7 (last 4)",
        }
        order = ["baseline", "beta_0.3", "beta_0.5", "beta_0.9", "first_4", "last_4"]

        for key in order:
            if key in all_results:
                r = all_results[key]
                name = nice_names[key]
                print(f"│  {name:<26} Acc: {r['best_acc']:.2f}%  "
                      f"Gap: {r['final_gap']:+.2f}pp  Rob: {r['rob_mean']:.2f}%  "
                      f"Time: {r['avg_epoch_time']:.1f}s/ep")

        # Δ vs Baseline
        if "baseline" in all_results:
            base = all_results["baseline"]
            print(f"│")
            print(f"│  Δ vs Baseline:")
            for key in order:
                if key in all_results and key != "baseline":
                    r = all_results[key]
                    d_a = r["best_acc"] - base["best_acc"]
                    d_r = r["rob_mean"] - base["rob_mean"]
                    d_g = r["final_gap"] - base["final_gap"]
                    name = nice_names[key]
                    print(f"│    {name:<26} Acc {d_a:+.2f}pp  "
                          f"Gap {d_g:+.2f}pp  Rob {d_r:+.2f}pp")

        print(f"│")
        print(f"│  Note: β=0.7 with all 12 blocks is from Exp 14:")
        print(f"│    ViT-Small Baseline: 55.28%  SES β=0.7: 57.01% (+1.73pp)")
        print(f"└──────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Exp 17: ViT Ablations")
    parser.add_argument("--worker", choices=["gpu0", "gpu1"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
