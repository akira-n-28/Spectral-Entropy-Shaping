#!/usr/bin/env python3
"""
SES — Exp 15: ViT + RandAugment + Mixup/CutMix (2x T4 16GB)
=============================================================
Tests SES orthogonality with modern augmentation on Vision Transformers.
All configs use RandAugment as baseline augmentation (standard for ViTs).

GPU 0 (3 configs): RandAug, RandAug+SES, RandAug+Mixup
GPU 1 (3 configs): RandAug+SES+Mixup, RandAug+CutMix, RandAug+SES+CutMix

Mixed Precision FP16 + SES regularizer in FP32.
True parallelism via subprocess (not threading).

Usage:
  python ses_vit_augmentation.py                   # Launches both workers
  python ses_vit_augmentation.py --worker gpu0     # Internal: 3 configs on GPU 0
  python ses_vit_augmentation.py --worker gpu1     # Internal: 3 configs on GPU 1
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
    OUTPUT_DIR = Path("/kaggle/working/outputs_vit_aug") if Path("/kaggle").exists() else Path("./outputs_vit_aug")
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

    # RandAugment
    RANDAUG_N = 2       # number of transforms
    RANDAUG_M = 9       # magnitude


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
# VISION TRANSFORMER (same as ses_vit_experiment.py)
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
    print(f"  [ViT-Small/4] {nparams:.1f}M params")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# VIT ACTIVATION COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class ViTCollector:
    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []
        for idx, block in enumerate(model.blocks):
            self.hooks.append(block.register_forward_hook(self._hook_fn))
            self.hook_names.append(f"block_{idx}")
        print(f"  [ViT Hooks] {len(self.hooks)} blocks")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            pooled = output.mean(dim=1)  # [B, seq_len, D] → [B, D]
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
# DATA — with RandAugment
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

    # RandAugment pipeline (standard for ViTs on small datasets)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=CFG.RANDAUG_N, magnitude=CFG.RANDAUG_M),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
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
# MIXUP / CUTMIX
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x_cut, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.7, aug_type="none"):
    model.train()
    loss_m = AverageMeter()
    correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if collector:
            collector.clear()

        # Mixup / CutMix (applied on top of RandAugment from dataloader)
        if aug_type == "mixup":
            X, y_a, y_b, lam = mixup_data(X, y)
        elif aug_type == "cutmix":
            X, y_a, y_b, lam = cutmix_data(X, y)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(X)
            if aug_type in ("mixup", "cutmix"):
                task_loss = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam)
            else:
                task_loss = F.cross_entropy(output, y)

        # SES in FP32
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
        if aug_type in ("mixup", "cutmix"):
            correct += (lam * (output.argmax(1) == y_a).float()
                        + (1 - lam) * (output.argmax(1) == y_b).float()).sum().item()
        else:
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
                      label, use_ses=False, aug_type="none"):
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
                                     beta=CFG.BETA, aug_type=aug_type)
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
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def experiment_gpu0(device):
    """GPU 0: RandAug, RandAug+SES, RandAug+Mixup"""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT-Small/4 + RandAugment on CIFAR-100")
    print(f"{tag} Configs: RA, RA+SES, RA+Mixup")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("randaug",          False, "none",  "RA"),
        ("randaug_ses",      True,  "none",  "RA+SES"),
        ("randaug_mixup",    False, "mixup", "RA+Mixup"),
    ]

    results = {}
    for key, use_ses, aug_type, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = ViTCollector(model) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, aug_type=aug_type)

        del model, collector
        torch.cuda.empty_cache()

    return results


def experiment_gpu1(device):
    """GPU 1: RandAug+SES+Mixup, RandAug+CutMix, RandAug+SES+CutMix"""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT-Small/4 + RandAugment on CIFAR-100")
    print(f"{tag} Configs: RA+SES+Mixup, RA+CutMix, RA+SES+CutMix")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("randaug_ses_mixup",    True,  "mixup",  "RA+SES+Mixup"),
        ("randaug_cutmix",       False, "cutmix", "RA+CutMix"),
        ("randaug_ses_cutmix",   True,  "cutmix", "RA+SES+CutMix"),
    ]

    results = {}
    for key, use_ses, aug_type, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = ViTCollector(model) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, aug_type=aug_type)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results, save_dir):
    """Combined 6-config bar chart + orthogonality analysis."""
    methods = ["RA", "RA+SES", "RA+Mixup", "RA+SES+Mixup", "RA+CutMix", "RA+SES+CutMix"]
    keys = ["randaug", "randaug_ses", "randaug_mixup", "randaug_ses_mixup",
            "randaug_cutmix", "randaug_ses_cutmix"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("ViT-Small/4 + RandAugment + SES/Mixup/CutMix (CIFAR-100, FP16)",
                 fontsize=14, fontweight="bold")
    for idx, (title, ylabel, fn) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in keys]
        bars = ax.bar(range(len(methods)), vals, color=colors, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "19_vit_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 19_vit_augmentation.png")

    # Training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color, label in zip(keys, colors, methods):
        h = results[key]["history"]
        ax.plot(h["test_acc"], color=color, lw=2,
                label=f"{label} ({results[key]['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("ViT-Small/4 + RandAugment: Test Accuracy Curves")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "19b_vit_augmentation_curves.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 19b_vit_augmentation_curves.png")

    # Per-corruption
    fig, ax = plt.subplots(figsize=(20, 6))
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    severities = [1, 3, 5]
    xlabels = [f"{c.replace('_',' ')}\ns{s}" for c in corruptions for s in severities]
    x = np.arange(len(xlabels)); w = 0.13
    for i, (key, color, label) in enumerate(zip(keys, colors, methods)):
        rob = results[key]["robustness_detail"]
        vals = [rob[f"{c}_s{s}"] for c in corruptions for s in severities]
        ax.bar(x + (i - 2.5) * w, vals, w, color=color, alpha=0.85, label=label)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=7, ncol=3)
    ax.set_title("ViT-Small/4 + RandAugment: Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "19c_vit_augmentation_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 19c_vit_augmentation_percorruption.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — SUBPROCESS ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def worker_main(worker_name):
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    tag = gpu_tag()
    print(f"{tag} Worker '{worker_name}' started — {name} ({vram:.1f} GB)")

    if worker_name == "gpu0":
        results = experiment_gpu0(device)
        out_path = CFG.OUTPUT_DIR / "gpu0_results.json"
    elif worker_name == "gpu1":
        results = experiment_gpu1(device)
        out_path = CFG.OUTPUT_DIR / "gpu1_results.json"
    else:
        raise ValueError(f"Unknown worker: {worker_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Exp 15: ViT + RandAugment + SES/Mixup/CutMix         ║")
    print("║   GPU 0: RA, RA+SES, RA+Mixup                                 ║")
    print("║   GPU 1: RA+SES+Mixup, RA+CutMix, RA+SES+CutMix             ║")
    print("║   ViT-Small/4, FP16, AdamW + Cosine + Warmup                  ║")
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
        print(f"\n[PARALLEL] Launching 2 workers (3 configs each)...")
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

    # ── Load and merge results ───────────────────────────
    gpu0_path = CFG.OUTPUT_DIR / "gpu0_results.json"
    gpu1_path = CFG.OUTPUT_DIR / "gpu1_results.json"

    all_results = {}

    if gpu0_path.exists():
        with open(gpu0_path) as f:
            all_results.update(json.load(f))
    if gpu1_path.exists():
        with open(gpu1_path) as f:
            all_results.update(json.load(f))

    if len(all_results) == 6:
        plot_results(all_results, CFG.OUTPUT_DIR)

    # ── Summary JSON (without history) ───────────────────
    def strip_history(d):
        return {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                for k, v in d.items()}

    summary = strip_history(all_results) if all_results else {}
    summary_path = CFG.OUTPUT_DIR / "vit_aug_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Save merged results too ──────────────────────────
    merged_path = CFG.OUTPUT_DIR / "vit_aug_results.json"
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXP 15 COMPLETE — {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"{'=' * 70}")

    if len(all_results) >= 6:
        print(f"\n┌─ ViT-Small/4 + RandAugment (CIFAR-100, FP16, AdamW) ────────────┐")
        for key, label in [
            ("randaug", "RA"),
            ("randaug_ses", "RA+SES"),
            ("randaug_mixup", "RA+Mixup"),
            ("randaug_ses_mixup", "RA+SES+Mix"),
            ("randaug_cutmix", "RA+CutMix"),
            ("randaug_ses_cutmix", "RA+SES+Cut"),
        ]:
            r = all_results[key]
            print(f"│  {label:<14} Acc: {r['best_acc']:.2f}%  Gap: {r['final_gap']:+.2f}pp  "
                  f"Rob: {r['rob_mean']:.2f}%  {r['avg_epoch_time']:.1f}s/ep")

        print(f"│")
        print(f"│  SES additivity (over RandAugment baseline):")
        for base_key, ses_key, name in [
            ("randaug", "randaug_ses", "RA→RA+SES"),
            ("randaug_mixup", "randaug_ses_mixup", "RA+Mix→RA+SES+Mix"),
            ("randaug_cutmix", "randaug_ses_cutmix", "RA+Cut→RA+SES+Cut"),
        ]:
            d_a = all_results[ses_key]["best_acc"] - all_results[base_key]["best_acc"]
            d_r = all_results[ses_key]["rob_mean"] - all_results[base_key]["rob_mean"]
            print(f"│    {name:<25} Acc {d_a:+.2f}pp  Rob {d_r:+.2f}pp")
        print(f"└──────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")
    print(f"Summary: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Exp 15: ViT + Augmentation")
    parser.add_argument("--worker", choices=["gpu0", "gpu1"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
