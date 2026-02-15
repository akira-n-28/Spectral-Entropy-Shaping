#!/usr/bin/env python3
"""
SES — Experiment 18: SES on Attention Matrices (2x T4 16GB)
============================================================
Applies SES to attention weight matrices and sub-layer outputs,
not just block outputs. Tests whether controlling the *mechanism*
(attention diversity) helps beyond controlling the *representation*.

GPU 0 (3 configs):
  1. block_l4     — Block output SES, β=0.5, Last-4  (best combo from Exp 17)
  2. attn_out_l4  — Attention sub-layer output SES, β=0.5, Last-4
  3. attn_wt_l4   — Attention weight matrix SES, β=0.5, Last-4

GPU 1 (3 configs):
  4. dual_l4      — Block output + Attention weight SES, β=0.5, Last-4
  5. attn_wt_all  — Attention weight matrix SES, β=0.5, All-12
  6. triple_l4    — Block + Attn output + Attn weight SES, β=0.5, Last-4

Reference baseline from Exp 17: 55.28% (ViT-Small/4, CIFAR-100).
Reference block_l4 is also the first test of the Exp 17 optimal combo
(β=0.5 + Last-4 were tested separately, never together).

Setup: ViT-Small/4 (22M), CIFAR-100, AdamW lr=1e-3, cosine+warmup,
FP16, batch 512, seed 42, 50 epochs.

Usage:
  python ses_vit_attention.py                    # Launches both workers
  python ses_vit_attention.py --worker gpu0      # Internal: GPU0 worker
  python ses_vit_attention.py --worker gpu1      # Internal: GPU1 worker
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
    OUTPUT_DIR = Path("/kaggle/working/outputs_vit_attn") if Path("/kaggle").exists() else Path("./outputs_vit_attn")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    EPOCHS = 50
    BATCH_SIZE = 512
    NUM_WORKERS = 2

    LR = 1e-3
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5

    LAMBDA_SES = 0.01
    BETA = 0.5       # optimal for ViT (from Exp 17)
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


def ses_regularizer(activations, beta=0.5, eps=1e-12):
    """SES loss — always computed in float32."""
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
# VISION TRANSFORMER (modified to expose attention info)
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
    """
    TransformerBlock with optional attention info storage.

    When store_attn=True, stores:
      - _attn_output: attention sub-layer output [B, seq, D] (before residual)
      - _attn_weights: per-head attention weights [B, heads, seq, seq]
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.num_heads = num_heads
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

        # Attention info storage
        self.store_attn = False
        self._attn_output = None
        self._attn_weights = None

    def _drop_path(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob

    def forward(self, x):
        normed = self.norm1(x)

        if self.store_attn:
            attn_out, attn_w = self.attn(
                normed, normed, normed,
                need_weights=True, average_attn_weights=False)
            self._attn_output = attn_out    # [B, seq, D]
            self._attn_weights = attn_w     # [B, heads, seq, seq]
        else:
            attn_out, _ = self.attn(normed, normed, normed)
            self._attn_output = None
            self._attn_weights = None

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
# COLLECTORS (block output, attention output, attention weights, combined)
# ══════════════════════════════════════════════════════════════════════════════

class BlockOutputCollector:
    """Standard: hook on TransformerBlock output, mean-pool seq dim."""

    def __init__(self, model, hook_mode="last_4"):
        self.activations = []
        self.hooks = []
        self.enabled = True
        blocks = list(model.blocks)
        n = len(blocks)
        indices = list(range(max(0, n - 4), n)) if hook_mode == "last_4" else list(range(n))
        for idx in indices:
            self.hooks.append(blocks[idx].register_forward_hook(self._hook_fn))
        print(f"  [BlockOutput] {len(self.hooks)} hooks ({hook_mode}): blocks {indices}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            self.activations.append(output.mean(dim=1))  # [B, seq, D] -> [B, D]

    def clear(self):
        self.activations = []
    def disable(self):
        self.enabled = False; self.activations = []
    def enable(self):
        self.enabled = True
    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks = []


class AttentionOutputCollector:
    """Hook on attention sub-layer output (before MLP, before residual)."""

    def __init__(self, model, hook_mode="last_4"):
        self.activations = []
        self.enabled = True
        self.target_blocks = []

        blocks = list(model.blocks)
        n = len(blocks)
        indices = list(range(max(0, n - 4), n)) if hook_mode == "last_4" else list(range(n))
        for idx in indices:
            blocks[idx].store_attn = True
            self.target_blocks.append(blocks[idx])
        print(f"  [AttnOutput] {len(indices)} blocks ({hook_mode}): blocks {indices}")

    def collect(self):
        """Call after forward pass to gather attention sub-layer outputs."""
        if not self.enabled:
            return
        for block in self.target_blocks:
            if block._attn_output is not None:
                # [B, seq, D] -> mean-pool -> [B, D]
                pooled = block._attn_output.mean(dim=1)
                self.activations.append(pooled)

    def clear(self):
        self.activations = []
    def disable(self):
        self.enabled = False; self.activations = []
        for b in self.target_blocks:
            b.store_attn = False
    def enable(self):
        self.enabled = True
        for b in self.target_blocks:
            b.store_attn = True
    def remove(self):
        for b in self.target_blocks:
            b.store_attn = False
        self.target_blocks = []


class AttentionWeightCollector:
    """
    Hook on attention weight matrices.
    Per-head mean attention pattern: [B, heads, seq, seq] ->
      mean over queries -> [B, heads, seq] -> flatten -> [B, heads*seq]
    This captures "batch-level diversity of attention patterns".
    """

    def __init__(self, model, hook_mode="last_4"):
        self.activations = []
        self.enabled = True
        self.target_blocks = []

        blocks = list(model.blocks)
        n = len(blocks)
        indices = list(range(max(0, n - 4), n)) if hook_mode == "last_4" else list(range(n))
        for idx in indices:
            blocks[idx].store_attn = True
            self.target_blocks.append(blocks[idx])

        # Compute expected dim
        seq_len = (32 // 4) ** 2 + 1  # 65 for CIFAR-32 with patch_size=4
        num_heads = model.blocks[0].num_heads
        self.expected_dim = num_heads * seq_len  # 6 * 65 = 390
        print(f"  [AttnWeights] {len(indices)} blocks ({hook_mode}): "
              f"blocks {indices}, dim={self.expected_dim} ({num_heads}h × {seq_len}seq)")

    def collect(self):
        """Call after forward pass to gather attention weight features."""
        if not self.enabled:
            return
        for block in self.target_blocks:
            if block._attn_weights is not None:
                # [B, heads, seq, seq] -> mean over query dim -> [B, heads, seq]
                attn_w = block._attn_weights
                mean_attn = attn_w.mean(dim=2)  # [B, heads, seq]
                flat = mean_attn.flatten(1)       # [B, heads * seq]
                self.activations.append(flat)

    def clear(self):
        self.activations = []
    def disable(self):
        self.enabled = False; self.activations = []
        for b in self.target_blocks:
            b.store_attn = False
    def enable(self):
        self.enabled = True
        for b in self.target_blocks:
            b.store_attn = True
    def remove(self):
        for b in self.target_blocks:
            b.store_attn = False
        self.target_blocks = []


class CombinedCollector:
    """Combines multiple collectors into one. All activations pooled for SES."""

    def __init__(self, collectors):
        self.collectors = collectors
        self.activations = []
        self.enabled = True

    @property
    def activations(self):
        all_acts = []
        for c in self.collectors:
            all_acts.extend(c.activations)
        return all_acts

    @activations.setter
    def activations(self, val):
        pass  # activations are computed on-the-fly

    def collect(self):
        for c in self.collectors:
            if hasattr(c, 'collect'):
                c.collect()

    def clear(self):
        for c in self.collectors:
            c.clear()

    def disable(self):
        self.enabled = False
        for c in self.collectors:
            c.disable()

    def enable(self):
        self.enabled = True
        for c in self.collectors:
            c.enable()

    def remove(self):
        for c in self.collectors:
            c.remove()


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
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.5):
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

        # Collect attention-based activations (if collector needs post-forward gather)
        if collector and hasattr(collector, 'collect'):
            collector.collect()

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
                      label, beta=0.5):
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
                                     lambda_ses=CFG.LAMBDA_SES,
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
        pbar.set_postfix_str(f"Te {test_res['acc']:.1f}% Gap {gap:.1f} {epoch_time:.1f}s")

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
# COLLECTOR FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_collector(model, mode, hook_mode="last_4"):
    """
    Create a collector for the specified mode.

    Modes:
      "block"      — Block output SES (standard)
      "attn_out"   — Attention sub-layer output SES
      "attn_wt"    — Attention weight matrix SES
      "dual"       — Block output + Attention weight SES
      "triple"     — Block output + Attn sub-layer output + Attention weight SES
    """
    if mode == "block":
        return BlockOutputCollector(model, hook_mode=hook_mode)
    elif mode == "attn_out":
        return AttentionOutputCollector(model, hook_mode=hook_mode)
    elif mode == "attn_wt":
        return AttentionWeightCollector(model, hook_mode=hook_mode)
    elif mode == "dual":
        c1 = BlockOutputCollector(model, hook_mode=hook_mode)
        c2 = AttentionWeightCollector(model, hook_mode=hook_mode)
        return CombinedCollector([c1, c2])
    elif mode == "triple":
        c1 = BlockOutputCollector(model, hook_mode=hook_mode)
        c2 = AttentionOutputCollector(model, hook_mode=hook_mode)
        c3 = AttentionWeightCollector(model, hook_mode=hook_mode)
        return CombinedCollector([c1, c2, c3])
    else:
        raise ValueError(f"Unknown collector mode: {mode}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def experiment_gpu0(device):
    """GPU 0: block_l4, attn_out_l4, attn_wt_l4."""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT Attention SES — GPU 0")
    print(f"{tag} Block output / Attn output / Attn weights (Last-4)")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        # (key, collector_mode, hook_mode, label)
        ("block_l4",    "block",    "last_4", "Block output β=0.5 L4"),
        ("attn_out_l4", "attn_out", "last_4", "Attn sub-layer β=0.5 L4"),
        ("attn_wt_l4",  "attn_wt",  "last_4", "Attn weights β=0.5 L4"),
    ]

    results = {}
    for key, coll_mode, hook_mode, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = make_collector(model, coll_mode, hook_mode)

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, beta=CFG.BETA)

        del model, collector
        torch.cuda.empty_cache()

    return results


def experiment_gpu1(device):
    """GPU 1: dual_l4, attn_wt_all, triple_l4."""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} ViT Attention SES — GPU 1")
    print(f"{tag} Dual / Attn weights All-12 / Triple (Last-4)")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("dual_l4",     "dual",    "last_4", "Block+AttnWt β=0.5 L4"),
        ("attn_wt_all", "attn_wt", "all",    "Attn weights β=0.5 All12"),
        ("triple_l4",   "triple",  "last_4", "Block+AttnOut+AttnWt β=0.5 L4"),
    ]

    results = {}
    for key, coll_mode, hook_mode, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_vit_small(num_classes=100).to(device)
        collector = make_collector(model, coll_mode, hook_mode)

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, beta=CFG.BETA)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_ACC = 55.28
BASELINE_ROB = 34.75
BASELINE_GAP = 44.50

def plot_all_configs(results, save_dir):
    order = ["block_l4", "attn_out_l4", "attn_wt_l4", "dual_l4", "attn_wt_all", "triple_l4"]
    labels = [
        "Block\nLast-4", "Attn Output\nLast-4", "Attn Weights\nLast-4",
        "Block+AttnWt\nLast-4", "Attn Weights\nAll-12", "Triple\nLast-4"
    ]
    colors = ["#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#e41a1c", "#a65628"]

    present = [k for k in order if k in results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    # ═══════ Bar charts with baseline reference ═══════
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Exp 18: SES on Attention Matrices — ViT-Small/4 (CIFAR-100, β=0.5)",
                 fontsize=14, fontweight="bold")

    for idx, (title, ylabel, fn, baseline_val) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"], BASELINE_ACC),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"], BASELINE_GAP),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"], BASELINE_ROB),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in present]
        bars = ax.bar(range(len(present)), vals, color=present_colors, alpha=0.85)
        ax.axhline(baseline_val, color="gray", ls="--", lw=1.5, label=f"Baseline: {baseline_val:.2f}")
        for bar, val in zip(bars, vals):
            offset = 0.15 if val >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present_labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_dir / "22_vit_attention_ses.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 22_vit_attention_ses.png")

    # ═══════ Accuracy curves ═══════
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    for key in present:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)].replace('\n', ' ')
        ax.plot(h["test_acc"], color=c, lw=2,
                label=f"{lbl} ({results[key]['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy Curves")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key in present:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)].replace('\n', ' ')
        ax.plot(h["train_acc"], color=c, ls="--", alpha=0.5)
        ax.plot(h["test_acc"], color=c, lw=2, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train (dashed) vs Test (solid)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "22b_vit_attention_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 22b_vit_attention_curves.png")

    # ═══════ Δ vs Baseline ═══════
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(present))
    w = 0.25
    d_acc = [results[k]["best_acc"] - BASELINE_ACC for k in present]
    d_rob = [results[k]["rob_mean"] - BASELINE_ROB for k in present]
    d_gap = [results[k]["final_gap"] - BASELINE_GAP for k in present]

    ax.bar(x - w, d_acc, w, color="#4daf4a", alpha=0.85, label="Δ Acc (pp)")
    ax.bar(x,     d_rob, w, color="#377eb8", alpha=0.85, label="Δ Rob (pp)")
    ax.bar(x + w, d_gap, w, color="#e41a1c", alpha=0.85, label="Δ Gap (pp)")
    ax.axhline(0, color="black", lw=0.8)

    for i in range(len(present)):
        ax.text(x[i] - w, d_acc[i] + 0.05, f"{d_acc[i]:+.2f}", ha="center", fontsize=7)
        ax.text(x[i],     d_rob[i] + 0.05, f"{d_rob[i]:+.2f}", ha="center", fontsize=7)
        ax.text(x[i] + w, d_gap[i] - 0.15, f"{d_gap[i]:+.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('\n', ' ') for l in present_labels], fontsize=8)
    ax.set_ylabel("Δ vs Baseline (pp)")
    ax.set_title("Exp 18: Δ vs No-SES Baseline (Acc↑, Rob↑, Gap↓ = better)", fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "22c_vit_attention_delta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 22c_vit_attention_delta.png")

    # ═══════ Per-corruption ═══════
    fig, ax = plt.subplots(figsize=(20, 6))
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    severities = [1, 3, 5]
    xlabels = [f"{c.replace('_',' ')}\ns{s}" for c in corruptions for s in severities]
    xpos = np.arange(len(xlabels))
    n = len(present)
    bw = 0.8 / n

    for i, key in enumerate(present):
        rob = results[key]["robustness_detail"]
        vals = [rob[f"{c}_s{s}"] for c in corruptions for s in severities]
        ax.bar(xpos + (i - n/2 + 0.5) * bw, vals, bw,
               color=present_colors[i], alpha=0.85,
               label=present_labels[i].replace('\n', ' '))

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "22d_vit_attention_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 22d_vit_attention_percorruption.png")


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
        out_path = CFG.OUTPUT_DIR / "gpu0_attn_results.json"
    elif experiment_name == "gpu1":
        results = experiment_gpu1(device)
        out_path = CFG.OUTPUT_DIR / "gpu1_attn_results.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Experiment 18: SES on Attention Matrices               ║")
    print("║   GPU 0: Block L4 / Attn Output L4 / Attn Weights L4          ║")
    print("║   GPU 1: Dual L4 / Attn Weights All-12 / Triple L4            ║")
    print("║   ViT-Small/4, CIFAR-100, β=0.5, AdamW, FP16, 50 epochs       ║")
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
    gpu0_path = CFG.OUTPUT_DIR / "gpu0_attn_results.json"
    gpu1_path = CFG.OUTPUT_DIR / "gpu1_attn_results.json"

    all_results = {}
    for path in [gpu0_path, gpu1_path]:
        if path.exists():
            with open(path) as f:
                all_results.update(json.load(f))

    if all_results:
        combined_path = CFG.OUTPUT_DIR / "vit_attention_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        summary = {}
        for k, v in all_results.items():
            summary[k] = {kk: vv for kk, vv in v.items() if kk != "history"}
        summary_path = CFG.OUTPUT_DIR / "vit_attention_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        plot_all_configs(all_results, CFG.OUTPUT_DIR)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 18: VIT ATTENTION SES COMPLETE — {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    if all_results:
        print(f"\n┌─ VIT-SMALL/4 ATTENTION SES (CIFAR-100, β=0.5, FP16) ────────────┐")
        print(f"│  Baseline (Exp 17):            Acc: {BASELINE_ACC:.2f}%  Rob: {BASELINE_ROB:.2f}%")
        print(f"│")

        nice_names = {
            "block_l4":    "Block output (L4)",
            "attn_out_l4": "Attn sub-layer (L4)",
            "attn_wt_l4":  "Attn weights (L4)",
            "dual_l4":     "Block + AttnWt (L4)",
            "attn_wt_all": "Attn weights (All12)",
            "triple_l4":   "Triple (L4)",
        }
        order = ["block_l4", "attn_out_l4", "attn_wt_l4", "dual_l4", "attn_wt_all", "triple_l4"]

        for key in order:
            if key in all_results:
                r = all_results[key]
                name = nice_names[key]
                d_a = r["best_acc"] - BASELINE_ACC
                d_r = r["rob_mean"] - BASELINE_ROB
                print(f"│  {name:<26} Acc: {r['best_acc']:.2f}% ({d_a:+.2f})  "
                      f"Rob: {r['rob_mean']:.2f}% ({d_r:+.2f})  "
                      f"Time: {r['avg_epoch_time']:.1f}s/ep")

        print(f"└──────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Exp 18: ViT Attention SES")
    parser.add_argument("--worker", choices=["gpu0", "gpu1"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
