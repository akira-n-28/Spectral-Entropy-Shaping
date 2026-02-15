#!/usr/bin/env python3
"""
SES — Experiment 16: Adaptive β Scheduling (2x T4 16GB)
========================================================
Tests whether scheduling β over training improves upon fixed β=0.7.

Hypothesis: Start with low β (strong compression, fast convergence) and
gradually increase it (more expressivity as the network refines representations).

GPU 0: Baseline (no SES), Fixed β=0.7, Linear warmup β 0.3→0.9
GPU 1: Cosine warmup β 0.3→0.9, Reverse β 0.9→0.3, Step β 0.3→0.7→0.9

All on CIFAR-100, ResNet-18, SGD, FP16, batch 512, 50 epochs, seed 42.

Usage:
  python ses_adaptive_beta.py                    # Launches both workers
  python ses_adaptive_beta.py --worker gpu0      # Internal: GPU0 worker
  python ses_adaptive_beta.py --worker gpu1      # Internal: GPU1 worker
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
import torchvision.models as models
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    OUTPUT_DIR = Path("/kaggle/working/outputs_adaptive_beta") if Path("/kaggle").exists() else Path("./outputs_adaptive_beta")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    EPOCHS = 50
    BATCH_SIZE = 512
    LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    LR_MILESTONES = [20, 35, 45]
    LR_GAMMA = 0.2
    NUM_WORKERS = 2

    LAMBDA_SES = 0.01
    BETA = 0.7  # default fixed β
    SEED = 42


CFG = Config()
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# β SCHEDULES
# ══════════════════════════════════════════════════════════════════════════════

def beta_fixed(epoch, total_epochs, beta=0.7):
    """Constant β (current default)."""
    return beta


def beta_linear(epoch, total_epochs, beta_start=0.3, beta_end=0.9):
    """Linear warmup from beta_start to beta_end."""
    t = epoch / max(total_epochs - 1, 1)
    return beta_start + (beta_end - beta_start) * t


def beta_cosine(epoch, total_epochs, beta_start=0.3, beta_end=0.9):
    """Cosine warmup: slow start, fast end."""
    t = epoch / max(total_epochs - 1, 1)
    return beta_start + (beta_end - beta_start) * 0.5 * (1 - math.cos(math.pi * t))


def beta_reverse(epoch, total_epochs, beta_start=0.9, beta_end=0.3):
    """Reverse schedule: start high, decrease (cooldown)."""
    t = epoch / max(total_epochs - 1, 1)
    return beta_start + (beta_end - beta_start) * t


def beta_step(epoch, total_epochs, milestones=None, betas=None):
    """Step schedule: β changes at LR milestones.
    Default: 0.3 for ep 0-19, 0.7 for ep 20-34, 0.9 for ep 35-49.
    """
    if milestones is None:
        milestones = [20, 35]
    if betas is None:
        betas = [0.3, 0.7, 0.9]
    for i, m in enumerate(milestones):
        if epoch < m:
            return betas[i]
    return betas[-1]


# Registry of all schedules
BETA_SCHEDULES = {
    "fixed_0.7":    ("Fixed β=0.7",          lambda e, T: beta_fixed(e, T, 0.7)),
    "linear_03_09": ("Linear 0.3→0.9",       lambda e, T: beta_linear(e, T, 0.3, 0.9)),
    "cosine_03_09": ("Cosine 0.3→0.9",       lambda e, T: beta_cosine(e, T, 0.3, 0.9)),
    "reverse_09_03":("Reverse 0.9→0.3",      lambda e, T: beta_reverse(e, T, 0.9, 0.3)),
    "step_03_07_09":("Step 0.3→0.7→0.9",     lambda e, T: beta_step(e, T)),
}


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
        H = H.float()  # force float32 for eigendecomposition
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


class ActivationCollector:
    """Hooks on ResNet BasicBlock + AdaptiveAvgPool2d."""

    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []
        for name, module in model.named_modules():
            if isinstance(module, (models.resnet.BasicBlock, models.resnet.Bottleneck)):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
        print(f"  [Hooks] {len(self.hooks)}: "
              f"{', '.join(self.hook_names[:4])}{'...' if len(self.hook_names) > 4 else ''}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            self.activations.append(output)

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
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def make_resnet18(num_classes=100):
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


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
# TRAINING — Mixed Precision FP16 with adaptive β
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.7):
    """Training loop with FP16 autocast, SES in FP32."""
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
# GENERIC TRAINING RUNNER (with adaptive β)
# ══════════════════════════════════════════════════════════════════════════════

def run_single_config(model, collector, device, train_loader, test_loader,
                      label, use_ses=False, beta_schedule_fn=None):
    """Train one (model, config) pair with adaptive β scheduling.

    Args:
        beta_schedule_fn: callable(epoch, total_epochs) -> float
            If None and use_ses=True, uses fixed β=0.7.
    """
    tag = gpu_tag()
    optimizer = optim.SGD(model.parameters(), lr=CFG.LR,
                          momentum=CFG.MOMENTUM, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG.LR_MILESTONES,
                                                gamma=CFG.LR_GAMMA)
    scaler = torch.amp.GradScaler("cuda")

    history = defaultdict(list)
    best_test_acc = 0.0
    best_state = None
    epoch_times = []

    pbar = tqdm(range(CFG.EPOCHS), desc=f"{tag} {label}", unit="ep", leave=True)
    for epoch in pbar:
        t0 = time.time()

        # ── Compute current β ──
        if use_ses and beta_schedule_fn is not None:
            current_beta = beta_schedule_fn(epoch, CFG.EPOCHS)
        else:
            current_beta = CFG.BETA

        train_res = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                     collector=collector,
                                     lambda_ses=CFG.LAMBDA_SES if use_ses else 0.0,
                                     beta=current_beta)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        if collector:
            collector.disable()
        test_res = evaluate(model, test_loader, device)
        if collector:
            collector.enable()
        scheduler.step()

        history["train_acc"].append(train_res["acc"])
        history["test_acc"].append(test_res["acc"])
        history["beta"].append(current_beta)

        if test_res["acc"] > best_test_acc:
            best_test_acc = test_res["acc"]
            best_state = copy.deepcopy(model.state_dict())

        gap = train_res["acc"] - test_res["acc"]
        vram = torch.cuda.max_memory_allocated(device) / 1e9
        pbar.set_postfix_str(f"Te {test_res['acc']:.1f}% β={current_beta:.2f} Gap {gap:.1f} {epoch_time:.1f}s")

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
# EXPERIMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def experiment_gpu0(device):
    """GPU 0: Baseline, Fixed β=0.7, Linear 0.3→0.9."""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} Adaptive β Scheduling — GPU 0")
    print(f"{tag} Baseline + Fixed β=0.7 + Linear 0.3→0.9")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("baseline",      False, None,                                    "Baseline (no SES)"),
        ("fixed_0.7",     True,  BETA_SCHEDULES["fixed_0.7"][1],         "Fixed β=0.7"),
        ("linear_03_09",  True,  BETA_SCHEDULES["linear_03_09"][1],      "Linear 0.3→0.9"),
    ]

    results = {}
    for key, use_ses, sched_fn, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_resnet18(num_classes=100).to(device)
        collector = ActivationCollector(model) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, beta_schedule_fn=sched_fn)

        del model, collector
        torch.cuda.empty_cache()

    return results


def experiment_gpu1(device):
    """GPU 1: Cosine 0.3→0.9, Reverse 0.9→0.3, Step 0.3→0.7→0.9."""
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} Adaptive β Scheduling — GPU 1")
    print(f"{tag} Cosine 0.3→0.9 + Reverse 0.9→0.3 + Step 0.3→0.7→0.9")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("cosine_03_09",  True, BETA_SCHEDULES["cosine_03_09"][1],  "Cosine 0.3→0.9"),
        ("reverse_09_03", True, BETA_SCHEDULES["reverse_09_03"][1], "Reverse 0.9→0.3"),
        ("step_03_07_09", True, BETA_SCHEDULES["step_03_07_09"][1], "Step 0.3→0.7→0.9"),
    ]

    results = {}
    for key, use_ses, sched_fn, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_resnet18(num_classes=100).to(device)
        collector = ActivationCollector(model) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, beta_schedule_fn=sched_fn)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_adaptive_beta(results, save_dir):
    """Create comprehensive plots for adaptive β experiment."""

    # Order: baseline, fixed, linear, cosine, reverse, step
    order = ["baseline", "fixed_0.7", "linear_03_09", "cosine_03_09",
             "reverse_09_03", "step_03_07_09"]
    labels = ["Baseline\n(no SES)", "Fixed\nβ=0.7", "Linear\n0.3→0.9",
              "Cosine\n0.3→0.9", "Reverse\n0.9→0.3", "Step\n0.3→0.7→0.9"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]

    present = [k for k in order if k in results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    # ═══════════════ Figure 1: Bar charts (Acc, Gap, Rob) ═══════════════

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Experiment 16: Adaptive β Scheduling (CIFAR-100, ResNet-18, FP16)",
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
        ax.set_xticklabels(present_labels, fontsize=7)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "20_adaptive_beta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 20_adaptive_beta.png")

    # ═══════════════ Figure 2: β schedule + accuracy curves ═══════════════

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1: β schedules over epochs
    ax = axes[0]
    ses_keys = [k for k in present if k != "baseline"]
    for key in ses_keys:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)].replace("\n", " ")
        ax.plot(h["beta"], color=c, lw=2, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("β")
    ax.set_title("β Schedule Over Training")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)

    # Panel 2: Test accuracy curves
    ax = axes[1]
    for key in present:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)].replace("\n", " ")
        ax.plot(h["test_acc"], color=c, lw=2,
                label=f"{lbl} ({results[key]['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy Curves")
    ax.legend(fontsize=6, loc="lower right"); ax.grid(True, alpha=0.3)

    # Panel 3: Train accuracy curves (dashed)
    ax = axes[2]
    for key in present:
        h = results[key]["history"]
        c = present_colors[present.index(key)]
        lbl = present_labels[present.index(key)].replace("\n", " ")
        ax.plot(h["train_acc"], color=c, ls="--", alpha=0.5)
        ax.plot(h["test_acc"], color=c, lw=2, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train (dashed) vs Test (solid)")
    ax.legend(fontsize=6, loc="lower right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "20b_adaptive_beta_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 20b_adaptive_beta_curves.png")

    # ═══════════════ Figure 3: Per-corruption breakdown ═══════════════

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
               label=present_labels[i].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Adaptive β Scheduling: Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "20c_adaptive_beta_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] 20c_adaptive_beta_percorruption.png")

    # ═══════════════ Figure 4: Δ vs Fixed β=0.7 (spider/radar-like bar) ═══════════════

    if "fixed_0.7" in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Adaptive Schedules vs Fixed β=0.7", fontsize=14, fontweight="bold")

        adaptive_keys = [k for k in present if k not in ("baseline", "fixed_0.7")]
        adaptive_labels = [present_labels[present.index(k)].replace("\n", " ") for k in adaptive_keys]
        adaptive_colors = [present_colors[present.index(k)] for k in adaptive_keys]
        fixed = results["fixed_0.7"]

        # Panel 1: Δ Accuracy
        ax = axes[0]
        deltas = [results[k]["best_acc"] - fixed["best_acc"] for k in adaptive_keys]
        bars = ax.barh(range(len(adaptive_keys)), deltas, color=adaptive_colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_width() + 0.02 * (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                    f"{val:+.2f}pp", ha="left" if val >= 0 else "right", va="center", fontsize=9)
        ax.set_yticks(range(len(adaptive_keys)))
        ax.set_yticklabels(adaptive_labels)
        ax.set_xlabel("Δ Accuracy (pp) vs Fixed β=0.7")
        ax.set_title("Accuracy Improvement")
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 2: Δ Robustness
        ax = axes[1]
        deltas = [results[k]["rob_mean"] - fixed["rob_mean"] for k in adaptive_keys]
        bars = ax.barh(range(len(adaptive_keys)), deltas, color=adaptive_colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_width() + 0.02 * (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                    f"{val:+.2f}pp", ha="left" if val >= 0 else "right", va="center", fontsize=9)
        ax.set_yticks(range(len(adaptive_keys)))
        ax.set_yticklabels(adaptive_labels)
        ax.set_xlabel("Δ Robustness (pp) vs Fixed β=0.7")
        ax.set_title("Robustness Improvement")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(save_dir / "20d_adaptive_beta_delta.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[PLOT] 20d_adaptive_beta_delta.png")


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

    if experiment_name == "gpu0":
        results = experiment_gpu0(device)
        out_path = CFG.OUTPUT_DIR / "gpu0_beta_results.json"
    elif experiment_name == "gpu1":
        results = experiment_gpu1(device)
        out_path = CFG.OUTPUT_DIR / "gpu1_beta_results.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    """Main process: download data, launch workers, combine results, plot."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Experiment 16: Adaptive β Scheduling                   ║")
    print("║   GPU 0: Baseline + Fixed + Linear                             ║")
    print("║   GPU 1: Cosine + Reverse + Step                               ║")
    print("║   CIFAR-100, ResNet-18, FP16, batch 512, 50 epochs             ║")
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
    gpu0_path = CFG.OUTPUT_DIR / "gpu0_beta_results.json"
    gpu1_path = CFG.OUTPUT_DIR / "gpu1_beta_results.json"

    all_results = {}
    for path in [gpu0_path, gpu1_path]:
        if path.exists():
            with open(path) as f:
                all_results.update(json.load(f))

    if all_results:
        # ── Save combined results ─────────────────────────
        combined_path = CFG.OUTPUT_DIR / "adaptive_beta_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # ── Save summary (without history) ─────────────────
        summary = {}
        for k, v in all_results.items():
            summary[k] = {kk: vv for kk, vv in v.items() if kk != "history"}
        summary_path = CFG.OUTPUT_DIR / "adaptive_beta_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # ── Plot ──────────────────────────────────────────
        plot_adaptive_beta(all_results, CFG.OUTPUT_DIR)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 16: ADAPTIVE β SCHEDULING COMPLETE — {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    if all_results:
        print(f"\n┌─ ADAPTIVE β SCHEDULING (CIFAR-100, ResNet-18, FP16) ────────────┐")
        order = ["baseline", "fixed_0.7", "linear_03_09", "cosine_03_09",
                 "reverse_09_03", "step_03_07_09"]
        nice_names = {
            "baseline": "Baseline",
            "fixed_0.7": "Fixed β=0.7",
            "linear_03_09": "Linear 0.3→0.9",
            "cosine_03_09": "Cosine 0.3→0.9",
            "reverse_09_03": "Reverse 0.9→0.3",
            "step_03_07_09": "Step 0.3→0.7→0.9",
        }
        for key in order:
            if key in all_results:
                r = all_results[key]
                name = nice_names[key]
                print(f"│  {name:<22} Acc: {r['best_acc']:.2f}%  "
                      f"Gap: {r['final_gap']:+.2f}pp  Rob: {r['rob_mean']:.2f}%  "
                      f"VRAM: {r['vram_peak_gb']:.1f}GB")

        if "fixed_0.7" in all_results:
            fixed = all_results["fixed_0.7"]
            print(f"│")
            print(f"│  Δ vs Fixed β=0.7:")
            for key in order:
                if key in all_results and key not in ("baseline", "fixed_0.7"):
                    r = all_results[key]
                    d_a = r["best_acc"] - fixed["best_acc"]
                    d_r = r["rob_mean"] - fixed["rob_mean"]
                    d_g = r["final_gap"] - fixed["final_gap"]
                    name = nice_names[key]
                    print(f"│    {name:<22} Acc {d_a:+.2f}pp  "
                          f"Gap {d_g:+.2f}pp  Rob {d_r:+.2f}pp")

        print(f"└──────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Exp 16: Adaptive β Scheduling")
    parser.add_argument("--worker", choices=["gpu0", "gpu1"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
