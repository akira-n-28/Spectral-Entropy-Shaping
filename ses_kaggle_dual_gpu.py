#!/usr/bin/env python3
"""
SES — Kaggle Dual-GPU Experiments (2x T4 16GB)
===============================================
Mixed Precision FP16 + SES regularizer in FP32.
True parallelism via subprocess (not threading).

GPU 0: WideResNet-28-10 on CIFAR-100 (Baseline + SES)
GPU 1: SES + Mixup/CutMix on CIFAR-100 (6 configs)

Usage:
  python ses_kaggle_dual_gpu.py                  # Launches both workers
  python ses_kaggle_dual_gpu.py --worker wrn     # Internal: WRN on assigned GPU
  python ses_kaggle_dual_gpu.py --worker aug     # Internal: Aug on assigned GPU
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
    OUTPUT_DIR = Path("/kaggle/working/outputs_dual_gpu") if Path("/kaggle").exists() else Path("./outputs_dual_gpu")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    LR_MILESTONES = [20, 35, 45]
    LR_GAMMA = 0.2
    NUM_WORKERS = 2

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
    """Returns [GPU X] prefix based on CUDA_VISIBLE_DEVICES."""
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
    """Hooks on ResNet BasicBlock/Bottleneck + AdaptiveAvgPool2d."""

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


class WRNCollector:
    """Hooks on WideResNet WideBasicBlocks + avgpool."""

    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "WideBasicBlock":
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
        print(f"  [WRN Hooks] {len(self.hooks)}: "
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
    """Pre-download CIFAR-100 (call from main before spawning workers)."""
    root = str(CFG.DATA_DIR / "cifar100")
    torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    print("[DATA] CIFAR-100 ready.")


def get_loaders(batch_size=CFG.BATCH_SIZE):
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
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def make_resnet18(num_classes=100):
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_wide_resnet_28_10(num_classes=100):
    class WideBasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                )
        def forward(self, x):
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            out += self.shortcut(x)
            return out

    class WideResNet(nn.Module):
        def __init__(self, depth=28, widen_factor=10, num_classes=100):
            super().__init__()
            n = (depth - 4) // 6
            k = widen_factor
            nStages = [16, 16*k, 32*k, 64*k]
            self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
            self.layer1 = self._make_layer(WideBasicBlock, nStages[0], nStages[1], n, stride=1)
            self.layer2 = self._make_layer(WideBasicBlock, nStages[1], nStages[2], n, stride=2)
            self.layer3 = self._make_layer(WideBasicBlock, nStages[2], nStages[3], n, stride=2)
            self.bn1 = nn.BatchNorm2d(nStages[3])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(nStages[3], num_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1); m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

        def _make_layer(self, block, in_planes, out_planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(block(in_planes, out_planes, s))
                in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            return self.fc(out)

    model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [WRN-28-10] {nparams:.1f}M params")
    return model


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
# TRAINING — Mixed Precision FP16 (T4 Tensor Cores)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.7, aug_type="none"):
    """Unified training loop. FP16 autocast for forward, FP32 for SES."""
    model.train()
    loss_m = AverageMeter()
    correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if collector:
            collector.clear()

        # Augmentation
        if aug_type == "mixup":
            X, y_a, y_b, lam = mixup_data(X, y)
        elif aug_type == "cutmix":
            X, y_a, y_b, lam = cutmix_data(X, y)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward in FP16 ──
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(X)
            if aug_type in ("mixup", "cutmix"):
                task_loss = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam)
            else:
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
    """Train one (model, config) pair. Returns result dict with history."""
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
        scheduler.step()

        history["train_acc"].append(train_res["acc"])
        history["test_acc"].append(test_res["acc"])

        if test_res["acc"] > best_test_acc:
            best_test_acc = test_res["acc"]
            best_state = copy.deepcopy(model.state_dict())

        gap = train_res["acc"] - test_res["acc"]
        vram = torch.cuda.max_memory_allocated(device) / 1e9
        pbar.set_postfix_str(f"Te {test_res['acc']:.1f}% Gap {gap:.1f} {epoch_time:.1f}s {vram:.1f}GB")

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
# EXPERIMENT A: WIDERESNET-28-10
# ══════════════════════════════════════════════════════════════════════════════

def experiment_wideresnet(device):
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} WideResNet-28-10 on CIFAR-100 (Baseline + SES)")
    print(f"{tag} Mixed Precision FP16, SES in FP32")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()
    results = {}

    for use_ses, label in [(False, "WRN-Baseline"), (True, "WRN-SES")]:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_wide_resnet_28_10(num_classes=100).to(device)
        collector = WRNCollector(model) if use_ses else None

        key = "ses" if use_ses else "baseline"
        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, aug_type="none")

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: SES + MIXUP / CUTMIX
# ══════════════════════════════════════════════════════════════════════════════

def experiment_augmentation(device):
    tag = gpu_tag()
    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} SES + Mixup / CutMix on CIFAR-100 (6 configs)")
    print(f"{tag} Mixed Precision FP16, SES in FP32")
    print(f"{tag} {'=' * 55}")

    train_loader, test_loader = get_loaders()

    configs = [
        ("baseline",   False, "none",   "Baseline"),
        ("ses",        True,  "none",   "SES"),
        ("mixup",      False, "mixup",  "Mixup"),
        ("ses_mixup",  True,  "mixup",  "SES+Mixup"),
        ("cutmix",     False, "cutmix", "CutMix"),
        ("ses_cutmix", True,  "cutmix", "SES+CutMix"),
    ]

    results = {}

    for key, use_ses, aug_type, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = make_resnet18(num_classes=100).to(device)
        collector = ActivationCollector(model) if use_ses else None

        results[key] = run_single_config(
            model, collector, device, train_loader, test_loader,
            label=label, use_ses=use_ses, aug_type=aug_type)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_wideresnet(results, save_dir):
    methods = ["Baseline", "SES"]
    keys = ["baseline", "ses"]
    colors = ["#377eb8", "#e41a1c"]

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WideResNet-28-10 on CIFAR-100 (36.5M params, FP16)",
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
    plt.savefig(save_dir / "16_wideresnet.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 16_wideresnet.png")

    # Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color, label in zip(keys, colors, methods):
        h = results[key]["history"]
        ax.plot(h["train_acc"], color=color, ls="--", alpha=0.4)
        ax.plot(h["test_acc"], color=color, lw=2,
                label=f"{label} ({results[key]['best_acc']:.2f}%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("WRN-28-10 Training Curves (dashed=train, solid=test)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "16b_wideresnet_curves.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 16b_wideresnet_curves.png")

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
    ax.set_title("WRN-28-10: Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "16c_wideresnet_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 16c_wideresnet_percorruption.png")


def plot_augmentation(results, save_dir):
    methods = ["Baseline", "SES", "Mixup", "SES+Mixup", "CutMix", "SES+CutMix"]
    keys = ["baseline", "ses", "mixup", "ses_mixup", "cutmix", "ses_cutmix"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("SES + Mixup / CutMix on CIFAR-100 (ResNet-18, FP16)",
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
    plt.savefig(save_dir / "17_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 17_augmentation.png")

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
    ax.set_title("SES + Mixup/CutMix: Per-Corruption Robustness", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "17c_augmentation_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 17c_augmentation_percorruption.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — SUBPROCESS ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def worker_main(experiment_name):
    """Entry point for subprocess workers. GPU set via CUDA_VISIBLE_DEVICES."""
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    tag = gpu_tag()
    print(f"{tag} Worker '{experiment_name}' started — {name} ({vram:.1f} GB)")

    if experiment_name == "wrn":
        results = experiment_wideresnet(device)
        out_path = CFG.OUTPUT_DIR / "wrn_results.json"
    elif experiment_name == "aug":
        results = experiment_augmentation(device)
        out_path = CFG.OUTPUT_DIR / "aug_results.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")


def orchestrator_main():
    """Main process: download data, launch workers, combine results, plot."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Kaggle Dual-GPU Experiments                            ║")
    print("║   GPU 0: WRN-28-10  ·  GPU 1: SES + Mixup/CutMix             ║")
    print("║   Mixed Precision FP16  ·  SES regularizer in FP32             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    # Detect GPUs
    n_gpus = torch.cuda.device_count()
    print(f"\n[INFO] {n_gpus} GPU(s) detected:")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # Pre-download data (avoid race condition between workers)
    print("\n[DATA] Pre-downloading CIFAR-100...")
    download_cifar100()

    script_path = os.path.abspath(__file__)

    if n_gpus >= 2:
        # ── PARALLEL: 2 subprocesses, 1 per GPU ─────────────
        print(f"\n[PARALLEL] Launching 2 workers...")
        env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

        p0 = subprocess.Popen(
            [sys.executable, script_path, "--worker", "wrn"],
            env=env0, stdout=sys.stdout, stderr=sys.stderr)
        p1 = subprocess.Popen(
            [sys.executable, script_path, "--worker", "aug"],
            env=env1, stdout=sys.stdout, stderr=sys.stderr)

        ret0 = p0.wait()
        ret1 = p1.wait()
        print(f"\n[PARALLEL] Workers finished (exit codes: wrn={ret0}, aug={ret1})")
    else:
        # ── SEQUENTIAL: single GPU ──────────────────────────
        print(f"\n[SEQUENTIAL] Only {n_gpus} GPU — running sequentially...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        worker_main("wrn")
        worker_main("aug")

    # ── Load results ─────────────────────────────────────
    wrn_path = CFG.OUTPUT_DIR / "wrn_results.json"
    aug_path = CFG.OUTPUT_DIR / "aug_results.json"

    wrn_results = None
    aug_results = None

    if wrn_path.exists():
        with open(wrn_path) as f:
            wrn_results = json.load(f)
        plot_wideresnet(wrn_results, CFG.OUTPUT_DIR)

    if aug_path.exists():
        with open(aug_path) as f:
            aug_results = json.load(f)
        plot_augmentation(aug_results, CFG.OUTPUT_DIR)

    # ── Summary JSON (without history) ───────────────────
    def strip_history(d):
        return {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                for k, v in d.items()}

    summary = {}
    if wrn_results:
        summary["wideresnet_28_10"] = strip_history(wrn_results)
    if aug_results:
        summary["augmentation_mixup_cutmix"] = strip_history(aug_results)

    summary_path = CFG.OUTPUT_DIR / "dual_gpu_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"DUAL-GPU EXPERIMENTS COMPLETE — {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"{'=' * 70}")

    if wrn_results:
        wrn = wrn_results
        print(f"\n┌─ WIDERESNET-28-10 (CIFAR-100, FP16) ─────────────────────────┐")
        for key, label in [("baseline", "WRN-Base"), ("ses", "WRN-SES")]:
            r = wrn[key]
            print(f"│  {label:<12} Acc: {r['best_acc']:.2f}%  Gap: {r['final_gap']:+.2f}pp  "
                  f"Rob: {r['rob_mean']:.2f}%  VRAM: {r['vram_peak_gb']:.1f}GB")
        d_acc = wrn["ses"]["best_acc"] - wrn["baseline"]["best_acc"]
        d_rob = wrn["ses"]["rob_mean"] - wrn["baseline"]["rob_mean"]
        print(f"│  Δ SES:      Acc {d_acc:+.2f}pp   Rob {d_rob:+.2f}pp")
        print(f"└──────────────────────────────────────────────────────────────┘")

    if aug_results:
        aug = aug_results
        print(f"\n┌─ SES + MIXUP / CUTMIX (CIFAR-100, FP16) ────────────────────┐")
        for key, label in [("baseline", "Base"), ("ses", "SES"), ("mixup", "Mixup"),
                            ("ses_mixup", "SES+Mix"), ("cutmix", "CutMix"), ("ses_cutmix", "SES+Cut")]:
            r = aug[key]
            print(f"│  {label:<12} Acc: {r['best_acc']:.2f}%  Gap: {r['final_gap']:+.2f}pp  "
                  f"Rob: {r['rob_mean']:.2f}%")
        print(f"│")
        print(f"│  SES additivity:")
        for base_key, ses_key, name in [("baseline", "ses", "None→SES"),
                                         ("mixup", "ses_mixup", "Mixup→SES+Mixup"),
                                         ("cutmix", "ses_cutmix", "CutMix→SES+CutMix")]:
            d_a = aug[ses_key]["best_acc"] - aug[base_key]["best_acc"]
            d_r = aug[ses_key]["rob_mean"] - aug[base_key]["rob_mean"]
            print(f"│    {name:<22} Acc {d_a:+.2f}pp  Rob {d_r:+.2f}pp")
        print(f"└──────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")
    print(f"Summary: {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Dual-GPU Experiments")
    parser.add_argument("--worker", choices=["wrn", "aug"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
