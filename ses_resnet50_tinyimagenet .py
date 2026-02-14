#!/usr/bin/env python3
"""
SES Phase 3 — Exp 2: ResNet-50 on Tiny-ImageNet (L40S)
=======================================================
Target: Lightning AI, 1× NVIDIA L40S (48GB VRAM, 16 cores)
Estimated runtime: ~1h

ResNet-50 (25M params, Bottleneck blocks) — different from ResNet-18 (BasicBlock).
Tiny-ImageNet: 200 classes, 64×64, 100k train / 10k val.
"""

import os
import time
import json
import math
import copy
import warnings
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

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

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    OUTPUT_DIR = Path("./outputs_resnet50_tinyimagenet")
    DATA_DIR = Path("./data")

    EPOCHS = 60
    BATCH_SIZE = 256          # Safe for ResNet-50 + SES hooks on L40S
    LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    LR_MILESTONES = [25, 40, 52]
    LR_GAMMA = 0.2
    NUM_WORKERS = 10

    LAMBDA_SES = 0.01
    BETA = 0.7
    SEED = 42


CFG = Config()
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CFG.DATA_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# GPU SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_gpu():
    if not torch.cuda.is_available():
        print("[WARN] No GPU"); return torch.device("cpu")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {name} — {vram:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[GPU] TF32 + cuDNN benchmark enabled")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# ══════════════════════════════════════════════════════════════════════════════
# SES CORE — hooks pool spatially to avoid OOM
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
    device = activations[0].device
    reg = torch.tensor(0.0, device=device)
    for H in activations:
        if H.dim() > 2:
            H = H.mean(dim=[2, 3]) if H.dim() == 4 else H.flatten(1)
        H = H.float()  # eigendecomposition needs float32
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


class ActivationCollector:
    """Hooks on layer4 Bottleneck blocks + AdaptiveAvgPool2d only.
    
    Why only layer4? Two reasons:
    1. OOM prevention: 17 hooks with eigendecomposition in the backward graph
       causes OOM even with spatial pooling. 4 hooks is safe.
    2. Layer ablation (Exp 9) showed hooking only the final layers gives
       the best accuracy/gap trade-off.
    
    Pools spatial dims in-hook: (B,C,H,W) → (B,C) immediately.
    """

    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []

        # Collect all eligible layers in order
        all_eligible = []
        for name, module in model.named_modules():
            if isinstance(module, (models.resnet.Bottleneck, models.resnet.BasicBlock)):
                all_eligible.append((name, module))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                all_eligible.append((name, module))

        # ResNet-50 has 16 Bottleneck + 1 avgpool = 17 eligible
        # Hook only last 4: layer4.0, layer4.1, layer4.2, avgpool
        targets = all_eligible[-4:] if len(all_eligible) > 4 else all_eligible

        for name, module in targets:
            self.hooks.append(module.register_forward_hook(self._hook_fn))
            self.hook_names.append(name)

        print(f"  [Hooks] {len(self.hooks)} of {len(all_eligible)} eligible (last group only): "
              f"{', '.join(self.hook_names)}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            if output.dim() == 4:
                self.activations.append(output.mean(dim=[2, 3]))  # (B,C,H,W) → (B,C)
            else:
                self.activations.append(output.flatten(1) if output.dim() > 2 else output)

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
# TINY-IMAGENET
# ══════════════════════════════════════════════════════════════════════════════

def download_tiny_imagenet(data_dir):
    tinyim_dir = Path(data_dir) / "tiny-imagenet-200"
    if tinyim_dir.exists() and (tinyim_dir / "train").exists():
        # Check if val is already restructured
        val_ann = tinyim_dir / "val" / "val_annotations.txt"
        if not val_ann.exists():
            print(f"[DATA] Tiny-ImageNet ready at {tinyim_dir}")
            return tinyim_dir

    if not (tinyim_dir / "train").exists():
        print("[DATA] Downloading Tiny-ImageNet (237 MB)...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = Path(data_dir) / "tiny-imagenet-200.zip"
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(zip_path), url], check=True)
        print("[DATA] Extracting...")
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(data_dir)], check=True)
        zip_path.unlink()

    # Fix val structure
    val_dir = tinyim_dir / "val"
    val_ann = val_dir / "val_annotations.txt"
    if val_ann.exists():
        print("[DATA] Restructuring val/ into class folders...")
        with open(val_ann) as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, cls = parts[0], parts[1]
                cls_dir = val_dir / cls
                cls_dir.mkdir(exist_ok=True)
                src = val_dir / "images" / fname
                dst = cls_dir / fname
                if src.exists():
                    src.rename(dst)
        images_dir = val_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir, ignore_errors=True)
        val_ann.unlink(missing_ok=True)

    print(f"[DATA] Tiny-ImageNet ready at {tinyim_dir}")
    return tinyim_dir


def get_loaders(batch_size=CFG.BATCH_SIZE):
    tinyim_dir = download_tiny_imagenet(CFG.DATA_DIR)
    mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

    train_tf = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.ImageFolder(str(tinyim_dir / "train"), transform=train_tf)
    val_set   = torchvision.datasets.ImageFolder(str(tinyim_dir / "val"), transform=test_tf)

    print(f"[DATA] Train: {len(train_set)}, Val: {len(val_set)}, Classes: {len(train_set.classes)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — ResNet-50 adapted for 64×64
# ══════════════════════════════════════════════════════════════════════════════

def make_resnet50(num_classes=200):
    """ResNet-50 with Bottleneck blocks, adapted for 64×64 Tiny-ImageNet.
    Standard ImageNet ResNet-50 uses 7×7 conv + maxpool → 56×56 for 224×224 input.
    For 64×64: use 3×3 conv stride 1, no maxpool → keeps 64×64 into layer1.
    """
    model = models.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [ResNet-50] {nparams:.1f}M params, {num_classes} classes")
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
def evaluate_robustness(model, loader, device, severities=[1, 3, 5]):
    model.eval()
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    results = {}
    for corr in corruptions:
        for sev in severities:
            correct = total = 0
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                X_c = apply_corruption(X, corr, sev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    correct += (model(X_c).argmax(1) == y).sum().item()
                total += y.size(0)
            results[f"{corr}_s{sev}"] = 100.0 * correct / total
    results["mean_corruption_acc"] = np.mean([v for k, v in results.items() if k != "mean_corruption_acc"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m = AverageMeter()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(X)
            loss_m.update(F.cross_entropy(output, y).item(), X.size(0))
        correct += (output.argmax(1) == y).sum().item()
        total += X.size(0)
    return {"loss": loss_m.avg, "acc": 100.0 * correct / total}


def run_training(device, use_ses=False, label="R50"):
    set_seed(CFG.SEED)
    train_loader, val_loader = get_loaders()

    model = make_resnet50(num_classes=200).to(device)
    optimizer = optim.SGD(model.parameters(), lr=CFG.LR,
                          momentum=CFG.MOMENTUM, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG.LR_MILESTONES,
                                                gamma=CFG.LR_GAMMA)
    scaler = torch.amp.GradScaler("cuda")
    print(f"  [AMP] BFloat16 + GradScaler enabled")

    collector = None
    if use_ses:
        collector = ActivationCollector(model)

    history = defaultdict(list)
    best_val_acc = 0.0
    best_state = None
    epoch_times = []

    pbar = tqdm(range(CFG.EPOCHS), desc=f"[{label}]", unit="ep", leave=True)
    for epoch in pbar:
        t0 = time.time()
        model.train()
        loss_m, reg_m = AverageMeter(), AverageMeter()
        correct = total = 0

        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if collector: collector.clear()

            optimizer.zero_grad(set_to_none=True)

            # Forward pass in BF16
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(X)
                task_loss = F.cross_entropy(output, y)

            # SES regularizer outside autocast — eigdecomp needs float32
            reg_loss = torch.tensor(0.0, device=device)
            if collector and use_ses and collector.activations:
                reg_loss = CFG.LAMBDA_SES * ses_regularizer(
                    collector.activations, beta=CFG.BETA).to(device)

            total_loss = task_loss.float() + reg_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = X.size(0)
            loss_m.update(total_loss.item(), bs)
            reg_m.update(reg_loss.item(), bs)
            correct += (output.argmax(1) == y).sum().item()
            total += bs

        train_acc = 100.0 * correct / total
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        if collector: collector.disable()
        val_res = evaluate(model, val_loader, device)
        if collector: collector.enable()
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_res["acc"])
        history["train_loss"].append(loss_m.avg)
        history["val_loss"].append(val_res["loss"])

        if val_res["acc"] > best_val_acc:
            best_val_acc = val_res["acc"]
            best_state = copy.deepcopy(model.state_dict())

        gap = train_acc - val_res["acc"]
        vram = torch.cuda.max_memory_allocated() / 1e9
        pbar.set_postfix_str(
            f"Tr {train_acc:.1f}% | Val {val_res['acc']:.1f}% | "
            f"Gap {gap:.1f} | {epoch_time:.0f}s | {vram:.1f}GB")

    if collector: collector.remove()
    model.load_state_dict(best_state)

    # Save best model checkpoint
    save_name = f"resnet50_tinyimagenet_{'ses' if use_ses else 'baseline'}_best.pt"
    save_path = CFG.OUTPUT_DIR / save_name
    torch.save({
        "model_state_dict": best_state,
        "best_val_acc": best_val_acc,
        "config": {"lambda_ses": CFG.LAMBDA_SES, "beta": CFG.BETA,
                   "epochs": CFG.EPOCHS, "use_ses": use_ses,
                   "seed": CFG.SEED, "batch_size": CFG.BATCH_SIZE,
                   "lr": CFG.LR, "weight_decay": CFG.WEIGHT_DECAY},
    }, save_path)
    print(f"  [{label}] Model saved → {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")

    print(f"  [{label}] Best: {best_val_acc:.2f}% — evaluating robustness...")
    rob = evaluate_robustness(model, val_loader, device)

    gap = history["train_acc"][-1] - history["val_acc"][-1]
    result = {
        "best_val_acc": best_val_acc,
        "final_gap": gap,
        "rob_mean": rob["mean_corruption_acc"],
        "robustness_detail": {k: v for k, v in rob.items() if k != "mean_corruption_acc"},
        "avg_epoch_time": float(np.mean(epoch_times)),
        "history": dict(history),
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "model_path": str(save_path),
    }
    print(f"  [{label}] Acc: {best_val_acc:.2f}% | Gap: {gap:.2f}pp | "
          f"Rob: {rob['mean_corruption_acc']:.2f}% | {np.mean(epoch_times):.0f}s/ep")

    del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(results, save_dir):
    methods = ["Baseline", "SES"]
    keys = ["baseline", "ses"]
    colors = ["#377eb8", "#e41a1c"]

    # --- Bars ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ResNet-50 on Tiny-ImageNet (200 classes, 64×64, 25M params)",
                 fontsize=14, fontweight="bold")
    for idx, (title, ylabel, fn) in enumerate([
        ("Best Val Accuracy", "Acc (%)", lambda r: r["best_val_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [fn(results[k]) for k in keys]
        bars = ax.bar(range(2), vals, color=colors, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15,
                    f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_xticks(range(2)); ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "15_resnet50_tinyimagenet.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 15_resnet50_tinyimagenet.png")

    # --- Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("ResNet-50 Training on Tiny-ImageNet", fontsize=14, fontweight="bold")
    for key, color, label in zip(keys, colors, methods):
        h = results[key]["history"]
        axes[0].plot(h["train_acc"], color=color, ls="--", alpha=0.4)
        axes[0].plot(h["val_acc"], color=color, lw=2,
                     label=f"{label} ({results[key]['best_val_acc']:.2f}%)")
        gaps = [t-v for t, v in zip(h["train_acc"], h["val_acc"])]
        axes[1].plot(gaps, color=color, lw=2, label=label)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Train (--) / Val (—)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Gap (pp)")
    axes[1].set_title("Generalization Gap"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "15b_resnet50_tinyimagenet_curves.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 15b_resnet50_tinyimagenet_curves.png")

    # --- Per-corruption ---
    fig, ax = plt.subplots(figsize=(16, 6))
    corruptions = ["gaussian_noise", "gaussian_blur", "contrast", "brightness", "shot_noise"]
    severities = [1, 3, 5]
    xlabels = [f"{c.replace('_',' ')}\ns{s}" for c in corruptions for s in severities]
    x = np.arange(len(xlabels)); w = 0.35
    for i, (key, color, label) in enumerate(zip(keys, colors, methods)):
        rob = results[key]["robustness_detail"]
        vals = [rob[f"{c}_s{s}"] for c in corruptions for s in severities]
        ax.bar(x + (i-0.5)*w, vals, w, color=color, alpha=0.85, label=label)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.legend()
    ax.set_title("Per-Corruption Robustness: ResNet-50 on Tiny-ImageNet",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "15c_resnet50_tinyimagenet_percorruption.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[PLOT] 15c_resnet50_tinyimagenet_percorruption.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — ResNet-50 on Tiny-ImageNet (200 cls, 64×64)            ║")
    print("║   L40S · BFloat16 · batch 256 · 60 epochs                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    device = setup_gpu()
    t_start = time.time()
    results = {}

    print("\n" + "=" * 60)
    print("  BASELINE")
    print("=" * 60)
    results["baseline"] = run_training(device, use_ses=False, label="R50-Base")

    print("\n" + "=" * 60)
    print("  SES (λ=0.01, β=0.7)")
    print("=" * 60)
    results["ses"] = run_training(device, use_ses=True, label="R50-SES")

    plot_all(results, CFG.OUTPUT_DIR)

    # Save JSON (without history)
    json_out = {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                for k, v in results.items()}
    with open(CFG.OUTPUT_DIR / "resnet50_tinyimagenet_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE — {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"{'='*60}")
    print(f"\n  {'Method':<12} {'Acc':>7} {'Gap':>8} {'Rob':>7} {'Time':>8} {'VRAM':>6}")
    print(f"  {'-'*50}")
    for key, label in [("baseline", "Baseline"), ("ses", "SES")]:
        r = results[key]
        print(f"  {label:<12} {r['best_val_acc']:>6.2f}% {r['final_gap']:>7.2f}pp "
              f"{r['rob_mean']:>6.2f}% {r['avg_epoch_time']:>6.0f}s/ep {r['vram_peak_gb']:>5.1f}GB")
    d_acc = results["ses"]["best_val_acc"] - results["baseline"]["best_val_acc"]
    d_rob = results["ses"]["rob_mean"] - results["baseline"]["rob_mean"]
    print(f"\n  Δ Accuracy:   {d_acc:+.2f} pp")
    print(f"  Δ Robustness: {d_rob:+.2f} pp")
    print(f"\nOutputs: {CFG.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
