#!/usr/bin/env python3
"""
SES — Experiment 28: Effective Rank Dynamics (2x T4 16GB)
==========================================================
Tracks effective rank per layer throughout training for Baseline vs SES.
Visualizes convergence of erank to the target β·log(d).

GPU 0: ResNet-18 CIFAR-100 (Baseline + SES β=0.7)
GPU 1: ViT-Small/4 CIFAR-100 (Baseline + SES β=0.5)

Generates:
  - erank_dynamics_resnet18.png: per-layer erank over epochs (ResNet-18)
  - erank_dynamics_vit.png: per-layer erank over epochs (ViT-Small)
  - erank_summary.png: combined view

Usage:
  python ses_erank_dynamics.py
  python ses_erank_dynamics.py --worker resnet
  python ses_erank_dynamics.py --worker vit
"""

import os, sys, time, json, math, copy, argparse, warnings, subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    OUTPUT_DIR = Path("/kaggle/working/outputs_erank") if Path("/kaggle").exists() else Path("./outputs_erank")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")
    EPOCHS = 50
    BATCH_SIZE = 512
    NUM_WORKERS = 2
    LAMBDA_SES = 0.01
    SEED = 42
    ERANK_SAMPLES = 2048  # samples for erank computation


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
    return f"[GPU{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}]"


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


def ses_regularizer(activations, beta, eps=1e-12):
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
# ERANK MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_erank(model, hook_points, loader, device, n_samples=2048):
    """
    Forward pass on n_samples, collect activations at hook_points,
    compute erank per hook point.
    Returns dict: {hook_name: {"erank": float, "dim": int, "entropy": float}}
    """
    model.eval()
    collected = {name: [] for name in hook_points}
    hooks = []

    def make_hook(name, pool_mode):
        def hook_fn(module, input, output):
            out = output
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 4:   # CNN: [B, C, H, W]
                out = out.mean(dim=[2, 3])
            elif out.dim() == 3: # ViT: [B, seq, D]
                out = out.mean(dim=1)
            collected[name].append(out.float().cpu())
        return hook_fn

    # Register hooks
    for name, (module, pool_mode) in hook_points.items():
        hooks.append(module.register_forward_hook(make_hook(name, pool_mode)))

    # Forward pass on subset
    total = 0
    for X, _ in loader:
        if total >= n_samples:
            break
        X = X.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            model(X)
        total += X.size(0)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute erank
    results = {}
    for name in hook_points:
        if collected[name]:
            H = torch.cat(collected[name], dim=0)[:n_samples]
            d = H.shape[1]
            ent, er = spectral_entropy(H)
            results[name] = {
                "erank": er.item(),
                "dim": d,
                "entropy": ent.item(),
                "max_entropy": math.log(d),
            }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# RESNET-18 (CIFAR-adapted)
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


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

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
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_resnet_hook_points(model):
    """Returns all hookable points in ResNet-18 with their dimensions."""
    points = {}
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            name = f"{layer_name}.{i}"
            points[name] = (block, "cnn")
    points["avgpool"] = (model.avgpool, "cnn")
    return points


class ResNetCollector:
    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, name)
            for block in layer:
                self.hooks.append(block.register_forward_hook(self._hook_fn))
        self.hooks.append(model.avgpool.register_forward_hook(self._hook_fn))

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


# ══════════════════════════════════════════════════════════════════════════════
# VIT-SMALL/4 (reused from previous experiments)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(mlp_hidden, dim), nn.Dropout(drop))
        self.drop_path_rate = drop_path

    def _drop_path(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_path_rate
        mask = torch.bernoulli(torch.full((x.shape[0],)+(1,)*(x.ndim-1), keep, device=x.device, dtype=x.dtype))
        return x * mask / keep

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + self._drop_path(a)
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def get_vit_hook_points(model):
    points = {}
    for i, block in enumerate(model.blocks):
        points[f"block_{i}"] = (block, "vit")
    return points


class ViTCollector:
    def __init__(self, model, hook_mode="all"):
        self.activations = []
        self.hooks = []
        self.enabled = True
        blocks = list(model.blocks)
        n = len(blocks)
        if hook_mode == "last_4":
            indices = list(range(max(0, n-4), n))
        else:
            indices = list(range(n))
        for idx in indices:
            self.hooks.append(blocks[idx].register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            self.activations.append(output.mean(dim=1))

    def clear(self):
        self.activations = []
    def disable(self):
        self.enabled = False; self.activations = []
    def enable(self):
        self.enabled = True
    def remove(self):
        for h in self.hooks: h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_loaders():
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = str(CFG.DATA_DIR / "cifar100")
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=CFG.BATCH_SIZE, shuffle=False,
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
            p = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH ERANK LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def train_with_erank_logging(model, collector, hook_points, device,
                              train_loader, test_loader, label,
                              use_ses, beta, optimizer_type="sgd"):
    tag = gpu_tag()

    if optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = CosineWarmupScheduler(optimizer, 5, CFG.EPOCHS)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35, 45], gamma=0.1)

    scaler = torch.amp.GradScaler("cuda")

    erank_history = []  # list of dicts per epoch
    history = defaultdict(list)
    best_test_acc = 0.0
    best_state = None

    pbar = tqdm(range(CFG.EPOCHS), desc=f"{tag} {label}", unit="ep", leave=True)
    for epoch in pbar:
        model.train()
        loss_m = AverageMeter()
        correct = total = 0

        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if collector:
                collector.clear()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(X)
                task_loss = F.cross_entropy(output, y)

            reg_loss = torch.tensor(0.0, device=device)
            if collector and use_ses and collector.activations:
                reg_loss = CFG.LAMBDA_SES * ses_regularizer(collector.activations, beta=beta)

            total_loss = task_loss.float() + reg_loss
            scaler.scale(total_loss).backward()

            if optimizer_type == "adamw":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            bs = X.size(0)
            loss_m.update(total_loss.item(), bs)
            correct += (output.argmax(1) == y).sum().item()
            total += bs

        if optimizer_type == "sgd":
            scheduler.step()
        else:
            scheduler.step(epoch)

        train_acc = 100.0 * correct / total

        # Evaluate
        if collector:
            collector.disable()
        model.eval()
        correct = total = 0
        for X, y in test_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        test_acc = 100.0 * correct / total
        if collector:
            collector.enable()

        # Measure erank
        erank_data = measure_erank(model, hook_points, test_loader, device,
                                    n_samples=CFG.ERANK_SAMPLES)
        erank_data["epoch"] = epoch
        erank_history.append(erank_data)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        pbar.set_postfix_str(f"Te {test_acc:.1f}% Gap {train_acc - test_acc:.1f}")

    if collector:
        collector.remove()
    model.load_state_dict(best_state)

    print(f"{tag} {label} DONE: Best Acc {best_test_acc:.2f}%")

    return {
        "best_acc": best_test_acc,
        "history": dict(history),
        "erank_history": erank_history,
        "beta": beta,
        "use_ses": use_ses,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: ResNet-18
# ══════════════════════════════════════════════════════════════════════════════

def experiment_resnet(device):
    tag = gpu_tag()
    print(f"\n{tag} {'='*55}")
    print(f"{tag} Erank Dynamics — ResNet-18 CIFAR-100")
    print(f"{tag} {'='*55}")

    train_loader, test_loader = get_loaders()
    results = {}

    for use_ses, beta, label in [
        (False, 0.7, "ResNet18-Baseline"),
        (True,  0.7, "ResNet18-SES-b0.7"),
    ]:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = ResNet18(num_classes=100).to(device)
        hook_points = get_resnet_hook_points(model)
        collector = ResNetCollector(model) if use_ses else None

        key = "ses" if use_ses else "baseline"
        results[key] = train_with_erank_logging(
            model, collector, hook_points, device,
            train_loader, test_loader, label,
            use_ses=use_ses, beta=beta, optimizer_type="sgd")

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: ViT-Small
# ══════════════════════════════════════════════════════════════════════════════

def experiment_vit(device):
    tag = gpu_tag()
    print(f"\n{tag} {'='*55}")
    print(f"{tag} Erank Dynamics — ViT-Small/4 CIFAR-100")
    print(f"{tag} {'='*55}")

    train_loader, test_loader = get_loaders()
    results = {}

    for use_ses, beta, label in [
        (False, 0.5, "ViTS-Baseline"),
        (True,  0.5, "ViTS-SES-b0.5"),
    ]:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)
        model = VisionTransformer(
            img_size=32, patch_size=4, num_classes=100,
            embed_dim=384, depth=12, num_heads=6).to(device)
        hook_points = get_vit_hook_points(model)
        collector = ViTCollector(model) if use_ses else None

        key = "ses" if use_ses else "baseline"
        results[key] = train_with_erank_logging(
            model, collector, hook_points, device,
            train_loader, test_loader, label,
            use_ses=use_ses, beta=beta, optimizer_type="adamw")

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_erank_dynamics(results, arch_name, beta, save_dir):
    """Plot erank per layer over epochs for Baseline vs SES."""
    baseline = results["baseline"]["erank_history"]
    ses_data = results["ses"]["erank_history"]

    # Get layer names from first epoch
    layer_names = [k for k in baseline[0].keys() if k != "epoch"]
    n_layers = len(layer_names)
    epochs = list(range(len(baseline)))

    # Color maps
    cmap = cm.get_cmap("viridis", n_layers)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(f"Exp 28: Effective Rank Dynamics — {arch_name} (CIFAR-100, β={beta})",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Baseline erank over epochs ──
    ax = axes[0]
    for i, name in enumerate(layer_names):
        vals = [baseline[ep][name]["erank"] for ep in range(len(baseline))]
        d = baseline[0][name]["dim"]
        ax.plot(epochs, vals, color=cmap(i), lw=1.5, alpha=0.8,
                label=f"{name} (d={d})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Effective Rank")
    ax.set_title("Baseline (no SES)")
    ax.grid(True, alpha=0.3)
    if n_layers <= 12:
        ax.legend(fontsize=5, ncol=2, loc="upper right")

    # ── Panel 2: SES erank over epochs ──
    ax = axes[1]
    for i, name in enumerate(layer_names):
        vals = [ses_data[ep][name]["erank"] for ep in range(len(ses_data))]
        d = ses_data[0][name]["dim"]
        target = d ** beta
        ax.plot(epochs, vals, color=cmap(i), lw=1.5, alpha=0.8,
                label=f"{name} (d={d})")
        ax.axhline(target, color=cmap(i), ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Effective Rank")
    ax.set_title(f"SES (β={beta}) — dotted = target d^β")
    ax.grid(True, alpha=0.3)
    if n_layers <= 12:
        ax.legend(fontsize=5, ncol=2, loc="upper right")

    # ── Panel 3: Final epoch comparison (bar chart) ──
    ax = axes[2]
    x = np.arange(n_layers)
    w = 0.35
    final_base = [baseline[-1][name]["erank"] for name in layer_names]
    final_ses = [ses_data[-1][name]["erank"] for name in layer_names]
    targets = [baseline[-1][name]["dim"] ** beta for name in layer_names]

    ax.bar(x - w/2, final_base, w, color="#377eb8", alpha=0.8, label="Baseline")
    ax.bar(x + w/2, final_ses, w, color="#e41a1c", alpha=0.8, label="SES")
    ax.scatter(x, targets, color="black", marker="*", s=60, zorder=5, label=f"Target d^{beta}")

    short_names = [n.replace("layer", "L").replace("block_", "B") for n in layer_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Final Epoch: Baseline vs SES vs Target")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = f"28_erank_dynamics_{arch_name.lower().replace('-','').replace('/','')}.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {fname}")


def plot_erank_convergence(results, arch_name, beta, save_dir):
    """Plot erank/target ratio over epochs — shows convergence to 1.0."""
    ses_data = results["ses"]["erank_history"]
    layer_names = [k for k in ses_data[0].keys() if k != "epoch"]
    n_layers = len(layer_names)
    epochs = list(range(len(ses_data)))
    cmap = cm.get_cmap("viridis", n_layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Target (erank = d^β)")

    for i, name in enumerate(layer_names):
        d = ses_data[0][name]["dim"]
        target = d ** beta
        ratios = [ses_data[ep][name]["erank"] / target for ep in range(len(ses_data))]
        short = name.replace("layer", "L").replace("block_", "B")
        ax.plot(epochs, ratios, color=cmap(i), lw=1.2, alpha=0.8, label=short)

    ax.set_xlabel("Epoch"); ax.set_ylabel("erank / d^β")
    ax.set_title(f"Erank Convergence to Target — {arch_name} (β={beta})", fontweight="bold")
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=6, ncol=3 if n_layers > 6 else 2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"28b_erank_convergence_{arch_name.lower().replace('-','').replace('/','')}.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def worker_main(experiment_name):
    device = torch.device("cuda")
    tag = gpu_tag()
    print(f"{tag} Worker '{experiment_name}' started — {torch.cuda.get_device_name(0)}")

    if experiment_name == "resnet":
        results = experiment_resnet(device)
        out_path = CFG.OUTPUT_DIR / "erank_resnet18.json"
        plot_erank_dynamics(results, "ResNet-18", 0.7, CFG.OUTPUT_DIR)
        plot_erank_convergence(results, "ResNet-18", 0.7, CFG.OUTPUT_DIR)
    elif experiment_name == "vit":
        results = experiment_vit(device)
        out_path = CFG.OUTPUT_DIR / "erank_vit_small.json"
        plot_erank_dynamics(results, "ViT-Small/4", 0.5, CFG.OUTPUT_DIR)
        plot_erank_convergence(results, "ViT-Small/4", 0.5, CFG.OUTPUT_DIR)
    else:
        raise ValueError(f"Unknown: {experiment_name}")

    # Save (strip large history for JSON)
    save_data = {}
    for k, v in results.items():
        save_data[k] = {
            "best_acc": v["best_acc"],
            "beta": v["beta"],
            "use_ses": v["use_ses"],
            "erank_history": v["erank_history"],
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"{tag} Saved to {out_path}")


def orchestrator_main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Experiment 28: Effective Rank Dynamics                  ║")
    print("║   GPU 0: ResNet-18 CIFAR-100 (Baseline + SES β=0.7)           ║")
    print("║   GPU 1: ViT-Small/4 CIFAR-100 (Baseline + SES β=0.5)        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Download data
    root = str(CFG.DATA_DIR / "cifar100")
    torchvision.datasets.CIFAR100(root=root, train=True, download=True)

    script = os.path.abspath(__file__)
    if n_gpus >= 2:
        p0 = subprocess.Popen([sys.executable, script, "--worker", "resnet"],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                               stdout=sys.stdout, stderr=sys.stderr)
        p1 = subprocess.Popen([sys.executable, script, "--worker", "vit"],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
                               stdout=sys.stdout, stderr=sys.stderr)
        p0.wait(); p1.wait()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        worker_main("resnet")
        worker_main("vit")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 28 COMPLETE — {elapsed/60:.1f} min")
    print(f"Outputs: {CFG.OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["resnet", "vit"], default=None)
    args = parser.parse_args()
    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
