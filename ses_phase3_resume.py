#!/usr/bin/env python3
"""
Spectral Entropy Shaping (SES) — Phase 3: Practical Viability
==============================================================
Target: Kaggle Notebook, 1× NVIDIA T4 GPU
Estimated runtime: ~6-7 hours

Experiments:
  1. Periodic SES (every k steps, CIFAR-100)          → Reduce overhead from 50% to ~10%
  2. WideResNet-28-10 on CIFAR-100 (baseline + SES)   → Architecture generalization
  3. SES + Mixup/CutMix (CIFAR-100)                   → Interaction with modern augmentation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json
import math
import copy
import warnings
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
    OUTPUT_DIR = Path("/kaggle/working/outputs_phase3") if Path("/kaggle").exists() else Path("./outputs_phase3")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    EPOCHS = 50
    BATCH_SIZE = 128           # WRN-28-10 needs more memory; use 128 for all for fairness
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


def get_device():
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return torch.device("cuda")
    print("[INFO] No GPU — running on CPU")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


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
    anchor_device = torch.device("cuda:0") if activations[0].is_cuda else activations[0].device
    reg = torch.tensor(0.0, device=anchor_device)
    for H in activations:
        if H.device != anchor_device:
            H = H.to(anchor_device)
        if H.dim() > 2:
            H = H.mean(dim=[2, 3]) if H.dim() == 4 else H.flatten(1)
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


class ActivationCollector:
    """Collects activations from BasicBlock / Bottleneck / AdaptiveAvgPool2d."""

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
        print(f"  [Hooks] {len(self.hooks)} hooks: {', '.join(self.hook_names[:4])}{'...' if len(self.hook_names)>4 else ''}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            self.activations.append(output)

    def clear(self):
        self.activations = []

    def disable(self):
        self.enabled = False
        self.activations = []

    def enable(self):
        self.enabled = True

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class WideResNetCollector:
    """Collects activations from WideResNet basic blocks + final pool."""

    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []
        for name, module in model.named_modules():
            # WRN uses BasicBlock inside groups
            if isinstance(module, models.resnet.BasicBlock):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
        print(f"  [WRN Hooks] {len(self.hooks)} hooks: {', '.join(self.hook_names[:4])}{'...' if len(self.hook_names)>4 else ''}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            self.activations.append(output)

    def clear(self):
        self.activations = []

    def disable(self):
        self.enabled = False
        self.activations = []

    def enable(self):
        self.enabled = True

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_loaders(dataset="cifar100", batch_size=CFG.BATCH_SIZE, use_cutmix=False,
                use_mixup=False):
    if dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        DatasetClass = torchvision.datasets.CIFAR100
        num_classes = 100
    elif dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        DatasetClass = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    root = str(CFG.DATA_DIR / dataset)
    train_set = DatasetClass(root=root, train=True, download=True, transform=train_tf)
    test_set  = DatasetClass(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader, num_classes


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def make_resnet18(num_classes=100):
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_wide_resnet_28_10(num_classes=100):
    """WideResNet-28-10: depth=28, widen_factor=10.
    Uses wide_resnet50_2 as base and reconstructs for 28 layers.
    Actually, we build WRN-28-10 manually for CIFAR since torchvision
    doesn't have a direct WRN-28-10 for 32x32 images.
    """
    # WRN-28-10: 3 groups of 4 blocks each (4*3*2 + 4 = 28 layers), width multiplier 10
    # Base widths: [16, 16, 32, 64] -> with widen=10: [16, 160, 320, 640]

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
            assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
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

            # Init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
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
            out = self.fc(out)
            return out

    model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [WRN-28-10] {nparams:.1f}M parameters")
    return model


class WRNCollector:
    """Collects activations from WideResNet WideBasicBlocks + avgpool."""

    def __init__(self, model):
        self.activations = []
        self.hooks = []
        self.enabled = True
        self.hook_names = []

        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name == "WideBasicBlock":
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
                self.hook_names.append(name)

        print(f"  [WRN Hooks] {len(self.hooks)} hooks: {', '.join(self.hook_names[:4])}{'...' if len(self.hook_names)>4 else ''}")

    def _hook_fn(self, module, input, output):
        if self.enabled and isinstance(output, torch.Tensor):
            self.activations.append(output)

    def clear(self):
        self.activations = []

    def disable(self):
        self.enabled = False
        self.activations = []

    def enable(self):
        self.enabled = True

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ══════════════════════════════════════════════════════════════════════════════
# MIXUP / CUTMIX
# ══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Returns cutmix inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x_cut, y_a, y_b, lam


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
        pad = k // 2
        c = images.shape[1]
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
                correct += (model(X_c).argmax(1) == y).sum().item()
                total += y.size(0)
            results[f"{corr}_s{sev}"] = 100.0 * correct / total
    results["mean_corruption_acc"] = np.mean([v for k, v in results.items() if k != "mean_corruption_acc"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, collector=None,
                    lambda_ses=0.0, beta=0.7, ses_every_k=1, global_step=0):
    """Train one epoch. ses_every_k: apply SES loss every k steps (1=always)."""
    model.train()
    loss_m, task_m, reg_m = AverageMeter(), AverageMeter(), AverageMeter()
    correct = total = 0
    steps_with_ses = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        global_step += 1

        # Decide whether to apply SES this step
        apply_ses = (lambda_ses > 0 and collector is not None and global_step % ses_every_k == 0)

        if collector:
            if apply_ses:
                collector.enable()
            else:
                collector.disable()
            collector.clear()

        output = model(X)
        task_loss = F.cross_entropy(output, y)

        reg_loss = torch.tensor(0.0, device=device)
        if apply_ses and collector.activations:
            reg_loss = lambda_ses * ses_regularizer(collector.activations, beta=beta).to(device)
            steps_with_ses += 1

        total_loss = task_loss + reg_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        bs = X.size(0)
        loss_m.update(total_loss.item(), bs)
        task_m.update(task_loss.item(), bs)
        reg_m.update(reg_loss.item(), bs)
        correct += (output.argmax(1) == y).sum().item()
        total += bs

    return {"loss": loss_m.avg, "task_loss": task_m.avg, "reg_loss": reg_m.avg,
            "acc": 100.0 * correct / total, "global_step": global_step,
            "ses_steps": steps_with_ses}


def train_one_epoch_augmix(model, loader, optimizer, device, collector=None,
                           lambda_ses=0.0, beta=0.7, aug_type="none"):
    """Train one epoch with optional Mixup or CutMix."""
    model.train()
    loss_m, task_m, reg_m = AverageMeter(), AverageMeter(), AverageMeter()
    correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if collector:
            collector.clear()

        # Apply augmentation
        if aug_type == "mixup":
            X, y_a, y_b, lam = mixup_data(X, y, alpha=1.0)
        elif aug_type == "cutmix":
            X, y_a, y_b, lam = cutmix_data(X, y, alpha=1.0)

        output = model(X)

        if aug_type in ("mixup", "cutmix"):
            task_loss = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam)
        else:
            task_loss = F.cross_entropy(output, y)

        reg_loss = torch.tensor(0.0, device=device)
        if collector and lambda_ses > 0 and collector.activations:
            reg_loss = lambda_ses * ses_regularizer(collector.activations, beta=beta).to(device)

        total_loss = task_loss + reg_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        bs = X.size(0)
        loss_m.update(total_loss.item(), bs)
        task_m.update(task_loss.item(), bs)
        reg_m.update(reg_loss.item(), bs)

        # For mixed labels, count based on dominant label
        if aug_type in ("mixup", "cutmix"):
            correct += (lam * (output.argmax(1) == y_a).float()
                        + (1 - lam) * (output.argmax(1) == y_b).float()).sum().item()
        else:
            correct += (output.argmax(1) == y).sum().item()
        total += bs

    return {"loss": loss_m.avg, "task_loss": task_m.avg, "reg_loss": reg_m.avg,
            "acc": 100.0 * correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m = AverageMeter()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        output = model(X)
        loss_m.update(F.cross_entropy(output, y).item(), X.size(0))
        correct += (output.argmax(1) == y).sum().item()
        total += X.size(0)
    return {"loss": loss_m.avg, "acc": 100.0 * correct / total}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: PERIODIC SES
# ══════════════════════════════════════════════════════════════════════════════

def experiment_periodic_ses(device):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Periodic SES (CIFAR-100)")
    print("  k=1 (every step), k=3, k=5, k=10 + baseline")
    print("=" * 70)

    set_seed(CFG.SEED)
    train_loader, test_loader, _ = get_loaders("cifar100")

    results = {}
    k_values = [0, 1, 3, 5, 10]  # 0 = baseline (no SES)

    for k in k_values:
        label = "Baseline" if k == 0 else f"SES-k{k}"
        use_ses = (k > 0)

        print(f"\n--- {label} ---")
        set_seed(CFG.SEED)
        model = make_resnet18(num_classes=100).to(device)
        optimizer = optim.SGD(model.parameters(), lr=CFG.LR,
                              momentum=CFG.MOMENTUM, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG.LR_MILESTONES,
                                                    gamma=CFG.LR_GAMMA)

        collector = None
        if use_ses:
            collector = ActivationCollector(model)

        history = defaultdict(list)
        best_test_acc = 0.0
        best_state = None
        global_step = 0
        epoch_times = []

        pbar = tqdm(range(CFG.EPOCHS), desc=f"[{label}]", unit="ep", leave=True)
        for epoch in pbar:
            t0 = time.time()
            train_res = train_one_epoch(model, train_loader, optimizer, device,
                                         collector=collector,
                                         lambda_ses=CFG.LAMBDA_SES if use_ses else 0.0,
                                         beta=CFG.BETA, ses_every_k=max(k, 1),
                                         global_step=global_step)
            global_step = train_res["global_step"]
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
            pbar.set_postfix_str(f"Te {test_res['acc']:.1f}% | {epoch_time:.1f}s/ep")

        if collector:
            collector.remove()
        model.load_state_dict(best_state)

        # Evaluate robustness
        rob = evaluate_robustness(model, test_loader, device)

        avg_time = np.mean(epoch_times)
        gap = history["train_acc"][-1] - history["test_acc"][-1]

        results[f"k{k}"] = {
            "label": label,
            "k": k,
            "best_acc": best_test_acc,
            "final_gap": gap,
            "rob_mean": rob["mean_corruption_acc"],
            "avg_epoch_time": avg_time,
            "history": dict(history),
        }
        print(f"  [{label}] Acc: {best_test_acc:.2f}% | Gap: {gap:.2f}pp | "
              f"Rob: {rob['mean_corruption_acc']:.2f}% | Time: {avg_time:.1f}s/ep")

        del model, optimizer, scheduler, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: WIDERESNET-28-10
# ══════════════════════════════════════════════════════════════════════════════

def experiment_wideresnet(device):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: WideResNet-28-10 on CIFAR-100")
    print("=" * 70)

    set_seed(CFG.SEED)
    train_loader, test_loader, _ = get_loaders("cifar100", batch_size=CFG.BATCH_SIZE)

    results = {}

    for use_ses, label in [(False, "WRN-Baseline"), (True, "WRN-SES")]:
        print(f"\n--- {label} ---")
        set_seed(CFG.SEED)
        model = make_wide_resnet_28_10(num_classes=100).to(device)

        optimizer = optim.SGD(model.parameters(), lr=CFG.LR,
                              momentum=CFG.MOMENTUM, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG.LR_MILESTONES,
                                                    gamma=CFG.LR_GAMMA)

        collector = None
        if use_ses:
            collector = WRNCollector(model)

        history = defaultdict(list)
        best_test_acc = 0.0
        best_state = None

        pbar = tqdm(range(CFG.EPOCHS), desc=f"[{label}]", unit="ep", leave=True)
        for epoch in pbar:
            # Standard training loop
            model.train()
            loss_m, reg_m = AverageMeter(), AverageMeter()
            correct = total = 0

            for X, y in train_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if collector:
                    collector.clear()

                output = model(X)
                task_loss = F.cross_entropy(output, y)

                reg_loss = torch.tensor(0.0, device=device)
                if collector and use_ses and collector.activations:
                    reg_loss = CFG.LAMBDA_SES * ses_regularizer(
                        collector.activations, beta=CFG.BETA).to(device)

                total_loss = task_loss + reg_loss
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                bs = X.size(0)
                loss_m.update(total_loss.item(), bs)
                reg_m.update(reg_loss.item(), bs)
                correct += (output.argmax(1) == y).sum().item()
                total += bs

            train_acc = 100.0 * correct / total

            if collector:
                collector.disable()
            test_res = evaluate(model, test_loader, device)
            if collector:
                collector.enable()
            scheduler.step()

            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_res["acc"])

            if test_res["acc"] > best_test_acc:
                best_test_acc = test_res["acc"]
                best_state = copy.deepcopy(model.state_dict())

            gap = train_acc - test_res["acc"]
            pbar.set_postfix_str(f"Tr {train_acc:.1f}% | Te {test_res['acc']:.1f}% | Gap {gap:.1f}")

        if collector:
            collector.remove()
        model.load_state_dict(best_state)

        rob = evaluate_robustness(model, test_loader, device)
        gap = history["train_acc"][-1] - history["test_acc"][-1]

        key = "ses" if use_ses else "baseline"
        results[key] = {
            "best_acc": best_test_acc,
            "final_gap": gap,
            "rob_mean": rob["mean_corruption_acc"],
            "history": dict(history),
        }
        print(f"  [{label}] Acc: {best_test_acc:.2f}% | Gap: {gap:.2f}pp | Rob: {rob['mean_corruption_acc']:.2f}%")

        del model, optimizer, scheduler, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: SES + MIXUP / CUTMIX
# ══════════════════════════════════════════════════════════════════════════════

def experiment_augmentation(device):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SES + Mixup / CutMix (CIFAR-100)")
    print("=" * 70)

    set_seed(CFG.SEED)
    train_loader, test_loader, _ = get_loaders("cifar100")

    configs = [
        ("baseline",       False, "none",   "Baseline"),
        ("ses",            True,  "none",   "SES"),
        ("mixup",          False, "mixup",  "Mixup"),
        ("ses_mixup",      True,  "mixup",  "SES + Mixup"),
        ("cutmix",         False, "cutmix", "CutMix"),
        ("ses_cutmix",     True,  "cutmix", "SES + CutMix"),
    ]

    results = {}

    for key, use_ses, aug_type, label in configs:
        print(f"\n--- {label} ---")
        set_seed(CFG.SEED)
        model = make_resnet18(num_classes=100).to(device)
        optimizer = optim.SGD(model.parameters(), lr=CFG.LR,
                              momentum=CFG.MOMENTUM, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG.LR_MILESTONES,
                                                    gamma=CFG.LR_GAMMA)

        collector = None
        if use_ses:
            collector = ActivationCollector(model)

        history = defaultdict(list)
        best_test_acc = 0.0
        best_state = None

        pbar = tqdm(range(CFG.EPOCHS), desc=f"[{label}]", unit="ep", leave=True)
        for epoch in pbar:
            train_res = train_one_epoch_augmix(model, train_loader, optimizer, device,
                                                collector=collector,
                                                lambda_ses=CFG.LAMBDA_SES if use_ses else 0.0,
                                                beta=CFG.BETA, aug_type=aug_type)
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
            pbar.set_postfix_str(f"Te {test_res['acc']:.1f}%")

        if collector:
            collector.remove()
        model.load_state_dict(best_state)

        rob = evaluate_robustness(model, test_loader, device)
        gap = history["train_acc"][-1] - history["test_acc"][-1]

        results[key] = {
            "best_acc": best_test_acc,
            "final_gap": gap,
            "rob_mean": rob["mean_corruption_acc"],
            "history": dict(history),
        }
        print(f"  [{label}] Acc: {best_test_acc:.2f}% | Gap: {gap:.2f}pp | Rob: {rob['mean_corruption_acc']:.2f}%")

        del model, optimizer, scheduler, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_periodic_ses(results, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Experiment 1: Periodic SES — Overhead vs Performance (CIFAR-100)",
                 fontsize=14, fontweight="bold")

    k_vals = sorted(results.keys(), key=lambda x: int(x[1:]))
    labels = [results[k]["label"] for k in k_vals]
    colors = ["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]

    # (a) Accuracy
    ax = axes[0]
    accs = [results[k]["best_acc"] for k in k_vals]
    bars = ax.bar(range(len(labels)), accs, color=colors[:len(labels)], alpha=0.8)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Best Test Acc (%)"); ax.set_title("Accuracy"); ax.grid(True, alpha=0.3, axis="y")

    # (b) Robustness
    ax = axes[1]
    robs = [results[k]["rob_mean"] for k in k_vals]
    bars = ax.bar(range(len(labels)), robs, color=colors[:len(labels)], alpha=0.8)
    for bar, val in zip(bars, robs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean Corr. Acc (%)"); ax.set_title("Robustness"); ax.grid(True, alpha=0.3, axis="y")

    # (c) Time/epoch
    ax = axes[2]
    times = [results[k]["avg_epoch_time"] for k in k_vals]
    baseline_time = times[0]
    overhead = [(t / baseline_time - 1) * 100 for t in times]
    bars = ax.bar(range(len(labels)), times, color=colors[:len(labels)], alpha=0.8)
    for bar, val, ov in zip(bars, times, overhead):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}s\n(+{ov:.0f}%)" if ov > 0 else f"{val:.1f}s",
                ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Seconds / Epoch"); ax.set_title("Computational Cost"); ax.grid(True, alpha=0.3, axis="y")

    # (d) Pareto: overhead vs robustness
    ax = axes[3]
    for i, k in enumerate(k_vals):
        ov = (results[k]["avg_epoch_time"] / baseline_time - 1) * 100
        rob = results[k]["rob_mean"]
        ax.scatter(ov, rob, color=colors[i], s=100, zorder=5)
        ax.annotate(results[k]["label"], (ov, rob), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Overhead (%)"); ax.set_ylabel("Mean Corr. Acc (%)")
    ax.set_title("Pareto: Cost vs Robustness"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "14_periodic_ses.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] 14_periodic_ses.png")


def plot_wideresnet(results, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 2: WideResNet-28-10 on CIFAR-100 (36.5M params)",
                 fontsize=14, fontweight="bold")

    methods = ["Baseline", "SES"]
    keys = ["baseline", "ses"]
    colors = ["#377eb8", "#e41a1c"]

    for idx, (metric, ylabel, extract) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [extract(results[k]) for k in keys]
        bars = ax.bar(range(2), vals, color=colors, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(2)); ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel); ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "15_wideresnet.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] 15_wideresnet.png")

    # Training curves
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
    plt.savefig(save_dir / "15b_wideresnet_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] 15b_wideresnet_curves.png")


def plot_augmentation(results, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Experiment 3: SES + Mixup / CutMix (CIFAR-100)",
                 fontsize=14, fontweight="bold")

    methods = ["Baseline", "SES", "Mixup", "SES+Mixup", "CutMix", "SES+CutMix"]
    keys = ["baseline", "ses", "mixup", "ses_mixup", "cutmix", "ses_cutmix"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]

    for idx, (metric, ylabel, extract) in enumerate([
        ("Best Test Accuracy", "Acc (%)", lambda r: r["best_acc"]),
        ("Generalization Gap", "Gap (pp)", lambda r: r["final_gap"]),
        ("Corruption Robustness", "Mean Corr. Acc (%)", lambda r: r["rob_mean"]),
    ]):
        ax = axes[idx]
        vals = [extract(results[k]) for k in keys]
        bars = ax.bar(range(len(methods)), vals, color=colors, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{val:.2f}", ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "16_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] 16_augmentation.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SES — Phase 3: Practical Viability                       ║")
    print("║   Periodic SES · WideResNet · Mixup/CutMix                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    device = get_device()
    t_start = time.time()

    # ── Experiment 1: SKIPPED (already completed) ────────
    periodic_results = None
    print("\n[SKIP] Experiment 1 (Periodic SES) already completed — skipping.")

    # ── Experiment 2: WideResNet-28-10 ───────────────────
    wrn_results = experiment_wideresnet(device)
    plot_wideresnet(wrn_results, CFG.OUTPUT_DIR)

    # ── Experiment 3: Augmentation interaction ───────────
    aug_results = experiment_augmentation(device)
    plot_augmentation(aug_results, CFG.OUTPUT_DIR)

    # ── Save JSON ────────────────────────────────────────
    def strip_history(d):
        return {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                for k, v in d.items()} if isinstance(d, dict) else d

    json_out = {
        "periodic_ses": "SKIPPED — run separately",
        "wideresnet": strip_history(wrn_results),
        "augmentation": strip_history(aug_results),
    }

    with open(CFG.OUTPUT_DIR / "phase3_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    elapsed = time.time() - t_start

    # ── Final summary ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PHASE 3 (RESUME) COMPLETE — {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"{'=' * 70}")

    print("\n┌─ WIDERESNET-28-10 ────────────────────────────────────────────┐")
    for key, label in [("baseline", "WRN-Base"), ("ses", "WRN-SES")]:
        r = wrn_results[key]
        print(f"│  {label:<12} Acc: {r['best_acc']:.2f}%  Gap: {r['final_gap']:.2f}pp  "
              f"Rob: {r['rob_mean']:.2f}%                  │")

    print("├─ AUGMENTATION ────────────────────────────────────────────────┤")
    for key, label in [("baseline", "Base"), ("ses", "SES"), ("mixup", "Mixup"),
                        ("ses_mixup", "SES+Mix"), ("cutmix", "CutMix"), ("ses_cutmix", "SES+Cut")]:
        r = aug_results[key]
        print(f"│  {label:<12} Acc: {r['best_acc']:.2f}%  Gap: {r['final_gap']:.2f}pp  "
              f"Rob: {r['rob_mean']:.2f}%                  │")
    print("└────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs saved to: {CFG.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
