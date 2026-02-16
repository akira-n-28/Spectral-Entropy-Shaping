#!/usr/bin/env python3
"""
SES — Experiment 22: BERT Fine-Tuning on NLP Tasks (2x T4 16GB)
================================================================
First application of SES to NLP: fine-tune BERT-base on SST-2 (sentiment)
and MRPC (paraphrase detection) from GLUE benchmark.

GPU 0 (3 configs — SST-2):
  1. sst2_baseline   — No SES
  2. sst2_ses_l4     — SES Last-4 layers β=0.5
  3. sst2_ses_all    — SES All-12 layers β=0.5

GPU 1 (3 configs — MRPC):
  4. mrpc_baseline   — No SES
  5. mrpc_ses_l4     — SES Last-4 layers β=0.5
  6. mrpc_ses_all    — SES All-12 layers β=0.5

Hooks on BERT encoder.layer[i] outputs, mean-pool over seq_len → [B, 768].
AdamW lr=2e-5, batch 32, 3 epochs, linear warmup 6%.

Usage:
  python ses_bert_nlp.py                    # Launches both workers
  python ses_bert_nlp.py --worker gpu0      # Internal: GPU0 worker (SST-2)
  python ses_bert_nlp.py --worker gpu1      # Internal: GPU1 worker (MRPC)
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
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    OUTPUT_DIR = Path("/kaggle/working/outputs_bert") if Path("/kaggle").exists() else Path("./outputs_bert")
    DATA_DIR = Path("/kaggle/working/data") if Path("/kaggle").exists() else Path("./data")

    EPOCHS = 3
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    LR = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.06
    MAX_SEQ_LEN = 128

    LAMBDA_SES = 0.01
    BETA = 0.5
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
            H = H.mean(dim=1) if H.dim() == 3 else H.flatten(1)
        H = H.float()
        d = H.shape[1]
        entropy, _ = spectral_entropy(H, eps)
        target = beta * math.log(d)
        reg = reg + (entropy - target) ** 2
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# DATA — GLUE SST-2 & MRPC via HuggingFace datasets
# ══════════════════════════════════════════════════════════════════════════════

def load_glue_dataset(task_name, tokenizer, max_len=128):
    """Load GLUE task using HuggingFace datasets library."""
    from datasets import load_dataset

    dataset = load_dataset("glue", task_name)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    def tokenize_fn(examples):
        if task_name == "sst2":
            return tokenizer(examples["sentence"], truncation=True,
                             max_length=max_len, padding="max_length")
        elif task_name == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"],
                             truncation=True, max_length=max_len, padding="max_length")
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds


def get_glue_loaders(task_name, tokenizer, batch_size=None):
    if batch_size is None:
        batch_size = CFG.BATCH_SIZE

    train_ds, val_ds = load_glue_dataset(task_name, tokenizer, CFG.MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"  [{task_name.upper()}] Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════════
# BERT COLLECTOR — hooks on encoder layer outputs
# ══════════════════════════════════════════════════════════════════════════════

class BertLayerCollector:
    """
    Hook on BERT encoder.layer[i] outputs.
    Each layer output is a tuple: (hidden_states, ...).
    We take hidden_states [B, seq, 768] and mean-pool → [B, 768].
    """

    def __init__(self, model, hook_mode="last_4"):
        self.activations = []
        self.hooks = []
        self.enabled = True

        # model.bert.encoder.layer is the list of BertLayer modules
        encoder_layers = list(model.bert.encoder.layer)
        n = len(encoder_layers)

        if hook_mode == "last_4":
            indices = list(range(max(0, n - 4), n))
        elif hook_mode == "all":
            indices = list(range(n))
        else:
            raise ValueError(f"Unknown hook_mode: {hook_mode}")

        for idx in indices:
            self.hooks.append(encoder_layers[idx].register_forward_hook(self._hook_fn))

        print(f"  [BertCollector] {len(self.hooks)} hooks ({hook_mode}): layers {indices}")

    def _hook_fn(self, module, input, output):
        if not self.enabled:
            return
        # BertLayer output is a tuple: (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output
        # [B, seq, 768] → mean-pool over seq → [B, 768]
        pooled = hidden.mean(dim=1)
        self.activations.append(pooled)

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
# LR SCHEDULER — linear warmup + linear decay
# ══════════════════════════════════════════════════════════════════════════════

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * max(0.0, 1.0 - progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    collector=None, lambda_ses=0.0, beta=0.5):
    model.train()
    loss_m = AverageMeter()
    correct = total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if collector:
            collector.clear()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            task_loss = outputs.loss

        reg_loss = torch.tensor(0.0, device=device)
        if collector and lambda_ses > 0 and collector.activations:
            reg_loss = lambda_ses * ses_regularizer(collector.activations, beta=beta)

        total_loss = task_loss.float() + reg_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        bs = labels.size(0)
        loss_m.update(total_loss.item(), bs)
        correct += (preds == labels).sum().item()
        total += bs

    return {"loss": loss_m.avg, "acc": 100.0 * correct / total}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m = AverageMeter()
    correct = total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_m.update(outputs.loss.item(), labels.size(0))

        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": loss_m.avg, "acc": 100.0 * correct / total}


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC TRAINING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_config(task_name, model, collector, device,
                      train_loader, val_loader, label, beta=0.5, lambda_ses=0.0):
    tag = gpu_tag()

    # Optimizer — differential LR: lower for BERT, higher for classifier
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": CFG.WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_params, lr=CFG.LR)

    total_steps = len(train_loader) * CFG.EPOCHS
    warmup_steps = int(total_steps * CFG.WARMUP_RATIO)
    scheduler = LinearWarmupScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    history = defaultdict(list)
    best_val_acc = 0.0
    best_state = None
    epoch_times = []

    pbar = tqdm(range(CFG.EPOCHS), desc=f"{tag} {label}", unit="ep", leave=True)
    for epoch in pbar:
        t0 = time.time()

        train_res = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            collector=collector, lambda_ses=lambda_ses, beta=beta)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        if collector:
            collector.disable()
        val_res = evaluate(model, val_loader, device)
        if collector:
            collector.enable()

        history["train_acc"].append(train_res["acc"])
        history["val_acc"].append(val_res["acc"])
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])

        if val_res["acc"] > best_val_acc:
            best_val_acc = val_res["acc"]
            best_state = copy.deepcopy(model.state_dict())

        gap = train_res["acc"] - val_res["acc"]
        pbar.set_postfix_str(f"Val {val_res['acc']:.1f}% Gap {gap:.1f} {epoch_time:.1f}s")

    if collector:
        collector.remove()
    model.load_state_dict(best_state)

    # Final eval
    final_val = evaluate(model, val_loader, device)
    gap = history["train_acc"][-1] - history["val_acc"][-1]

    result = {
        "task": task_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val["acc"],
        "final_gap": gap,
        "avg_epoch_time": float(np.mean(epoch_times)),
        "vram_peak_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "beta": beta,
        "lambda_ses": lambda_ses,
        "history": dict(history),
    }

    print(f"{tag} {label} DONE: Val Acc {best_val_acc:.2f}% | Gap {gap:.2f}pp | "
          f"{np.mean(epoch_times):.1f}s/ep")

    del optimizer, scheduler, scaler
    return result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS — SST-2 (GPU 0) and MRPC (GPU 1)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_task(task_name, device):
    """Run 3 configs for a given GLUE task: Baseline, SES Last-4, SES All-12."""
    from transformers import BertTokenizer, BertForSequenceClassification

    tag = gpu_tag()
    num_labels = 2  # both SST-2 and MRPC are binary

    print(f"\n{tag} {'=' * 55}")
    print(f"{tag} BERT NLP — {task_name.upper()}")
    print(f"{tag} {'=' * 55}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, val_loader = get_glue_loaders(task_name, tokenizer)

    configs = [
        # (key, hook_mode, lambda_ses, label)
        (f"{task_name}_baseline", None,     0.0,            f"{task_name.upper()} Baseline"),
        (f"{task_name}_ses_l4",   "last_4", CFG.LAMBDA_SES, f"{task_name.upper()} SES L4 β={CFG.BETA}"),
        (f"{task_name}_ses_all",  "all",    CFG.LAMBDA_SES, f"{task_name.upper()} SES All β={CFG.BETA}"),
    ]

    results = {}
    for key, hook_mode, lambda_ses, label in configs:
        print(f"\n{tag} --- {label} ---")
        set_seed(CFG.SEED)

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels).to(device)
        nparams = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [BERT-base] {nparams:.1f}M params")

        collector = None
        if hook_mode is not None:
            collector = BertLayerCollector(model, hook_mode=hook_mode)

        results[key] = run_single_config(
            task_name, model, collector, device,
            train_loader, val_loader,
            label=label, beta=CFG.BETA, lambda_ses=lambda_ses)

        del model, collector
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_task_results(results, task_name, save_dir):
    """Bar chart comparing Baseline vs SES configs for one task."""
    order = [f"{task_name}_baseline", f"{task_name}_ses_l4", f"{task_name}_ses_all"]
    labels = ["Baseline", "SES Last-4", "SES All-12"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a"]

    present = [k for k in order if k in results]
    present_labels = [labels[order.index(k)] for k in present]
    present_colors = [colors[order.index(k)] for k in present]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Exp 22: SES on BERT — {task_name.upper()} (β={CFG.BETA})",
                 fontsize=14, fontweight="bold")

    # Panel 1: Validation accuracy
    ax = axes[0]
    vals = [results[k]["best_val_acc"] for k in present]
    bars = ax.bar(range(len(present)), vals, color=present_colors, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present_labels, fontsize=10)
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Best Validation Accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Training curves
    ax = axes[1]
    for i, key in enumerate(present):
        h = results[key]["history"]
        ax.plot(h["val_acc"], color=present_colors[i], lw=2,
                label=f"{present_labels[i]} ({results[key]['best_val_acc']:.2f}%)")
        ax.plot(h["train_acc"], color=present_colors[i], ls="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train (dashed) vs Val (solid)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"22_{task_name}_bert_ses.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {fname}")


def plot_combined_results(all_results, save_dir):
    """Combined view of both tasks."""
    tasks = ["sst2", "mrpc"]
    task_labels = ["SST-2\n(Sentiment)", "MRPC\n(Paraphrase)"]
    configs = ["baseline", "ses_l4", "ses_all"]
    config_labels = ["Baseline", "SES Last-4", "SES All-12"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tasks))
    n = len(configs)
    w = 0.25

    for i, (cfg, lbl, col) in enumerate(zip(configs, config_labels, colors)):
        vals = []
        for task in tasks:
            key = f"{task}_{cfg}"
            if key in all_results:
                vals.append(all_results[key]["best_val_acc"])
            else:
                vals.append(0)
        bars = ax.bar(x + (i - n/2 + 0.5) * w, vals, w, color=col,
                      alpha=0.85, label=lbl)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=12)
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Exp 22: SES on BERT-base — GLUE Tasks (β=0.5, 3 epochs)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = "22_bert_combined.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {fname}")

    # Delta vs baseline
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    w = 0.3

    for i, (cfg, lbl, col) in enumerate(zip(["ses_l4", "ses_all"],
                                              ["SES Last-4", "SES All-12"],
                                              ["#e41a1c", "#4daf4a"])):
        deltas = []
        for task in tasks:
            base_key = f"{task}_baseline"
            ses_key = f"{task}_{cfg}"
            if base_key in all_results and ses_key in all_results:
                deltas.append(all_results[ses_key]["best_val_acc"] -
                              all_results[base_key]["best_val_acc"])
            else:
                deltas.append(0)
        bars = ax.bar(x + (i - 0.5) * w, deltas, w, color=col, alpha=0.85, label=lbl)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.05 if val >= 0 else -0.15),
                    f"{val:+.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=12)
    ax.set_ylabel("Δ Val Accuracy vs Baseline (pp)")
    ax.set_title("Exp 22: SES Improvement over BERT Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = "22b_bert_delta.png"
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
        results = experiment_task("sst2", device)
        out_path = CFG.OUTPUT_DIR / "gpu0_sst2_results.json"
    elif experiment_name == "gpu1":
        results = experiment_task("mrpc", device)
        out_path = CFG.OUTPUT_DIR / "gpu1_mrpc_results.json"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    # Save results (strip history for compact JSON)
    save_data = {}
    for k, v in results.items():
        save_data[k] = {kk: vv for kk, vv in v.items() if kk != "history"}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"{tag} Results saved to {out_path}")

    # Also save full results with history
    full_path = out_path.with_name(out_path.stem + "_full.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def orchestrator_main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   SES — Experiment 22: BERT Fine-Tuning on NLP Tasks           ║")
    print("║   GPU 0: SST-2 (Baseline / SES L4 / SES All-12)               ║")
    print("║   GPU 1: MRPC  (Baseline / SES L4 / SES All-12)               ║")
    print("║   BERT-base-uncased, AdamW lr=2e-5, FP16, 3 epochs            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"\n[INFO] {n_gpus} GPU(s) detected:")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # Pre-download model and tokenizer
    print("\n[DATA] Pre-downloading BERT-base-uncased...")
    from transformers import BertTokenizer, BertForSequenceClassification
    BertTokenizer.from_pretrained("bert-base-uncased")
    BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    print("[DATA] BERT ready.")

    # Pre-download datasets
    print("[DATA] Pre-downloading GLUE datasets...")
    from datasets import load_dataset
    load_dataset("glue", "sst2")
    load_dataset("glue", "mrpc")
    print("[DATA] GLUE datasets ready.")

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
    gpu0_path = CFG.OUTPUT_DIR / "gpu0_sst2_results.json"
    gpu1_path = CFG.OUTPUT_DIR / "gpu1_mrpc_results.json"

    all_results = {}
    full_results = {}
    for path in [gpu0_path, gpu1_path]:
        if path.exists():
            with open(path) as f:
                all_results.update(json.load(f))
        full_path = path.with_name(path.stem + "_full.json")
        if full_path.exists():
            with open(full_path) as f:
                full_results.update(json.load(f))

    if all_results:
        summary_path = CFG.OUTPUT_DIR / "bert_nlp_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[SAVE] {summary_path}")

    if full_results:
        # Plot
        for task in ["sst2", "mrpc"]:
            task_res = {k: v for k, v in full_results.items() if k.startswith(task)}
            if task_res:
                plot_task_results(task_res, task, CFG.OUTPUT_DIR)
        plot_combined_results(full_results, CFG.OUTPUT_DIR)

    elapsed = time.time() - t_start

    # ── FINAL REPORT ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 22: BERT NLP SES COMPLETE — {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    if all_results:
        print(f"\n┌─ BERT-base GLUE (β={CFG.BETA}, FP16) ─────────────────────────────┐")
        for task in ["sst2", "mrpc"]:
            base_key = f"{task}_baseline"
            if base_key in all_results:
                base_acc = all_results[base_key]["best_val_acc"]
                print(f"│  {task.upper()} Baseline:       Val Acc: {base_acc:.2f}%")
                for cfg_name, cfg_label in [("ses_l4", "SES Last-4"), ("ses_all", "SES All-12")]:
                    key = f"{task}_{cfg_name}"
                    if key in all_results:
                        acc = all_results[key]["best_val_acc"]
                        delta = acc - base_acc
                        print(f"│  {task.upper()} {cfg_label:<14} Val Acc: {acc:.2f}% ({delta:+.2f}pp)")
            print(f"│")
        print(f"└──────────────────────────────────────────────────────────────────┘")

    print(f"\nOutputs: {CFG.OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SES Exp 22: BERT NLP")
    parser.add_argument("--worker", choices=["gpu0", "gpu1"], default=None,
                        help="Internal: run as worker on assigned GPU")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
    else:
        orchestrator_main()
