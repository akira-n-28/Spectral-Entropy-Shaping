# CONTEXT.md — Spectral Entropy Shaping (SES) Project

**Autore**: Davide Le Bone
**Ultimo aggiornamento**: 15 febbraio 2026
**Paper**: 22+ pagine, 13 esperimenti, 30+ references

---

## 1. Stato attuale del progetto

### Cos'è SES

Regolarizzatore che penalizza le deviazioni dell'entropia spettrale delle attivazioni layer-wise da un target:

```
R_SES = Σ_l α_l (H^(l) - β·log d_l)²
```

Un solo iperparametro interpretabile: β ∈ (0,1). Implementazione ~10 righe PyTorch con forward hook. Due teoremi: (1) generalization bound con effective rank, (2) Lipschitz stability bound.

### Risultati completi

| Fase | Exp | Descrizione | Status | Risultato chiave |
|------|-----|-------------|--------|------------------|
| P0 | 1 | Training dynamics (CIFAR-10, seed 42) | ✅ | SES 92.91% vs Base 91.54% (single seed) |
| P0 | 2 | Beta sweep {0.3, 0.5, 0.7, 0.9} | ✅ | Tutti 91.6-92.6% → robusto a β |
| P0 | 3 | Corruption robustness (CIFAR-10) | ✅ | SES +0.99pp |
| P0 | 4 | Toy 2D Jacobian | ✅ | κ(J) ridotto 2.45× (57.3→23.4) |
| P1 | 5 | Multi-seed CIFAR-10 (3 seed) | ✅ | Acc pari (93.43±0.10 vs 93.37±0.23), Rob +0.45pp |
| P1 | 6 | Multi-seed CIFAR-100 (3 seed) | ✅ | Acc +0.47pp (var 2.6× più bassa), Gap -1.3%, Rob +0.65pp |
| P1 | 7 | Lambda ablation {0.001-0.1} | ✅ | λ=0.001→best acc, λ=0.05→best rob (+2.1pp) |
| P2 | 8 | SES vs Spectral Norm (CIFAR-100) | ✅ | SES batte SN su tutto: +0.24pp acc, +0.55pp rob |
| P2 | 9 | Layer hooking ablation (CIFAR-10) | ✅ | Last 3→best acc/gap, All 9→best rob |
| P3 | 10 | Periodic SES k={1,3,5,10} (CIFAR-100) | ✅ | **k=10: +0.86pp acc, +0.96pp rob, solo 19% overhead** |
| P3 | 11 | ResNet-50 Tiny-ImageNet (200 cls, 64×64) | ✅ | **+2.58pp acc, -6.1% gap, +2.37pp rob** |
| P3 | 12 | SES + Mixup/CutMix (CIFAR-100) | ✅ | **SES ortogonale: +0.26-0.79pp acc, +0.61-0.94pp rob su ogni combinazione** |
| P3 | 13 | WideResNet-28-10 (CIFAR-100) | ✅ | Acc pari (−0.31pp), Rob +0.54pp — neutro su architetture large |

### Trend chiave: benefici scalano con difficoltà del task

| Dataset | Classi | Architettura | Δ Acc | Δ Rob |
|---------|--------|--------------|-------|-------|
| CIFAR-10 | 10 | ResNet-18 | −0.06pp | +0.45pp |
| CIFAR-100 | 100 | ResNet-18 | +0.47pp | +0.65pp |
| CIFAR-100 | 100 | WRN-28-10 (36.5M) | −0.31pp | +0.54pp |
| Tiny-ImageNet | 200 | ResNet-50 | **+2.58pp** | **+2.37pp** |

### SES è ortogonale alle augmentation moderne (Exp 12)

| Combinazione | Δ Acc | Δ Rob | Interpretazione |
|---|---|---|---|
| None → SES | +0.34pp | +0.87pp | SES standalone |
| Mixup → SES+Mixup | +0.26pp | +0.94pp | SES aggiunge su Mixup |
| CutMix → SES+CutMix | **+0.79pp** | +0.61pp | SES aggiunge su CutMix |

Best overall: **SES+CutMix** 73.86% acc, **SES+Mixup** 48.21% rob

### Predizioni teoriche

| ID | Predizione | Status |
|----|-----------|--------|
| P1 | Reduced generalization gap | ✅ (scala con difficoltà: 0%→-1.3%→-6.1%) |
| P2 | Improved robustness | ✅ (consistente su tutti i dataset e architetture) |
| P3 | Controllable effective rank | ✅ |
| P4 | Reduced Jacobian κ | ✅ (2.45× riduzione) |
| P5 | No collapse | ✅ |
| P6 | β controls dynamics | ∼ (parziale) |

---

## 2. File importanti e funzione

### Paper

| File | Funzione |
|------|----------|
| `spectral_entropy_shaping.tex` | Sorgente LaTeX del paper (22 pagine, 11 esperimenti) |
| `spectral_entropy_shaping.pdf` | PDF compilato |
| `arxiv_submission.zip` | Pacchetto pronto per arXiv (.tex + 13 figure .png) |

### Figure (nel paper)

| File | Esperimento |
|------|-------------|
| `01_baseline_vs_ses.png` | Exp 1: training dynamics CIFAR-10 |
| `02_beta_sweep.png` | Exp 2: sweep β |
| `03_corruption_robustness.png` | Exp 3: corruption robustness CIFAR-10 |
| `04_toy_2d.png` | Exp 4: toy Jacobian 2D |
| `06_multiseed_cifar10.png` | Exp 5: multi-seed CIFAR-10 |
| `08_lambda_ablation.png` | Exp 7: lambda ablation |
| `10_multiseed_cifar100.png` | Exp 6: multi-seed CIFAR-100 |
| `11_sn_cifar100.png` | Exp 8: SES vs Spectral Norm |
| `11c_sn_cifar100_percorruption.png` | Exp 8: per-corruption breakdown SN |
| `12_layer_ablation.png` | Exp 9: layer hooking ablation |
| `14_periodic_ses.png` | Exp 10: periodic SES (Pareto plot) |
| `15_resnet50_tinyimagenet.png` | Exp 11: ResNet-50 Tiny-ImageNet bar chart |
| `15c_resnet50_tinyimagenet_percorruption.png` | Exp 11: per-corruption Tiny-ImageNet |

### Script esperimenti

| File | Funzione | GPU | Stato |
|------|----------|-----|-------|
| `ses_phase3.py` | Phase 3 completo (Exp 10+11+12) per Kaggle T4 | T4 | Exp 10 completato |
| `ses_phase3_resume.py` | Phase 3 senza Exp 10 (già fatto) | T4 | Superseded da dual-gpu |
| `ses_resnet50_tinyimagenet.py` | Exp 11: ResNet-50 Tiny-ImageNet standalone | L40S | ✅ Completato |
| `ses_kaggle_dual_gpu.py` | Exp 12 + Exp 13 (WRN) in parallelo | 2× T4 | ✅ Completato (batch 512, FP16) |

### Risultati JSON

| File | Contenuto |
|------|-----------|
| `resnet50_tinyimagenet_results.json` | Exp 11: acc, gap, rob, per-corruption detail, epoch time, VRAM |
| `phase3_results.json` | Exp 10: periodic SES (generato da ses_phase3.py) |
| `wrn_results.json` | Exp 13: WRN-28-10 baseline vs SES (con history) |
| `aug_results.json` | Exp 12: SES + Mixup/CutMix 6 config (con history) |
| `dual_gpu_summary.json` | Exp 12+13: summary senza history |

### Checkpoint modelli

| File | Contenuto |
|------|-----------|
| `resnet50_tinyimagenet_baseline_best.pt` | ResNet-50 baseline (state_dict + config) |
| `resnet50_tinyimagenet_ses_best.pt` | ResNet-50 SES (state_dict + config) |

### Prompt e contesto

| File | Funzione |
|------|----------|
| `CONTEXT.md` | Questo file |
| `system_prompt.md` | System prompt per continuare in nuove chat |

---

## 3. Prossimi task

### Priorità alta

- [x] **Exp 12: SES + Mixup/CutMix (CIFAR-100)** — ✅ 15 feb 2026
  - 6 config su Kaggle T4, batch 512, FP16
  - **SES ortogonale**: aggiunge +0.26-0.79pp acc e +0.61-0.94pp rob su ogni combinazione
  - Best: SES+CutMix 73.86% acc, SES+Mixup 48.21% rob

- [x] **Exp 13: WRN-28-10 (CIFAR-100)** — ✅ 15 feb 2026
  - Baseline 75.24% vs SES 74.93% (−0.31pp acc, +0.54pp rob)
  - SES neutro su architetture large — conferma che benefici scalano con task difficulty

- [ ] **Aggiornare paper con Exp 12 + Exp 13**
  - Sezioni già aggiunte nel .tex — verificare figure e summary table
  - Aggiornare `arxiv_submission.zip`

### Priorità media — rafforzare risultati

- [ ] **Multi-seed Tiny-ImageNet (3 seed)**
  - Exp 11 è single-seed — un reviewer lo noterà
  - ~3h su L40S (3 run × 60 epoch)
  - Rafforza il risultato più forte del paper

- [ ] **Multi-seed Periodic SES (3 seed)**
  - Exp 10 è single-seed
  - ~6h su T4 (5 config × 3 seed)
  - Meno critico perché il trend è chiaro dal Pareto plot

- [ ] **Aggiornare system_prompt.md** dopo ogni esperimento completato

### Priorità bassa — estensioni future (post-arXiv v1)

- [ ] **ImageNet-100 o full ImageNet**
  - Richiede multi-GPU, giorni di compute
  - Necessario per venue top-tier (NeurIPS, ICML)
  - Usa randomized SVD per d > 2048

- [ ] **Vision Transformer (ViT)**
  - Hook su attention output o MLP output
  - Verifica se SES funziona su architetture non-convolutive
  - ViT-B/16 su CIFAR-100 o Tiny-ImageNet

- [ ] **Fine-tuning LLM (BERT su GLUE)**
  - Testa SES su NLP
  - Hook su layer output di BERT/RoBERTa
  - Diversificherebbe enormemente i claim del paper

- [ ] **Pacchettizzare come libreria pip**
  - `pip install ses-regularizer`
  - API: `SESRegularizer(model, beta=0.7, lambda_ses=0.01, hook_last_n=3, every_k=10)`
  - Aumenta impatto pratico e citazioni

- [ ] **Implementare SVD trick**
  - Quando B < d, usare SVD su H_c (B×d) invece di eigendecomposition su Σ (d×d)
  - Riduce costo da O(d³) a O(B²d) — ~64× speedup per d=2048, B=256
  - Matematicamente equivalente, nessun impatto sui risultati

### Bug fix critici da NON reintrodurre

1. **Hook attivi durante eval** → OOM. Sempre `collector.disable()` prima di evaluate, `enable()` dopo
2. **Modelli non liberati tra run** → `del model; torch.cuda.empty_cache()` obbligatorio
3. **In-hook pooling** → `output.mean(dim=[2,3])` nell'hook per immagini >32×32 o molti hook
4. **SES eigendecomp in float32** → `H = H.float()` prima di `eigvalsh` quando si usa BF16/FP16
5. **Spectral norm API** → `nn.utils.spectral_norm(module, name='weight')` sul modulo diretto
6. **`total_memory` non `total_mem`** → attributo corretto di `CudaDeviceProperties`
7. **Variabili locali in closure** → sempre hardcodare `depth=28, widen_factor=10` nel builder
8. **ses_regularizer hardcoded `cuda:0`** → usare `activations[0].device` per multi-GPU
9. **Threading su Kaggle non parallelizza** → GIL serializza CPU, usare subprocess con `CUDA_VISIBLE_DEVICES`
10. **T4 non supporta BFloat16** → usare `torch.float16` con `GradScaler`, non `bfloat16`

---

## Setup tecnico di riferimento

| Parametro | CIFAR (Exp 1-10) | Tiny-ImageNet (Exp 11) | Dual-GPU (Exp 12-13) |
|-----------|-------------------|------------------------|----------------------|
| Architettura | ResNet-18 (11M) | ResNet-50 (25M) | R18 (11M) + WRN-28-10 (36.5M) |
| Adattamento | conv1 3×3 s1, no maxpool | conv1 3×3 s1, no maxpool | idem |
| Immagini | 32×32 | 64×64 | 32×32 |
| Batch | 256 | 256 | 512 |
| Optimizer | SGD lr=0.1 mom=0.9 wd=5e-4 | idem | idem |
| Epoch | 50 | 60 | 50 |
| Milestones | [20, 35, 45] | [25, 40, 52] | [20, 35, 45] |
| Precisione | FP32 | BFloat16 + GradScaler | FP16 + GradScaler |
| Hook | 9 (all blocks + avgpool) | 4 (layer4 + avgpool) | 9 (R18) / 13 (WRN) |
| SES default | λ=0.01, β=0.7 | λ=0.01, β=0.7 | λ=0.01, β=0.7 |
| GPU | T4 (16GB) | L40S (48GB) | 2× T4 (16GB) |
