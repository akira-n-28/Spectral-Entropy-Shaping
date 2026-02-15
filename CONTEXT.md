# CONTEXT.md — Spectral Entropy Shaping (SES) Project

**Autore**: Davide Le Bone
**Ultimo aggiornamento**: 15 febbraio 2026
**Paper**: 28+ pagine, 18 esperimenti, 35 references

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
| P3 | 14 | Vision Transformers (CIFAR-100) | ✅ | **ViT-S: +1.73pp acc, +1.33pp rob; ViT-T: +0.72pp acc, +1.31pp rob** |
| P3 | 15 | SES + ViT + Data Augmentation (CIFAR-100) | ✅ | **SES additivo con RA (+0.91 acc, +0.96 rob), forte con Mixup (+2.06 acc, +2.52 rob)** |
| P3 | 16 | Adaptive β Scheduling (CIFAR-100) | ✅ | **Warmup hurt (−0.73 to −1.22pp), Reverse ≈ Fixed, β=0.7 near-optimal** |
| P3 | 17 | ViT Ablations (β sweep + layer hooking) | ✅ | **β=0.5 optimal for ViT (+2.66pp), Last-4 > All-12 (+2.33 vs +1.73pp)** |
| P3 | 18 | SES on Attention Matrices (ViT-S) | ✅ | **Triple SES (block+attn_out+attn_wt) +2.65pp, block+attn complementari** |

### Trend chiave: benefici scalano con difficoltà del task

| Dataset | Classi | Architettura | Δ Acc | Δ Rob |
|---------|--------|--------------|-------|-------|
| CIFAR-10 | 10 | ResNet-18 | −0.06pp | +0.45pp |
| CIFAR-100 | 100 | ResNet-18 | +0.47pp | +0.65pp |
| CIFAR-100 | 100 | WRN-28-10 (36.5M) | −0.31pp | +0.54pp |
| CIFAR-100 | 100 | ViT-Tiny/4 (5.5M) | +0.72pp | +1.31pp |
| CIFAR-100 | 100 | ViT-Small/4 (22M) | **+1.73pp** | +1.33pp |
| Tiny-ImageNet | 200 | ResNet-50 | **+2.58pp** | **+2.37pp** |

### SES è ortogonale alle augmentation moderne (Exp 12)

| Combinazione | Δ Acc | Δ Rob | Interpretazione |
|---|---|---|---|
| None → SES | +0.34pp | +0.87pp | SES standalone |
| Mixup → SES+Mixup | +0.26pp | +0.94pp | SES aggiunge su Mixup |
| CutMix → SES+CutMix | **+0.79pp** | +0.61pp | SES aggiunge su CutMix |

Best overall: **SES+CutMix** 73.86% acc, **SES+Mixup** 48.21% rob

### SES è ortogonale alle augmentation su ViT (Exp 15)

| Confronto (ViT-Small/4) | Δ Acc | Δ Rob | Interpretazione |
|---|---|---|---|
| RA → RA+SES | +0.91pp | +0.96pp | SES additivo con RandAugment |
| RA+Mixup → RA+SES+Mixup | **+2.06pp** | **+2.52pp** | SES fortemente additivo con Mixup |
| RA+CutMix → RA+SES+CutMix | −0.65pp | +0.25pp | SES neutro con CutMix |

Nota: Mixup/CutMix danneggiano ViT accuracy (−6.6/−5.6pp) quando combinati con RandAugment — underfitting severo.
SES aiuta a contrastare l'underfitting da Mixup.

### Adaptive β scheduling — Fixed β=0.7 è quasi ottimale (Exp 16)

| Schedule | Acc | Gap | Rob | Δ Acc vs Fixed | Δ Rob vs Fixed |
|---|---|---|---|---|---|
| Baseline (no SES) | 73.00 | 26.99 | 45.21 | — | — |
| Fixed β=0.7 | 73.34 | 26.67 | **46.08** | — | — |
| Linear 0.3→0.9 | 72.61 | 27.17 | 45.27 | −0.73pp | −0.81pp |
| Cosine 0.3→0.9 | 72.30 | 27.62 | 45.05 | −1.04pp | −1.03pp |
| **Reverse 0.9→0.3** | **73.43** | **26.21** | 46.14 | +0.09pp | +0.06pp |
| Step 0.3→0.7→0.9 | 72.12 | 27.43 | 45.76 | −1.22pp | −0.32pp |

**Risultato chiave**: Warmup β (basso→alto) **danneggia** performance. Reverse (alto→basso) ≈ Fixed.
Fixed β=0.7 è quasi ottimale — conferma la semplicità del design a singolo iperparametro.

### ViT β sweep — β=0.5 ottimale per transformer (Exp 17a)

| Config | Acc | Gap | Rob | Δ Acc vs Base | Δ Rob vs Base |
|--------|-----|-----|-----|---------------|---------------|
| Baseline (no SES) | 55.28 | 44.50 | 34.75 | — | — |
| β=0.3 (all 12) | 55.28 | 44.61 | 34.42 | +0.00 | −0.32 |
| **β=0.5 (all 12)** | **57.94** | **42.13** | **36.27** | **+2.66** | **+1.52** |
| β=0.7 (all 12, Exp 14) | 57.01 | 43.10 | 36.08 | +1.73 | +1.33 |
| β=0.9 (all 12) | 56.66 | 43.35 | 35.96 | +1.38 | +1.22 |

**Inverted-U pattern**: β=0.3 troppo basso (nessun effetto), β=0.5 ottimale, β=0.7-0.9 decrescenti.
Optimal β più basso per ViT (0.5) che per CNN (0.7) — i transformer operano a entropia spettrale naturalmente più alta.

### ViT layer hooking ablation — Last-4 > All-12 (Exp 17b)

| Blocks hookati | Acc | Gap | Rob | Δ Acc | Δ Rob | Time/ep |
|----------------|-----|-----|-----|-------|-------|---------|
| Baseline (no SES) | 55.28 | 44.50 | 34.75 | — | — | 57.5s |
| First-4 (0–3) | 57.04 | 42.89 | 35.90 | +1.76 | +1.16 | 76.6s |
| All-12 (Exp 14) | 57.01 | 43.10 | 36.08 | +1.73 | +1.33 | 79.6s |
| **Last-4 (8–11)** | **57.61** | **42.19** | **36.12** | **+2.33** | **+1.38** | **65.3s** |

**Last-4 batte All-12**: +0.60pp acc, −18% overhead. I block finali (task-specific) beneficiano di più dalla regolarizzazione spettrale.
Raccomandazione pratica ViT: **β=0.5, last-4 blocks**.

### SES on Attention Matrices — block + attention complementari (Exp 18)

| Hook target | Blocks | Acc | Gap | Rob | Δ Acc | Δ Rob | Time/ep |
|-------------|--------|-----|-----|-----|-------|-------|---------|
| Baseline (Exp 17) | — | 55.28 | 44.50 | 34.75 | — | — | 57.5s |
| Block output | Last-4 | 57.31 | 42.49 | 36.00 | +2.03 | +1.25 | 75.2s |
| Attn sub-layer | Last-4 | 57.02 | 42.78 | 35.53 | +1.74 | +0.78 | 76.4s |
| Attn weights | Last-4 | 56.01 | 44.10 | 34.43 | +0.73 | −0.32 | 76.4s |
| Attn weights | All-12 | 56.88 | 42.98 | 35.66 | +1.60 | +0.91 | 81.2s |
| Dual (Block+AttnWt) | Last-4 | 57.66 | 42.26 | 36.12 | +2.38 | +1.37 | 73.9s |
| **Triple (Block+AttnOut+AttnWt)** | Last-4 | **57.93** | **42.06** | **36.12** | **+2.65** | **+1.37** | 80.9s |

**Findings**:
- Block output è il miglior target singolo (+2.03pp)
- Attn weights da solo è debole (+0.73pp L4) — attention weights vivono sul simplesso
- **Block + Attn sono complementari**: Dual +2.38pp, Triple +2.65pp
- Triple (57.93%) ≈ β=0.5 All-12 (57.94% Exp 17), ma con solo Last-4 hooks + info attention
- Nota: block_l4 β=0.5 (57.31%) < separati Exp 17 (β=0.5 All-12: 57.94%, L4 β=0.7: 57.61%)

### Predizioni teoriche

| ID | Predizione | Status |
|----|-----------|--------|
| P1 | Reduced generalization gap | ✅ (scala con difficoltà: 0%→-1.3%→-6.1%) |
| P2 | Improved robustness | ✅ (consistente su tutti i dataset, CNN e ViT) |
| P3 | Controllable effective rank | ✅ |
| P4 | Reduced Jacobian κ | ✅ (2.45× riduzione) |
| P5 | No collapse | ✅ |
| P6 | β controls dynamics | ∼ (parziale — Exp 16: warmup hurt, reverse ≈ fixed; Exp 17: inverted-U con β=0.5 ottimale per ViT, β=0.7 per CNN) |

---

## 2. File importanti e funzione

### Paper

| File | Funzione |
|------|----------|
| `spectral_entropy_shaping.tex` | Sorgente LaTeX del paper (24+ pagine, 16 esperimenti) |
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
| `16_wideresnet.png` | Exp 13: WRN-28-10 bar chart |
| `16b_wideresnet_curves.png` | Exp 13: WRN-28-10 training curves |
| `16c_wideresnet_percorruption.png` | Exp 13: per-corruption WRN |
| `17_augmentation.png` | Exp 12: SES + Mixup/CutMix bar chart |
| `17c_augmentation_percorruption.png` | Exp 12: per-corruption augmentation |
| `18_vit_comparison.png` | Exp 14: ViT-Tiny vs ViT-Small comparison |
| `18a_vit_small_bars.png` | Exp 14: ViT-Small bar chart |
| `18b_vit_tiny_bars.png` | Exp 14: ViT-Tiny bar chart |
| `19_vit_augmentation.png` | Exp 15: ViT+Aug bar chart |
| `19b_vit_augmentation_curves.png` | Exp 15: ViT+Aug training curves |
| `19c_vit_augmentation_percorruption.png` | Exp 15: per-corruption ViT+Aug |
| `20_adaptive_beta.png` | Exp 16: adaptive β bar chart |
| `20b_adaptive_beta_curves.png` | Exp 16: β schedules + accuracy curves |
| `20c_adaptive_beta_percorruption.png` | Exp 16: per-corruption adaptive β |
| `20d_adaptive_beta_delta.png` | Exp 16: Δ vs Fixed β=0.7 |
| `21_vit_ablations.png` | Exp 17: all 6 ablation configs bar chart |
| `21a_vit_beta_sweep.png` | Exp 17a: β sweep bar chart |
| `21b_vit_layer_ablation.png` | Exp 17b: layer hooking ablation bar chart |
| `21c_vit_ablations_curves.png` | Exp 17: accuracy curves |
| `21d_vit_ablations_percorruption.png` | Exp 17: per-corruption breakdown |
| `22_vit_attention_ses.png` | Exp 18: all 6 attention configs bar chart |
| `22b_vit_attention_curves.png` | Exp 18: accuracy curves |
| `22c_vit_attention_delta.png` | Exp 18: Δ vs baseline |
| `22d_vit_attention_percorruption.png` | Exp 18: per-corruption breakdown |

### Script esperimenti

| File | Funzione | GPU | Stato |
|------|----------|-----|-------|
| `ses_phase3.py` | Phase 3 completo (Exp 10+11+12) per Kaggle T4 | T4 | Exp 10 completato |
| `ses_phase3_resume.py` | Phase 3 senza Exp 10 (già fatto) | T4 | Superseded da dual-gpu |
| `ses_resnet50_tinyimagenet.py` | Exp 11: ResNet-50 Tiny-ImageNet standalone | L40S | ✅ Completato |
| `ses_kaggle_dual_gpu.py` | Exp 12 + Exp 13 (WRN) in parallelo | 2× T4 | ✅ Completato (batch 512, FP16) |
| `ses_vit_experiment.py` | Exp 14: ViT-Small + ViT-Tiny in parallelo | 2× T4 | ✅ Completato (batch 512, FP16, AdamW) |
| `ses_vit_augmentation.py` | Exp 15: ViT-S + RA/Mixup/CutMix in parallelo | 2× T4 | ✅ Completato (batch 512, FP16, AdamW, RA) |
| `ses_adaptive_beta.py` | Exp 16: Adaptive β scheduling (6 config) | 2× T4 | ✅ Completato (batch 512, FP16) |
| `ses_vit_ablations.py` | Exp 17: ViT β sweep + layer hooking ablation (6 config) | 2× T4 | ✅ Completato (batch 512, FP16, AdamW) |
| `ses_vit_attention.py` | Exp 18: SES on attention matrices (6 config) | 2× T4 | ✅ Completato (batch 512, FP16, AdamW, β=0.5) |

### Risultati JSON

| File | Contenuto |
|------|-----------|
| `resnet50_tinyimagenet_results.json` | Exp 11: acc, gap, rob, per-corruption detail, epoch time, VRAM |
| `phase3_results.json` | Exp 10: periodic SES (generato da ses_phase3.py) |
| `wrn_results.json` | Exp 13: WRN-28-10 baseline vs SES (con history) |
| `aug_results.json` | Exp 12: SES + Mixup/CutMix 6 config (con history) |
| `dual_gpu_summary.json` | Exp 12+13: summary senza history |
| `vit_small_results.json` | Exp 14: ViT-Small/4 baseline vs SES (con history) |
| `vit_tiny_results.json` | Exp 14: ViT-Tiny/4 baseline vs SES (con history) |
| `vit_summary.json` | Exp 14: summary senza history |
| `vit_aug_results.json` | Exp 15: ViT+Aug 6 config (con history) |
| `vit_aug_summary.json` | Exp 15: summary senza history |
| `gpu0_results.json` | Exp 15: GPU0 results (RA, RA+SES, RA+Mixup) |
| `gpu1_results.json` | Exp 15: GPU1 results (RA+SES+Mixup, RA+CutMix, RA+SES+CutMix) |
| `adaptive_beta_results.json` | Exp 16: 6 β schedule configs (con history) |
| `adaptive_beta_summary.json` | Exp 16: summary senza history |
| `gpu0_beta_results.json` | Exp 16: GPU0 (baseline, fixed, linear) |
| `gpu1_beta_results.json` | Exp 16: GPU1 (cosine, reverse, step) |
| `vit_ablations_results.json` | Exp 17: 6 config ViT ablations (con history) |
| `vit_ablations_summary.json` | Exp 17: summary senza history |
| `gpu0_vit_ablations.json` | Exp 17: GPU0 (baseline, β=0.5, last-4) |
| `gpu1_vit_ablations.json` | Exp 17: GPU1 (β=0.3, β=0.9, first-4) |
| `vit_attention_results.json` | Exp 18: 6 config attention SES (con history) |
| `vit_attention_summary.json` | Exp 18: summary senza history |
| `gpu0_attn_results.json` | Exp 18: GPU0 (block_l4, attn_out_l4, attn_wt_l4) |
| `gpu1_attn_results.json` | Exp 18: GPU1 (dual_l4, attn_wt_all, triple_l4) |

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

- [x] **Aggiornare paper con Exp 12–18** — ✅ 15 feb 2026
  - Sezioni Exp 12, 13, 14, 15, 16, 17, 18 aggiunte nel .tex
  - Abstract, intro, summary, conclusion, future work aggiornati
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

- [x] **Vision Transformer (ViT)** — ✅ 15 feb 2026
  - Hook su TransformerBlock output (mean-pool su seq dim)
  - ViT-Small/4: +1.73pp acc, -3.1% gap, +1.33pp rob
  - ViT-Tiny/4: +0.72pp acc, -1.5% gap, +1.31pp rob
  - SES funziona su architetture non-convolutive!

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
11. **ViT attivazioni 3D** → `output.mean(dim=1)` nell'hook per pooling su seq_len: [B,N,D]→[B,D]
12. **ViT gradient clipping** → `clip_grad_norm_(max_norm=1.0)` necessario con AdamW su ViT

---

## Setup tecnico di riferimento

| Parametro | CIFAR (Exp 1-10) | Tiny-ImageNet (Exp 11) | Dual-GPU (Exp 12-13) | ViT (Exp 14) | ViT+Aug (Exp 15) | β Sched (Exp 16) | ViT Abl (Exp 17) |
|-----------|-------------------|------------------------|----------------------|--------------|------------------|------------------|------------------|
| Architettura | ResNet-18 (11M) | ResNet-50 (25M) | R18 (11M) + WRN-28-10 (36.5M) | ViT-S (22M) + ViT-T (5.5M) | ViT-Small/4 (22M) | ResNet-18 (11M) | ViT-Small/4 (22M) |
| Adattamento | conv1 3×3 s1, no maxpool | conv1 3×3 s1, no maxpool | idem | patch_size=4, 64 patches | patch_size=4, 64 patches | conv1 3×3 s1, no maxpool | patch_size=4, 64 patches |
| Immagini | 32×32 | 64×64 | 32×32 | 32×32 | 32×32 | 32×32 | 32×32 |
| Batch | 256 | 256 | 512 | 512 | 512 | 512 | 512 |
| Optimizer | SGD lr=0.1 mom=0.9 wd=5e-4 | idem | idem | AdamW lr=1e-3 wd=0.05 | AdamW lr=1e-3 wd=0.05 | SGD lr=0.1 mom=0.9 wd=5e-4 | AdamW lr=1e-3 wd=0.05 |
| Epoch | 50 | 60 | 50 | 50 | 50 | 50 | 50 |
| Schedule | MultiStep [20,35,45] | MultiStep [25,40,52] | MultiStep [20,35,45] | Cosine + warmup 5ep | Cosine + warmup 5ep | MultiStep [20,35,45] | Cosine + warmup 5ep |
| Precisione | FP32 | BFloat16 + GradScaler | FP16 + GradScaler | FP16 + GradScaler | FP16 + GradScaler | FP16 + GradScaler | FP16 + GradScaler |
| Hook | 9 (all blocks + avgpool) | 4 (layer4 + avgpool) | 9 (R18) / 13 (WRN) | 12 (all transformer blocks) | 12 (all transformer blocks) | 9 (all blocks + avgpool) | all/first-4/last-4 |
| SES default | λ=0.01, β=0.7 | λ=0.01, β=0.7 | λ=0.01, β=0.7 | λ=0.01, β=0.7 | λ=0.01, β=0.7 | λ=0.01, β adaptive | λ=0.01, β={0.3,0.5,0.7,0.9} |
| Augmentation | Standard | Standard | Standard / Mixup / CutMix | Standard | RA(n=2,m=9) / Mixup / CutMix | Standard | Standard |
| GPU | T4 (16GB) | L40S (48GB) | 2× T4 (16GB) | 2× T4 (16GB) | 2× T4 (16GB) | 2× T4 (16GB) | 2× T4 (16GB) |
