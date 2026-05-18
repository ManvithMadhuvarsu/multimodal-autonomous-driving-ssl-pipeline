# Improvement plan for Multimodal-SSL-AD

Ranked by **acceptance-impact-per-effort** for an IEEE / top-venue submission, with each item
tied to the specific 2024–2026 paper that motivates it. *Tier-A* items are blockers a reviewer
will fail you on. *Tier-B* items materially raise scores. *Tier-C* items are nice-to-have polish.

---

## TIER A — fix these before submission

### A1. Report planning metrics in the UniAD protocol
**Gap.** Your README has no L2-displacement, no collision %, no min ADE/FDE in the
[UniAD](https://arxiv.org/abs/2212.10156) format. Every 2024–2026 planning paper does
(GraphAD, UAD, DriveWorld, GenAD, FSDrive, VLA-World).
**Why a reviewer cares.** Without this, your trajectory head is unverifiable against
SOTA. A "−34.2 % MSE" relative number on its own is a yellow flag.
**Effort.** ~1 week. No retraining. Just run nuScenes val under the UniAD eval split,
fill the 3-row table (1 s / 2 s / 3 s) and the planning row (avg L2 + collision %).
**Target.** Match or beat **GraphAD (L2 = 0.68 m, collision = 0.12 %)** — your nearest
architectural cousin (it also uses scene-graph reasoning).

### A2. Benchmark on adverse-weather (SeeingThroughFog or DENSE)
**Gap.** You list "extreme conditions" as a limitation but report **no** rain/fog/snow/night
numbers. You have thermal + radar — exactly the sensors that should win this benchmark.
**Why a reviewer cares.** [SAMFusion](https://arxiv.org/abs/2508.16408) (ECCV 2024) made
adverse-weather robustness a first-class metric in this sub-field; it is now expected.
**Effort.** 1–2 weeks. SeeingThroughFog is publicly available; the protocol is KITTI-style
AP at near/mid/far range × weather condition.
**Target.** Match SAMFusion on at least one of fog/snow pedestrian far-range AP. Even being
within 10 AP of SAMFusion is a publishable result given your fundamentally different
(SSL-pretrained, GNN-refined) approach.

### A3. Re-state per-class detection APs with NDS
**Gap.** README reports mAP 0.597 only; no NDS, no per-class AP, no mATE/mASE/mAOE breakdown.
nuScenes papers since 2022 always report NDS + 10-class AP table.
**Why a reviewer cares.** mAP alone hides whether your method is good at the rare classes
(motorcycle, construction vehicle, traffic cone) where SSL pretraining should help most.
**Effort.** A few hours. Just compute and tabulate from your existing predictions.
**Target.** Show ≥ 1 of the rare classes outperforms BEVFusion baseline — that becomes a
story sentence for the abstract.

### A4. Add multi-seed standard deviations to every headline metric
**Gap.** README §Results uses ± std over 3 seeds for the SSL linear probe (good!) but the
fine-tuned mAP / mIoU / FPS numbers are single-seed.
**Why a reviewer cares.** Single-seed top-line numbers are a top-3 reject reason at IEEE
Transactions venues right now (paper-with-code policy + post-replication-crisis norms).
**Effort.** ~1 week of repeat training. If GPU-bound, at minimum re-run only the final
fine-tune stage (cheaper than full SSL pretraining).

---

## TIER B — material gains, plan now for next revision

### B1. Replace frozen COCO Faster R-CNN proposals with learned query-based detection
**Gap.** Your scene graph nodes come from a frozen COCO-pretrained Faster R-CNN. This is
the second item a reviewer will circle: it weakens your "SSL without annotation" claim
(those COCO labels are annotation, even if not yours) and caps detection recall to
COCO-class IoU.
**Why a reviewer cares.** [DETR](https://arxiv.org/abs/2005.12872) →
[Deformable DETR](https://arxiv.org/abs/2010.04159) → [DINO-DETR](https://arxiv.org/abs/2203.03605)
made learned object queries the default; CenterPoint heads do the same for LiDAR.
**Effort.** 2–3 weeks. Swap Faster R-CNN with either:
  - **DETR-style learned queries** on your fusion-transformer output (cleaner story — single
    transformer does fusion + detection), OR
  - **CenterPoint head** on the BEV-pooled fused features (better mAP, less story coherence).
**Expected gain.** +0.05 to +0.10 mAP from a stronger detector head alone.

### B2. Heterogeneous-edge GNN to replace GraphSAGE
**Gap.** [GraphSAGE](https://arxiv.org/abs/1706.02216) treats all edges identically. Your
README defines three edge types (BEV proximity, Doppler similarity, learned relation) but
your GNN can't tell them apart at message-passing time → you lose typed semantics.
**Why a reviewer cares.** [GraphAD (Zhang et al. 2024)](https://arxiv.org/abs/2403.19098)
explicitly uses heterogeneous edge types and reports +14 % planning collision improvement
specifically attributed to typed edges.
**Effort.** 1 week. Replace GraphSAGE with `HeteroConv` (PyG) or a 1–2-layer
[HGT](https://arxiv.org/abs/2003.01332) / [HetEdgeGAT](https://arxiv.org/abs/1903.07293).
**Expected gain.** +1–3 mAP on detection downstream; bigger gain on trajectory ADE.
**Also fixes.** Your README's listed limitation of GNN over-smoothing — heterogeneous
attention with edge typing is the standard published cure.

### B3. Temporal modeling — add a 4D-pretraining pretext task
**Gap.** Your SSL is currently per-frame. nuScenes is sequential; you are leaving free signal
on the table.
**Why a reviewer cares.** [DriveWorld (CVPR 2024)](https://arxiv.org/abs/2405.04390) showed
that 4D world-model pretraining (predicting future 3D occupancy from past frames) yields
+4.8 to +7.5 mAP on the same detection task, with no architectural change. GASP (2025) and
FSDrive (NeurIPS 2025) double down on this.
**Effort.** 3–4 weeks. Add a 4D-occupancy or BEV-future-prediction head to your SSL stage.
Re-use existing fusion encoder weights.
**Expected gain.** +0.04 to +0.07 mAP based on DriveWorld's reported deltas;
+0.05 m minADE reduction.

### B4. Sensor-adaptive fusion (depth-weighted) instead of uniform attention
**Gap.** Your transformer fusion uses uniform-attention modality dropout. SAMFusion's edge is
**depth-weighted** attention: which modality dominates depends on object distance (RGB near,
radar/LiDAR far, thermal in low light).
**Why a reviewer cares.** Closes the obvious gap to SAMFusion without changing your sensor stack.
**Effort.** 1–2 weeks. Add a distance bucket → modality weight head before the
fusion-transformer cross-attention.
**Expected gain.** +5 to +10 AP on far-range pedestrians in adverse weather; almost no
clear-day regression.

### B5. Cut inference latency to ≥ 30 FPS (production threshold)
**Gap.** 18.4 FPS is below the 30 Hz sensor cadence of nuScenes. Limits deployment claims.
**Why a reviewer cares.** Reviewer-2 will ask "can this run on Drive Orin?" and 18 FPS is
the answer that loses points.
**Effort.** 1–2 weeks for the easy wins:
  - INT8 quantization of the 2× ResNet-50 encoders (~30 % latency drop, ≤ 1 mAP cost).
  - Encoder pruning of the Radar lightweight-CNN — already only 0.6 GFLOPs, low return.
  - Cache the frozen Faster R-CNN proposals across frames (or replace it per B1).
**Expected gain.** 18.4 → ~28–32 FPS; meets production claim.

---

## TIER C — polish that strengthens the paper

### C1. Add a 2026 baseline row: VLA-World
**Why.** [VLA-World (Qwen2-VL-2B)](https://arxiv.org/abs/2604.09059) is the headline 2026
planning result (L2 0.26 m, collision 0.08 %). Even mentioning it in Related Work and
explaining why you don't compare directly (different paradigm: 2B-param VLA vs 119 M-param
perception stack) strengthens your literature framing.

### C2. Cross-modal retrieval ablation on a held-out set
Your README's cross-modal R@1 numbers are excellent (multimodal → all = 69.5 %). Move from
table-only to a small qualitative figure of retrieved frames — easy paper polish.

### C3. Replace the unsupervised cluster t-SNE figure
The README mentions t-SNE/UMAP shows compact clusters. If you have the embeddings, generate
the actual t-SNE per modality and per fusion-output → 1 page of supplementary, high
information density.

### C4. Long-tail per-class breakdown for fog/snow + day/night
If A2 is done, slice the per-class APs by condition. Even one cell where you beat SAMFusion
(motorcycle in snow, pedestrian at very-far range, etc.) is a paper claim.

### C5. Open-source the four-modality alignment code
Your "phantom-modality" workaround (cross-dataset coupling: nuScenes for RGB/L/R, FLIR for
RGB/Thermal) is genuinely novel and reproducibility-relevant. Releasing this as a
standalone utility increases citation odds and is a soft demand of IEEE Open Journal.

---

## Cost-ordered roadmap

| Effort | Item | Reviewer-impact |
|---|---|---|
| 4 hrs | A3 NDS + per-class AP | medium |
| 1 day | C1 VLA-World related-work | low |
| 3 days | C3 t-SNE figures | low |
| 1 week | A1 Planning metrics (UniAD protocol) | **high** |
| 1 week | A4 Multi-seed std | **high** |
| 1 week | B2 Heterogeneous-edge GNN | medium |
| 1–2 weeks | A2 SeeingThroughFog benchmark | **very high** |
| 1–2 weeks | B4 Sensor-adaptive depth-weighted fusion | high |
| 1–2 weeks | B5 INT8 quantization → ≥ 30 FPS | high |
| 2–3 weeks | B1 Learned-query detection head | high |
| 3–4 weeks | B3 4D world-model pretraining | high |

**Recommended attack order for a 6-week revision window:**

1. **Week 1:** A3 + A1 + A4 (verifiable baseline numbers — opens up every other comparison)
2. **Week 2–3:** A2 (adverse-weather benchmark — biggest reviewer-impact single item)
3. **Week 4:** B2 + B5 (architectural cleanup + production-FPS, both 1-week)
4. **Week 5–6:** B4 + B1 (the two items most likely to move mAP above 0.65)
5. Reserve B3 (4D pretraining) for a follow-up revision if Week-1-to-6 work is enough.

---

## Bottom line

Your pipeline's contribution is **breadth (4 modalities) + label efficiency (SSL) +
relational reasoning (GNN)** — *not* topping the supervised mAP leaderboard. The paper writes
itself if you (a) close the planning-metric gap (A1), (b) put numbers in the weather table
where you currently have words (A2), and (c) replace the frozen COCO proposals with learned
queries (B1) so the "annotation-free" claim is airtight.
