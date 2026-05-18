# `research_comparison/` — Multimodal-SSL-AD vs 2023–2026 SOTA

Reviewer-grade research comparison of your thesis pipeline against the most recent published
and preprint work in autonomous-driving perception.

## What is in here

| File | Purpose |
|---|---|
| [`COMPARISON.md`](COMPARISON.md) | Sourced comparison: 19 baseline methods, every metric traceable to a paper / leaderboard. Read this first. |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | Ranked Tier-A / B / C change-list, each tied to a specific 2024–2026 paper. Read this second. |
| [`data.py`](data.py) | Central data file — every paper's reported numbers in one place. Edit here, regenerate GIFs. |
| [`generate_gifs.py`](generate_gifs.py) | Renders all seven animated comparison artifacts. `python generate_gifs.py`. |
| [`gifs/`](gifs/) | The animated artifacts. Open in any browser or markdown viewer. |

## Quick re-run

```bash
# from repo root, with your project venv active
cd research_comparison
python generate_gifs.py
# -> writes 7 GIFs into gifs/
```

The script depends only on `matplotlib`, `numpy`, `Pillow` — all already in your
`requirements.txt`.

## Headline takeaways

1. **You are not behind on mAP — you are on a different axis.** Supervised LiDAR-camera SOTA
   sits at 0.72–0.73 mAP; your 0.597 with 4-modality SSL + 10%-labels-equivalent is a
   different claim. Frame the paper around label efficiency, modality breadth, and the GNN
   reasoner — not raw mAP.
2. **The single biggest reviewer-impact item is benchmarking on SeeingThroughFog**
   (Tier-A2 in `IMPROVEMENTS.md`). You have thermal + radar; SAMFusion (ECCV 2024) made
   adverse-weather a first-class metric in your sub-field; you currently report words where
   it expects numbers.
3. **Report planning numbers in the UniAD protocol** (Tier-A1). Without L2 / collision %
   / ADE / FDE, your trajectory contribution is unverifiable against GraphAD, DriveWorld,
   UAD, FSDrive, VLA-World.
4. **Replace the frozen COCO Faster R-CNN proposals** (Tier-B1). It is the obvious weakness
   in your "annotation-free SSL" story — those COCO labels are annotation. Switch to learned
   DETR-style queries; closes the story and adds mAP.

The full ranked plan with effort estimates is in `IMPROVEMENTS.md`.

## Provenance & honesty markers

Every GIF and table flags what is **measured** vs **interpolated** vs **estimated**.
Reviewers accept honest markers; they do not accept fabricated-looking curves with no
provenance. See `COMPARISON.md` §4 for the full provenance matrix.
