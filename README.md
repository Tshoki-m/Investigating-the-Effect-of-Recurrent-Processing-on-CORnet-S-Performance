# Investigating the effect of recurrent processing on CORnet-S performance

Reproduction of the recurrence ablation in Kubilius et al. (2019), *Brain-Like Object Recognition with
High-Performing Shallow Recurrent ANNs* (NeurIPS 2019), **Fig. 4, row 1 / Sec. 4.3**: ImageNet top-1 change
as the number of recurrent steps in V2/V4/IT is varied (baseline 2/4/2; paper: 1/1/1 = -7, 2/2/2 = -1,
4/4/4 = 0 points).

**What this repo does:** re-implements CORnet-S (`src/cornet_s.py`, names match the official code), trains
the variants 1/1/1, 2/2/2, 2/4/2, 4/4/4 from scratch under one shared recipe, and compares their top-1 with
the paper. **Reduced protocol:** Tiny-ImageNet (200 classes, 64x64), not full ImageNet. The paper's trend
did **not** reproduce in this setting; see the report in `report/`.

## Setup
    conda env create -f environment.yml
    conda activate cornet-repro

`environment.yml` must pin the versions actually used for the reported runs: generate it in the Colab session that
produced the results with `python src/print_env.py > environment.yml` and commit it.

## One command to reproduce every number and figure
    python reproduce.py

This downloads Tiny-ImageNet, trains 4 configurations (43 epochs each, SGD momentum 0.9, batch 256,
lr 0.1 divided by 10 every 20 epochs, mixed precision) and writes `results/results_table.csv`,
`results/summary.txt`, `results/recurrence_ablation.png` and `results/training_curves.png`.
It needs a CUDA GPU and takes hours; interrupted runs resume from checkpoints and finished runs are skipped.

**Fast path from a clean checkout:** the finished run histories are committed in `results/runs/`, so

    python reproduce.py --skip-training

regenerates every table, number and figure in the report in seconds, with no GPU or data download. The full command
additionally runs the numerical equivalence check against the official model (`results/equivalence_check.txt`, needs
internet); its committed output is the number quoted in the report.
Extra seeds: `python reproduce.py --seeds 0 1 2`. Larger-final-map variant (V1 max-pool removed): `python reproduce.py --variant nopool`
(histories in `results/runs_nopool/`, outputs in `results_nopool/`).

## Provenance of the committed results
The histories in `results/runs/` come from running the same `src/train.py` commands that `reproduce.py`
issues, on a Google Colab GPU (one seed per configuration, seed 0). Reported accuracy is the last-epoch
validation top-1 (no best-epoch selection).

## Optional: check the re-implementation against the official model
    pip install git+https://github.com/dicarlolab/CORnet
    python src/check_against_official.py
Loads the official pretrained weights into both models (strict key match) and compares outputs.

## Layout
    reproduce.py            single entry point
    src/cornet_s.py         CORnet-S with configurable steps per area
    src/train.py            training / evaluation of one variant
    src/prep_tiny_imagenet.py   dataset download + folder layout
    src/analyze.py          tables and figures from results/runs/*.json
    results/                run histories (runs/) and generated tables and figures
    report/                 IEEE-format write-up (LaTeX)

## Assumptions not stated in the paper
Weight decay 1e-4 and RandomResizedCrop augmentation (minimum scale 0.35 at 64 px) follow my recollection of
the official training script and a low-resolution adaptation; they are not from the paper text.
