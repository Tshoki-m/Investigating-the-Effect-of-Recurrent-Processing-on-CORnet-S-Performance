#!/usr/bin/env python
"""One command to reproduce every number and figure.

    python reproduce.py                  # full: download data, train 4 configs, analyze (GPU, hours)
    python reproduce.py --skip-training  # regenerate tables/figures from the committed run histories (seconds)

Outputs: results/results_table.csv, results/summary.txt, results/recurrence_ablation.png,
results/training_curves.png. Training resumes from checkpoints and skips finished runs.
"""
import argparse
import os
import subprocess
import sys

CONFIGS = [(2, 4, 2), (1, 1, 1), (2, 2, 2), (4, 4, 4)]   # baseline first
# Reduced protocol used for the report: Tiny-ImageNet, 64x64, paper's optimizer/schedule.
PROTOCOL = ['--img-size', '64', '--min-scale', '0.35', '--amp', '--ckpt-every', '2']


def run(cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def check_official(py):
    """Numerical equivalence of src/cornet_s.py vs the official model (needs internet + the cornet package)."""
    os.makedirs('results', exist_ok=True)
    with open('results/equivalence_check.txt', 'w') as f:
        r = subprocess.run([py, 'src/check_against_official.py'], stdout=f, stderr=subprocess.STDOUT)
    print('equivalence check exit code', r.returncode, '-> results/equivalence_check.txt')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-training', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--data-root', default='data')
    p.add_argument('--variant', choices=['base', 'nopool'], default='base',
                   help='base: the reported protocol; nopool: V1 max-pool removed (final map 4x4 instead of 2x2)')
    p.add_argument('--runs', default=None, help='default results/runs (base) or results/runs_nopool')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    py = sys.executable
    nopool = a.variant == 'nopool'
    runs = a.runs or ('results/runs_nopool' if nopool else 'results/runs')
    out = 'results_nopool' if nopool else 'results'
    extra = ['--no-v1-pool'] if nopool else []

    if not a.skip_training:
        check_official(py)
        run([py, 'src/prep_tiny_imagenet.py', a.data_root])
        data = os.path.join(a.data_root, 'tiny-imagenet-200')
        for seed in a.seeds:
            for cfg in CONFIGS:
                name = f"cornet_s_{cfg[0]}-{cfg[1]}-{cfg[2]}_seed{seed}"
                if os.path.exists(os.path.join(runs, name + '.json')):
                    print('skip finished run', name); continue
                run([py, 'src/train.py', '--data', data, '--times', *map(str, cfg), '--seed', str(seed),
                     '--workers', str(a.workers), '--out', runs, *PROTOCOL, *extra])
    run([py, 'src/analyze.py', '--runs', runs, '--out', out])


if __name__ == '__main__':
    main()
