"""Aggregate results/runs/*.json into tables and figures (no GPU or data needed).
Outputs (in --out): results_table.csv, summary.txt, recurrence_ablation.png, training_curves.png
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Change in ImageNet top-1 (% points) vs the 2/4/2 baseline, Fig. 4 row 1 of Kubilius et al. (2019),
# read from the figure text.
PAPER = {'1-1-1': -7, '2-2-2': -1, '2-4-2': 0, '4-4-4': 0}
ORDER = ['1-1-1', '2-2-2', '2-4-2', '4-4-4']
N_VAL = 10000  # Tiny-ImageNet validation images


def main(runs_dir, out):
    os.makedirs(out, exist_ok=True)
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(runs_dir, 'cornet_s_*.json'))):
        r = json.load(open(f))
        groups['-'.join(map(str, r['times']))].append(r)
    if '2-4-2' not in groups:
        raise SystemExit(f'no 2-4-2 baseline runs found in {runs_dir}')
    base = np.mean([r['final_top1'] for r in groups['2-4-2']])

    rows = []
    for k in ORDER:
        if k not in groups:
            continue
        rs = groups[k]
        fin = np.array([r['final_top1'] for r in rs])
        best = np.array([max(e['val_top1'] for e in r['history']) for r in rs])
        p = fin.mean() / 100
        rows.append(dict(config=k, total_steps=sum(map(int, k.split('-'))), n_seeds=len(rs),
                         top1_mean=fin.mean(), top1_sd=fin.std(ddof=1) if len(fin) > 1 else float('nan'),
                         best_epoch_top1_mean=best.mean(), delta_vs_2_4_2=fin.mean() - base,
                         paper_delta=PAPER[k], params_M=rs[0]['params'] / 1e6,
                         val_sampling_se=100 * math.sqrt(p * (1 - p) / N_VAL),
                         final_train_loss=np.mean([r['history'][-1]['train_loss'] for r in rs])))

    with open(os.path.join(out, 'results_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lines = [f"{'config':8s} {'steps':>5s} {'n':>2s} {'top1':>6s} {'sd':>5s} {'delta':>6s} {'paper':>6s} "
             f"{'paramsM':>8s} {'valSE':>6s} {'train loss':>10s} {'best-ep top1':>12s}"]
    for r in rows:
        lines.append(f"{r['config']:8s} {r['total_steps']:5d} {r['n_seeds']:2d} {r['top1_mean']:6.2f} "
                     f"{r['top1_sd']:5.2f} {r['delta_vs_2_4_2']:6.2f} {r['paper_delta']:6.1f} "
                     f"{r['params_M']:8.2f} {r['val_sampling_se']:6.2f} {r['final_train_loss']:10.3f} "
                     f"{r['best_epoch_top1_mean']:12.2f}")
    open(os.path.join(out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.errorbar(x, [r['delta_vs_2_4_2'] for r in rows],
                yerr=[0 if math.isnan(r['top1_sd']) else r['top1_sd'] for r in rows],
                fmt='o-', capsize=3, label='this reproduction')
    ax.plot(x, [r['paper_delta'] for r in rows], 's--', label='paper (Fig. 4)')
    ax.set_xticks(x); ax.set_xticklabels([r['config'].replace('-', '/') for r in rows], fontsize=7)
    ax.set_xlabel('recurrent steps in V2/V4/IT', fontsize=8)
    ax.set_ylabel(r'$\Delta$ top-1 vs 2/4/2 (% pts)', fontsize=8)
    ax.axhline(0, color='gray', lw=0.5); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, 'recurrence_ablation.png'), dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k in ORDER:
        for r in groups.get(k, []):
            h = r['history']; lab = k.replace('-', '/') + (f" s{r['seed']}" if len(groups[k]) > 1 else '')
            ax[0].plot([e['train_loss'] for e in h], label=lab)
            ax[1].plot([e['val_top1'] for e in h], label=lab)
    ax[0].set_title('train loss', fontsize=8); ax[1].set_title('validation top-1 (%)', fontsize=8)
    for a in ax:
        a.set_xlabel('epoch', fontsize=8); a.tick_params(labelsize=7)
    ax[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, 'training_curves.png'), dpi=300)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='results/runs')
    p.add_argument('--out', default='results')
    a = p.parse_args()
    main(a.runs, a.out)
