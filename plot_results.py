"""Aggregate runs/*.json -> results table + plot of delta top-1 vs recurrence config,
overlaid with the deltas read off the paper's Figure 4 (row 1)."""
import glob
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Read from Fig. 4 of Kubilius et al. 2019 (change in ImageNet top-1, % points vs 2/4/2).
PAPER = {'1-1-1': -7, '2-2-2': -1, '2-4-2': 0, '4-4-4': 0}
ORDER = ['1-1-1', '2-2-2', '2-4-2', '4-4-4']

runs = defaultdict(list)
for f in glob.glob((sys.argv[1] if len(sys.argv) > 1 else 'runs') + '/cornet_s_*.json'):
    r = json.load(open(f))
    runs['-'.join(map(str, r['times']))].append(r['final_top1'])

base = np.mean(runs['2-4-2'])
rows = []
for k in ORDER:
    if k not in runs:
        continue
    v = np.array(runs[k])
    rows.append((k, len(v), v.mean(), v.std(ddof=1) if len(v) > 1 else float('nan'), v.mean() - base, PAPER[k]))

print(f"{'config':8s} {'n':>2s} {'top1':>7s} {'sd':>6s} {'delta':>7s} {'paper':>6s}")
for k, n, m, s, d, p in rows:
    print(f'{k:8s} {n:2d} {m:7.2f} {s:6.2f} {d:7.2f} {p:6.1f}')
with open('results_table.csv', 'w') as f:
    f.write('config,n_seeds,top1_mean,top1_sd,delta_vs_2-4-2,paper_delta\n')
    for r in rows:
        f.write(','.join(map(str, r)) + '\n')

x = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(x, [r[4] for r in rows], yerr=[0 if np.isnan(r[3]) else r[3] for r in rows],
            fmt='o-', capsize=3, label='this reproduction')
ax.plot(x, [r[5] for r in rows], 's--', label='paper, Fig. 4 (read from figure)')
ax.set_xticks(x); ax.set_xticklabels([r[0].replace('-', '/') for r in rows])
ax.set_xlabel('recurrent steps in V2 / V4 / IT'); ax.set_ylabel('change in ImageNet top-1 (% pts vs 2/4/2)')
ax.axhline(0, color='gray', lw=0.5); ax.legend(); fig.tight_layout(); fig.savefig('recurrence_ablation.png', dpi=200)
