"""Run the Figure-4 recurrence ablation: train each config x seed sequentially.
Edit CONFIGS/SEEDS or pass extra flags (they are forwarded to train.py), e.g.
  python run_all.py --data D:/imagenet --workers 8 --amp
  python run_all.py --data D:/imagenet --n-classes 100 --amp          # reduced: ImageNet-100
"""
import subprocess
import sys

CONFIGS = [(2, 4, 2), (1, 1, 1), (2, 2, 2), (4, 4, 4)]   # baseline first
SEEDS = [0]                                              # add 1, 2 if compute allows

for seed in SEEDS:
    for cfg in CONFIGS:
        cmd = [sys.executable, 'train.py', '--times', *map(str, cfg), '--seed', str(seed), *sys.argv[1:]]
        print('>>', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)   # train.py resumes automatically if interrupted
