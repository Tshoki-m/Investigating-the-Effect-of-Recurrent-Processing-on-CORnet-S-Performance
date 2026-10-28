"""Train one CORnet-S variant on ImageNet (or a class subset) with the paper's recipe.

Paper (Sec. 2.2): SGD momentum 0.9, batch 256, lr 0.1 divided by 10 every 20
epochs, 43 epochs, cross-entropy. Weight decay is not stated in the paper; 1e-4
is the value I recall from the official run.py -- verify and change if needed.

Data layout: <data>/train/<class>/*.JPEG and <data>/val/<class>/*.JPEG
(the raw ImageNet val set must first be sorted into class folders).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from cornet_s import CORnet_S, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--times', type=int, nargs=3, default=[2, 4, 2], help='V2 V4 IT steps')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=43)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--step-size', type=int, default=20)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--n-classes', type=int, default=0, help='0 = all 1000; else use every k-th class')
    p.add_argument('--img-size', type=int, default=224, help='crop size (use e.g. 128 for a cheaper run)')
    p.add_argument('--max-train-per-class', type=int, default=0, help='0 = all images')
    p.add_argument('--amp', action='store_true', help='mixed precision (faster; note in your report)')
    p.add_argument('--min-scale', type=float, default=0.08, help='RandomResizedCrop min scale (0.35 is saner at 64px)')
    p.add_argument('--no-v1-pool', action='store_true', help='drop V1 max-pool (larger final feature map at low resolution)')
    p.add_argument('--ckpt-every', type=int, default=1, help='save a resume checkpoint every N epochs')
    p.add_argument('--out', default='runs')
    p.add_argument('--name', default=None)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_dataset(root, tf, n_classes, max_per_class=0):
    ds = datasets.ImageFolder(root, tf)
    if n_classes and n_classes < len(ds.classes):
        k = len(ds.classes) // n_classes
        keep = list(range(0, len(ds.classes), k))[:n_classes]
        remap = {old: new for new, old in enumerate(keep)}
        samples = [(p, remap[c]) for p, c in ds.samples if c in remap]
        ds.classes = [ds.classes[i] for i in keep]
        ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}
        ds.samples = samples
    if max_per_class:
        counts, kept = {}, []
        for p, c in ds.samples:
            if counts.get(c, 0) < max_per_class:
                kept.append((p, c)); counts[c] = counts.get(c, 0) + 1
        ds.samples = kept
    ds.imgs = ds.samples
    ds.targets = [c for _, c in ds.samples]
    return ds


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    c1 = c5 = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=amp):
            out = model(x)
        top5 = out.topk(5, dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return 100.0 * c1 / n, 100.0 * c5 / n


def main():
    a = get_args()
    seed_all(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = a.name or f"cornet_s_{a.times[0]}-{a.times[1]}-{a.times[2]}_seed{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, name + '.ckpt')
    res_path = os.path.join(a.out, name + '.json')

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    s = a.img_size
    train_tf = transforms.Compose([transforms.RandomResizedCrop(s, scale=(a.min_scale, 1.0)), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.Resize(int(round(s * 256 / 224))), transforms.CenterCrop(s),
                                 transforms.ToTensor(), norm])
    train_ds = make_dataset(os.path.join(a.data, 'train'), train_tf, a.n_classes, a.max_train_per_class)
    val_ds = make_dataset(os.path.join(a.data, 'val'), val_tf, a.n_classes)
    n_classes = len(train_ds.classes)
    train_dl = torch.utils.data.DataLoader(train_ds, a.bs, shuffle=True, num_workers=a.workers,
                                           pin_memory=True, drop_last=True,
                                           persistent_workers=a.workers > 0)
    val_dl = torch.utils.data.DataLoader(val_ds, a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = CORnet_S(tuple(a.times), num_classes=n_classes, v1_pool=not a.no_v1_pool).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=a.step_size, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp and device == 'cuda')
    crit = nn.CrossEntropyLoss()

    history, start = [], 0
    if os.path.exists(ckpt_path):  # resume
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); history = ck['history']; start = ck['epoch'] + 1
        print(f'resumed {name} from epoch {start}')

    print(f'{name}: {count_params(model)/1e6:.2f}M params, {len(train_ds)} train / {len(val_ds)} val images, '
          f'{n_classes} classes, device={device}')
    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=a.amp and device == 'cuda'):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        top1, top5 = evaluate(model, val_dl, device, a.amp and device == 'cuda')
        history.append(dict(epoch=ep, train_loss=tot / nb, val_top1=top1, val_top5=top5, secs=time.time() - t0))
        print(f'[{name}] ep {ep+1}/{a.epochs} loss {tot/nb:.3f} top1 {top1:.2f} top5 {top5:.2f} ({time.time()-t0:.0f}s)')
        if (ep + 1) % a.ckpt_every == 0 or ep + 1 == a.epochs:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            history=history, epoch=ep), ckpt_path)

    final = history[-1]
    json.dump(dict(name=name, times=a.times, seed=a.seed, n_classes=n_classes, img_size=a.img_size,
                   epochs=a.epochs, params=count_params(model), final_top1=final['val_top1'],
                   final_top5=final['val_top5'], args=vars(a), history=history),
              open(res_path, 'w'), indent=1)
    print('final top-1 (last epoch, no best-epoch selection):', final['val_top1'])


if __name__ == '__main__':
    main()
