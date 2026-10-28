"""Numerically check cornet_s.py against the official implementation (needs internet).
    pip install git+https://github.com/dicarlolab/CORnet
    python check_against_official.py
Loads the official pretrained weights into BOTH models and compares outputs in eval mode.
"""
import torch

from cornet_s import CORnet_S

import cornet  # official package

off = cornet.cornet_s(pretrained=True, map_location='cpu')
off = getattr(off, 'module', off).cpu().eval()   # official loader may put the model on the GPU
mine = CORnet_S((2, 4, 2))
sd = {k.replace('module.', '', 1): v for k, v in off.state_dict().items()}
print(mine.load_state_dict(sd, strict=True))   # raises on any key/shape mismatch
mine.eval()

torch.manual_seed(0)
x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    a, b = off(x), mine(x)
print('max abs diff:', (a - b).abs().max().item(), '| same argmax:', bool((a.argmax(1) == b.argmax(1)).all()))
assert torch.allclose(a, b, atol=1e-4), 'outputs differ: re-implementation is NOT equivalent'
print('OK: re-implementation matches official CORnet-S')
