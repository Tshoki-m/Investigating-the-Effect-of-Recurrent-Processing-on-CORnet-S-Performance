"""Print an environment.yml with the EXACT versions installed in the current interpreter.
Run this in the same Colab session that produced the results, then commit the output as environment.yml:
    python src/print_env.py > environment.yml
"""
import platform
import sys

import matplotlib
import numpy
import torch
import torchvision

py = '.'.join(map(str, sys.version_info[:3]))
print(f"""name: cornet-repro
channels:
  - conda-forge
dependencies:
  - python={py}
  - pip
  - numpy={numpy.__version__}
  - matplotlib={matplotlib.__version__}
  - pip:
      - torch=={torch.__version__.split('+')[0]}
      - torchvision=={torchvision.__version__.split('+')[0]}
      - git+https://github.com/dicarlolab/CORnet   # only for src/check_against_official.py
# generated on {platform.platform()} ; CUDA {torch.version.cuda} ; GPU build {torch.cuda.is_available()}""")
