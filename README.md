# TensorGrad

Project 2/10 in the from-first-principles Dreamer build.

TensorGrad evolves MiniGrad from scalar values to NumPy tensors while keeping
the same reverse-mode autodiff ideas. The existing MiniGrad files are preserved
as the tested starting point and reference.

See `TODO.md` for the build order.

## Installation

TensorGrad requires Python 3.10 or newer. Install it directly from GitHub:

```bash
python -m pip install "benyebai-tensorgrad @ git+https://github.com/benyebai/TensorGrad.git@main"
```

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/benyebai/TensorGrad.git
cd TensorGrad
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```

After installation:

```python
from tensorgrad import Tensor
```
