# AmAn examples

[![BSD 3-Clause license](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/license/BSD-3-Clause)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Amplitude-analysis models and studies built on the [ComPWA](https://compwa.github.io)
libraries [AmpForm](https://ampform.rtfd.io) (models as SymPy expressions) and
[TensorWaves](https://tensorwaves.rtfd.io) (toy generation and fits), with
[JAX](https://jax.readthedocs.io) as backend. They reproduce the examples of
[cpviolation/tfa_examples](https://github.com/cpviolation/tfa_examples), which are built
on AmpliTF and TFA2. The repository follows the developer setup of ComPWA projects,
enforced by [ComPWA/policy](https://github.com/ComPWA/policy).

## Contents

| | |
|---|---|
| `src/aman_examples/d02kspipi.py` | Belle 2018 model of D0 → KS π+ π− (BaBar 2008 parametrisation): Breit-Wigner, LASS and ππ K-matrix terms with Zemach spin factors |
| `src/aman_examples/dynamics.py` | The lineshapes as AmpForm `@unevaluated` expressions; the 5×5 K-matrix inverse is evaluated numerically by the backend |
| `src/aman_examples/phasespace.py` | Uniform Dalitz-plot sampling as a TensorWaves `DataGenerator` (no TensorFlow needed) |
| `docs/belle-model.ipynb` | The model, its Dalitz plot and fit fractions |
| `docs/toy-fit.ipynb` | Toy generation and a fit of all free couplings |
| `tests/` | Term-by-term comparison with the Belle C++ code (agreement 1e-9) and with the binned Belle amplitude, and a toy fit |

Still to come from tfa_examples: D0–D0bar mixing toys and decay-time acceptance, the
bin-flip fitter, the strong-phase study and the D0 → K+ K− π+ π− AmpGen model.

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:cpviolation/aman-examples.git
cd aman-examples
uv sync --all-groups          # add --extra cuda for JAX on NVIDIA GPUs (Linux)
uv run pytest -n auto
```

Dependencies are locked to releases at least two weeks old (`exclude-newer` in
`pyproject.toml`).

JAX reserves 75% of the GPU memory when it starts. On a shared GPU, or to run several
notebooks at once, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (allocate on demand) or
`JAX_PLATFORMS=cpu`.

Developer tools, as in ComPWA projects:

```bash
uvx prek install              # or: uvx pre-commit install
uvx prek run --all-files      # policy, Ruff, Tombi, docstrfmt, uv lock
EXECUTE_NB=yes uv run sphinx-build -b html docs docs/_build/html
```

## Licence

[BSD 3-Clause](LICENSE).
