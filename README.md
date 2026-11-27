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
| `src/aman_examples/phasespace.py` | Uniform Dalitz-plot sampling (with exponential decay times for mixing) as TensorWaves `DataGenerator`s (no TensorFlow needed) |
| `src/aman_examples/mixing.py` | D0–D0bar mixing: time-dependent rate in terms of the D0 and D0bar amplitudes, computed per event by a data transformer; parametrised by x, y, q/p or by the bin-flip x_CP, y_CP, Δx, Δy |
| `src/aman_examples/acceptance.py` | Acceptance models in (m²(K⁰ₛπ∓), m²(π⁺π⁻), t, cos θ_hel): functions, 1–3D histograms and their products, and the ROOT files of the `EfficiencyModel` of d02kspipi_toys |
| `src/aman_examples/parametric_acceptance.py` | Parametric acceptance in (s+, s0, t): logistic of a free set of monomials in Dalitz-plot and time features, times a decay-time turn-on; the v4 model of d02kspipi_run3; fit to simulated samples |
| `src/aman_examples/binflip.py` | BaBar 2008 equal-Δδ_D binning, bin-flip ratios R_b(t), strong-phase parameters c_b, s_b, and the bin-flip fit of x_CP, y_CP, Δx, Δy |
| `src/aman_examples/estimator.py` | `SimultaneousEstimator`: sum of TensorWaves estimators (e.g. D0 and D0bar likelihoods) with shared parameters |
| `src/aman_examples/ampgen/` | Models from AmpGen options files: parsing, AmpGen's daughter ordering and orbital rules, covariant spin factors and AmpGen's lineshapes (Breit-Wigner, Gounaris-Sakurai, Flatté, FOCUS, K-matrix, spline running widths) |
| `src/aman_examples/d02kkpipi.py` | The LHCb D0 → K+ K− π+ π− model (AmpGen `DtoKKpipi_v2.opt`, 55 decay chains) and four-body phase space |
| `src/aman_examples/d02k3pi.py` | The LHCb D0 → K− π+ π+ π− (CF, 33 chains) and D0 → K+ π− π− π+ (DCS, 15 chains) models, EPJC 78 (2018) 443, from AmpGen's options files |
| `src/aman_examples/d024pi.py` | The BESIII D0 → π+ π− π+ π− model, CPC 48 (2024) 083001, in the paper's formalism (Zemach tensors, K-matrix with two-S-wave expansion, 3π running widths), evaluated in stages of per-event building blocks as in AmpGen |
| `docs/belle-model.ipynb` | The model, its Dalitz plot and fit fractions |
| `docs/toy-fit.ipynb` | Toy generation and a fit of all free couplings |
| `docs/mixing.ipynb` | Mixing toys with the parametric (v4) acceptance, fit of x and y, bin-flip ratios and strong phases |
| `docs/mixing-cpv.ipynb` | Mixing and CP violation: D0 and D0bar toys, simultaneous amplitude fit and bin-flip fit of x_CP, y_CP, Δx, Δy |
| `docs/acceptance.ipynb` | Mixing toys with a correlated 3D acceptance, and the fit of x and y with and without it |
| `docs/d02kkpipi.ipynb` | The D0 → K+ K− π+ π− model and a toy fit of its 25 free couplings |
| `docs/d02k3pi.ipynb` | The D0 → K∓ π± π± π∓ models: fit fractions and projections |
| `docs/d024pi.ipynb` | The D0 → π+ π− π+ π− model: fit fractions and F+ against the paper, projections |
| `tests/` | Term-by-term comparison with the Belle C++ code (agreement 1e-9) and with the binned Belle amplitude, every D0 → K+ K− π+ π− and K∓ π± π± π∓ chain against AmpGen (1e-5), the D0 → π+ π− π+ π− chains against an independent NumPy implementation (1e-13) and the paper's fit fractions and F+, the mixing rate against the mass-eigenstate evolution, toy fits of the couplings and of x, y, the zCP/Δz parametrisation and the bin-flip expansion, toy fits of x_CP, y_CP, Δx, Δy |

All the examples of tfa_examples are covered.

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

[BSD 3-Clause](LICENSE), except:

- `src/aman_examples/data/ampgen/DtoKKpipi_v2.opt`, `D02Kpipipi.opt`, `D02piKpipi.opt`,
  `Dbar02piKpipi.opt` and `kMatrix.opt` (copies of AmpGen files):
  GPL-2.0-only, as [AmpGen](https://github.com/GooFit/AmpGen); see
  `LICENSES/GPL-2.0-only.txt`;
- `src/aman_examples/data/ampgen/particles.json`: particle properties from the PDG
  (2008 edition), see `src/aman_examples/data/ampgen/README.md`.
