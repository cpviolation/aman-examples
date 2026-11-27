# AmpGen model files

| File | Origin | Licence |
| --- | --- | --- |
| `DtoKKpipi_v2.opt` | Unmodified copy of `options/DtoKKpipi_v2.opt` from [AmpGen](https://github.com/GooFit/AmpGen) (commit `fc651a7`): the LHCb D0 -> K+ K- pi+ pi- model, JHEP 02 (2019) 126 | GPL-2.0-only, as AmpGen ([`LICENSES/GPL-2.0-only.txt`](../../../../LICENSES/GPL-2.0-only.txt)) |
| `D02Kpipipi.opt`, `D02piKpipi.opt`, `Dbar02piKpipi.opt` | Unmodified copies of the files of the same name in `options/` of AmpGen (commit `fc651a7`): the LHCb D0 -> K- pi+ pi+ pi- (CF), D0 -> K+ pi- pi- pi+ (DCS) and Dbar0 -> K+ pi- pi- pi+ models, Eur. Phys. J. C 78 (2018) 443 | GPL-2.0-only, as AmpGen ([`LICENSES/GPL-2.0-only.txt`](../../../../LICENSES/GPL-2.0-only.txt)) |
| `kMatrix.opt` | Unmodified copy of `options/kMatrix.opt` of AmpGen (commit `fc651a7`): the pi pi S-wave K-matrix parameters imported by the K3pi models | GPL-2.0-only, as AmpGen ([`LICENSES/GPL-2.0-only.txt`](../../../../LICENSES/GPL-2.0-only.txt)) |
| `particles.json` | Particle properties from the Review of Particle Physics, 2008 edition (C. Amsler et al. (Particle Data Group), Phys. Lett. B 667, 1 (2008)), as compiled in AmpGen's `mass_width.csv`, and AmpGen's pseudo-particles; dumped with `macros/ampgen/DumpTerms.cpp` of cpviolation/tfa_examples | Factual data; please cite the PDG |

These files are not covered by the BSD 3-Clause licence of the rest of the repository.
The reference amplitudes in `tests/data/ampgen_d02kkpipi_v2_reference.npz`,
`ampgen_d02kpipipi_reference.npz` and `ampgen_d02pikpipi_reference.npz` were computed
with AmpGen `fc651a7`, as described in `macros/ampgen/README.md` of
cpviolation/tfa_examples.
