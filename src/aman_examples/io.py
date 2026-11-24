"""Readers for the model parameter files shipped with the package."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

DATA_DIR = Path(str(files("aman_examples") / "data"))
BELLE_MODEL = DATA_DIR / "belle_model.txt"
BELLE_MODEL_DEFINITIONS = DATA_DIR / "belle_model_definitions.txt"


def decode_model(
    filename: str | Path, definitions: str | Path = BELLE_MODEL_DEFINITIONS
) -> dict[str, list[float]]:
    """Read a parameter file whose keys are the numeric codes in ``definitions``.

    Returns:
        A mapping of parameter name to the list of values on its line (value and
        uncertainty).
    """
    names: dict[str, str] = {}
    for line in Path(definitions).read_text().splitlines():
        values = line.split()
        if values and not line.startswith("#"):
            names[values[1]] = values[0]
    model: dict[str, list[float]] = {}
    for line in Path(filename).read_text().splitlines():
        values = line.split()
        if values:
            model[names[values[0]]] = [float(v) for v in values[1:]]
    return model


def load_belle_model() -> dict[str, list[float]]:
    r"""Parameters of the Belle 2018 :math:`D^0 \to K^0_S \pi^+ \pi^-` model."""
    return decode_model(BELLE_MODEL)
