"""Reading AmpGen options files and particle tables.

Ported from ``tfaex.ampgen`` of cpviolation/tfa_examples. An AmpGen
(https://github.com/GooFit/AmpGen) options file defines the decay chains as descriptors
with couplings, and particle parameters, with aliases between them.
"""

from __future__ import annotations

import cmath
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def load_particles(file_name):
    """AmpGen particle properties (``two_spin``, ``parity``, ``pdg_id``, ``charge``, ``mass``, ``width``, ``radius``) keyed by AmpGen name."""
    with open(file_name) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------
# Decay descriptors


@dataclass
class Decay:
    name: str
    modifiers: list = field(default_factory=list)
    daughters: list = field(default_factory=list)

    def descriptor(self):
        mods = f"[{';'.join(self.modifiers)}]" if self.modifiers else ""
        ds = (
            "{" + ",".join(d.descriptor() for d in self.daughters) + "}"
            if self.daughters
            else ""
        )
        return self.name + mods + ds


def parse_descriptor(text):
    """Parse ``'D0[D]{K*(892)0{K+,pi-},K*(892)bar0{K-,pi+}}'`` into a :class:`Decay` tree."""
    m = re.match(r"([^\[\]{},]+)(\[([^\]]*)\])?", text)
    if m is None:
        msg = f"Cannot parse decay descriptor '{text}'"
        raise ValueError(msg)
    decay = Decay(m.group(1), [x for x in (m.group(3) or "").split(";") if x])
    rest = text[m.end() :]
    if rest:
        if not (rest.startswith("{") and rest.endswith("}")):
            msg = f"Cannot parse decay descriptor '{text}'"
            raise ValueError(msg)
        depth, start = 0, 1
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    decay.daughters.append(parse_descriptor(rest[start:i]))
            elif ch == "," and depth == 1:
                decay.daughters.append(parse_descriptor(rest[start:i]))
                start = i + 1
    return decay


# ----------------------------------------------------------------------------
# Options files


@dataclass
class Options:
    event_type: list
    couplings: dict  # descriptor -> (complex value, is_free)
    parameters: dict  # name -> (value, is_free)
    ignored: list
    splines: dict = field(default_factory=dict)  # particle -> (n knots, min s, max s)


def _resolve_expression(expr, values, pi=math.pi):
    expr = expr.strip()
    m = re.fullmatch(r"(\S+)\s*([+-])\s*(PI|[0-9.eE+-]+)", expr)
    if m:
        base = _resolve_expression(m.group(1), values, pi)
        offset = pi if m.group(3) == "PI" else float(m.group(3))
        return base + offset if m.group(2) == "+" else base - offset
    if expr in values:
        return values[expr]
    msg = f"Unknown parameter in expression '{expr}'"
    raise KeyError(msg)


N_TOKENS_COUPLING = (
    7  # name, flag, value, error (real or magnitude, then imaginary or phase)
)
N_TOKENS_PARAMETER = 4  # name, flag, value, error


def _lines(file_name):
    """Lines of an options file, with ``Import`` lines replaced by the imported file.

    ``Import $AMPGENROOT/options/<file>`` (AmpGen's own options directory) is read from
    the directory of the importing file, other paths relative to it.
    """
    path = Path(file_name)
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith("Import "):
            target = line.split(maxsplit=1)[1]
            target = target.removeprefix("$AMPGENROOT/options/")
            yield from _lines(path.parent / target)
        elif line:
            yield line


def read_options(file_name):  # ruff: ignore[complex-structure, too-many-branches]
    """Read an AmpGen options file: event type, couplings and parameters.

    Coupling values are complex (polar or Cartesian according to
    ``CouplingConstant::Coordinates``, phases in radians or, with
    ``CouplingConstant::AngularUnits deg``, degrees); flag 2 means fixed. Aliases ``A =
    B`` (optionally ``B +- number|PI``) are resolved, also for the real and imaginary
    (or magnitude and phase) parts of couplings via ``_Re``/``_Im``. ``Import`` lines
    include another options file; later definitions override earlier ones.
    """
    event_type, polar, degrees = None, False, False
    raw_couplings, raw_params, aliases, ignored, splines = {}, {}, [], [], {}
    for line in _lines(file_name):
        if "=" in line:
            lhs, rhs = (x.strip() for x in line.split("=", 1))
            aliases.append((lhs, rhs))
            continue
        tokens = line.split()
        key = tokens[0]
        if key == "EventType":
            event_type = tokens[1:]
        elif key == "CouplingConstant::Coordinates":
            polar = tokens[1] == "polar"
        elif key == "CouplingConstant::AngularUnits":
            degrees = tokens[1] == "deg"
        elif key.endswith("::Spline") and len(tokens) == N_TOKENS_PARAMETER:
            splines[key[: -len("::Spline")]] = (
                int(tokens[1]),
                float(tokens[2]),
                float(tokens[3]),
            )
        elif len(tokens) == N_TOKENS_COUPLING:
            raw_couplings[key] = (
                float(tokens[2]),
                float(tokens[5]),
                tokens[1] != "2" or tokens[4] != "2",
            )
        elif len(tokens) == N_TOKENS_PARAMETER:
            raw_params[key] = (float(tokens[2]), tokens[1] != "2")
        else:
            ignored.append(line)

    values = {k: v[0] for k, v in raw_params.items()}
    for k, (re_, im_, _) in raw_couplings.items():
        values[k + "_Re"], values[k + "_Im"] = re_, im_
    for lhs, rhs in aliases:  # in file order
        values[lhs] = _resolve_expression(rhs, values, 180.0 if degrees else math.pi)
        if lhs.endswith(("_Re", "_Im")):
            name = lhs[:-3]
            re_, im_ = values.get(name + "_Re"), values.get(name + "_Im")
            if re_ is not None and im_ is not None:
                raw_couplings[name] = (re_, im_, False)
        else:
            raw_params[lhs] = (values[lhs], False)

    couplings = {}
    for name, (a, b, free) in raw_couplings.items():
        if polar:
            couplings[name] = (cmath.rect(a, math.radians(b) if degrees else b), free)
        else:
            couplings[name] = (complex(a, b), free)
    params = {k: (values[k], free) for k, (_, free) in raw_params.items()}
    return Options(event_type, couplings, params, ignored, splines)
