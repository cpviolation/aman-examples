r"""Covariant (Lorentz-tensor) spin factors with the conventions of AmpGen.

Ported from :mod:`amplitf.covariant` of cpviolation/AmpliTF, which follows
``src/Vertex.cpp`` of AmpGen (https://github.com/GooFit/AmpGen), as SymPy expressions.
Vectors are lists of four expressions with upper (contravariant) indices, components
ordered :math:`(x, y, z, t)`; rank-2 tensors are :math:`4\times 4` nested lists. The
metric is :math:`g = \mathrm{diag}(-1, -1, -1, +1)`.

For a two-body vertex :math:`R \to a\,b`, :math:`P = p_a + p_b`, :math:`Q = p_a - p_b`,
and ``V1``, ``V2`` are the spin tensors of :math:`a` and :math:`b` (an expression for
spin 0). The vertices are named as in AmpGen: ``<R>_<a><b>_<L>`` with S/V/T for spin
0/1/2 and S/P/D for the orbital angular momentum.

Levi-Civita: AmpGen defines :math:`\epsilon^{xyzt} = +1` and contracts it with all
indices lowered, :math:`\epsilon_{xyzt} = \det g = -1`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable

Vector = list[sp.Expr]
Tensor = list[list[sp.Expr]]

METRIC = (-1, -1, -1, 1)
INDICES = range(4)


def _permutation_sign(perm: tuple[int, ...]) -> int:
    sign, perm = 1, list(perm)
    for i in INDICES:
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            sign = -sign
    return sign


#: Non-zero entries of the Levi-Civita tensor with lower indices, eps_{xyzt} = -1
LEVI_CIVITA_LOWER = {
    perm: -_permutation_sign(perm) for perm in itertools.permutations(INDICES)
}


def lower(v: Vector) -> Vector:
    r""":math:`v_\mu = g_{\mu\nu} v^\nu`."""
    return [METRIC[i] * v[i] for i in INDICES]


def dot(a: Vector, b: Vector) -> sp.Expr:
    r"""Minkowski product :math:`a^\mu g_{\mu\nu} b^\nu`."""
    return sum(METRIC[i] * a[i] * b[i] for i in INDICES)


def add(a: Vector, b: Vector, factor: sp.Expr = 1) -> Vector:
    """:math:`a + f b`."""
    return [a[i] + factor * b[i] for i in INDICES]


def scale(v: Vector, factor: sp.Expr) -> Vector:
    return [factor * v[i] for i in INDICES]


def outer(a: Vector, b: Vector) -> Tensor:
    return [[a[i] * b[j] for j in INDICES] for i in INDICES]


def spin1_projector(p: Vector) -> Tensor:
    r""":math:`S^{\mu\nu} = g^{\mu\nu} - p^\mu p^\nu / p^2`."""
    p2 = dot(p, p)
    return [
        [(METRIC[i] if i == j else 0) - p[i] * p[j] / p2 for j in INDICES]
        for i in INDICES
    ]


def orbital_pwave(p: Vector, q: Vector) -> Vector:
    r""":math:`L^\mu = q^\mu - p^\mu (p\cdot q)/p^2`."""
    return add(q, p, -dot(p, q) / dot(p, p))


def orbital_dwave(p: Vector, q: Vector) -> Tensor:
    r""":math:`L^\mu L^\nu - (L\cdot L)/3\, S^{\mu\nu}`."""
    l1 = orbital_pwave(p, q)
    s = spin1_projector(p)
    l2 = dot(l1, l1) / 3
    return [[l1[i] * l1[j] - l2 * s[i][j] for j in INDICES] for i in INDICES]


def contract_lower(t: Tensor, a: Vector, b: Vector) -> sp.Expr:
    r""":math:`t^{\mu\nu} a_\mu b_\nu` for upper-index vectors :math:`a, b`."""
    return sum(
        METRIC[i] * METRIC[j] * t[i][j] * a[i] * b[j] for i in INDICES for j in INDICES
    )


def contract_tensors(a: Tensor, b: Tensor) -> sp.Expr:
    r""":math:`a^{\mu\nu} b_{\mu\nu}`."""
    return sum(
        METRIC[i] * METRIC[j] * a[i][j] * b[i][j] for i in INDICES for j in INDICES
    )


def levi_civita_contract(a: Vector, b: Vector, c: Vector, d: Vector) -> sp.Expr:
    r""":math:`\epsilon_{\mu\nu\alpha\beta} a^\mu b^\nu c^\alpha d^\beta`."""
    return sum(
        s * a[i] * b[j] * c[k] * d[m] for (i, j, k, m), s in LEVI_CIVITA_LOWER.items()
    )


# Vertices R -> a b, as in AmpGen's Vertex.cpp
def S_SS_S(P: Vector, Q: Vector, V1: sp.Expr, V2: sp.Expr) -> sp.Expr:
    return V1 * V2


def S_VV_S(P: Vector, Q: Vector, V1: Vector, V2: Vector) -> sp.Expr:
    return dot(V1, V2)


def S_VV_P(P: Vector, Q: Vector, V1: Vector, V2: Vector) -> sp.Expr:
    return levi_civita_contract(V1, V2, orbital_pwave(P, Q), P)


def S_VV_D(P: Vector, Q: Vector, V1: Vector, V2: Vector) -> sp.Expr:
    return contract_lower(orbital_dwave(P, Q), V1, V2)


def S_VS_P(P: Vector, Q: Vector, V1: Vector, V2: sp.Expr) -> sp.Expr:
    return dot(V1, orbital_pwave(P, Q)) * V2


def S_TV_P(P: Vector, Q: Vector, V1: Tensor, V2: Vector) -> sp.Expr:
    return contract_lower(V1, orbital_pwave(P, Q), V2)


def S_TS_D(P: Vector, Q: Vector, V1: Tensor, V2: sp.Expr) -> sp.Expr:
    return contract_tensors(orbital_dwave(P, Q), V1) * V2


def V_SS_P(P: Vector, Q: Vector, V1: sp.Expr, V2: sp.Expr) -> Vector:
    return scale(orbital_pwave(P, Q), V1 * V2)


def T_SS_D(P: Vector, Q: Vector, V1: sp.Expr, V2: sp.Expr) -> Tensor:
    return [[V1 * V2 * x for x in row] for row in orbital_dwave(P, Q)]


def V_VS_S(P: Vector, Q: Vector, V1: Vector, V2: sp.Expr) -> Vector:
    s, v1 = spin1_projector(P), lower(V1)
    return [sum(s[m][n] * v1[n] for n in INDICES) * V2 for m in INDICES]


def V_VS_D(P: Vector, Q: Vector, V1: Vector, V2: sp.Expr) -> Vector:
    s, l2 = spin1_projector(P), orbital_dwave(P, Q)
    l2_v1 = [  # L_{2,nu alpha} V1^alpha
        METRIC[n] * sum(METRIC[a] * l2[n][a] * V1[a] for a in INDICES) for n in INDICES
    ]
    return [sum(s[m][n] * l2_v1[n] for n in INDICES) * V2 for m in INDICES]


VERTICES: dict[str, Callable] = {
    "S_SS_S": S_SS_S,
    "S_TS_D": S_TS_D,
    "S_TV_P": S_TV_P,
    "S_VS_P": S_VS_P,
    "S_VV_D": S_VV_D,
    "S_VV_P": S_VV_P,
    "S_VV_S": S_VV_S,
    "T_SS_D": T_SS_D,
    "V_SS_P": V_SS_P,
    "V_VS_D": V_VS_D,
    "V_VS_S": V_VS_S,
}


def vertex(name: str, P: Vector, Q: Vector, V1, V2):
    """Evaluate the vertex ``name`` (e.g. ``"S_VV_P"``)."""
    if name not in VERTICES:
        msg = f"Vertex {name} is not implemented (available: {sorted(VERTICES)})"
        raise NotImplementedError(msg)
    return VERTICES[name](P, Q, V1, V2)
