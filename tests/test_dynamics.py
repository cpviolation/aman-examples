from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from tensorwaves.function.sympy import create_function

from aman_examples.dynamics import BelleBreitWigner, InverseRowProduct


@pytest.mark.parametrize("row", [0, 2])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_inverse_row_product(backend, row):
    x = sp.Symbol("x", real=True)
    matrix = [2 + x, sp.I, 0, 1, 3 * x, x**2, sp.Rational(1, 2), 0, 4]
    vector = [1 + x, 2, sp.I * x]
    expr = InverseRowProduct(row, 3, *matrix, *vector)
    assert expr.func(*expr.args) == expr  # rebuilt by doit() and substitutions
    xs = np.linspace(0.5, 2.0, 7)
    function = create_function(expr, backend=backend)
    expected = [
        np.linalg.inv(
            np.array([complex(sp.sympify(m).subs(x, t)) for m in matrix]).reshape(3, 3)
        )[row]
        @ np.array([complex(sp.sympify(v).subs(x, t)) for v in vector])
        for t in xs
    ]
    np.testing.assert_allclose(np.asarray(function({"x": xs})), expected, rtol=1e-12)
    assert (
        sp.latex(expr)
        == Rf"\left[\mathbf{{M}}^{{-1}} \mathbf{{v}}\right]_{{{row}}}^{{(3)}}"
    )


def test_breit_wigner_at_pole():
    """At s = m0^2 the form factors are 1 and the width is Gamma0."""
    m0, gamma0 = 0.8937, 0.0472
    expr = BelleBreitWigner(
        m0**2, m0, gamma0, 0.1396, 0.4976, 0.1396, 1.8648, 1.8648, 1.5, 5.0, L=1
    )
    assert complex(expr.doit().evalf()) == pytest.approx(
        1 / (-1j * m0 * gamma0), rel=1e-12
    )
