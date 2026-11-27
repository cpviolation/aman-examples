"""Independent NumPy implementation of the BESIII D0 -> pi+ pi- pi+ pi- amplitudes.

A reference for tests/test_d024pi.py, written separately from aman_examples.d024pi
(plain arrays and einsum instead of SymPy expressions); only the running-width tables of
a1(1260), a1(1640) and pi(1300) are taken from the package. Vectors are arrays (..., 4)
with components (x, y, z, E), metric diag(-1, -1, -1, +1) and epsilon_{xyzE} = -1;
rank-2 tensors (..., 4, 4) have upper indices.
"""

from __future__ import annotations

import itertools

import numpy as np

from aman_examples import d024pi as model

G = np.diag([-1.0, -1.0, -1.0, 1.0])
EPS = np.zeros((4, 4, 4, 4))
for _perm in itertools.permutations(range(4)):
    _p, _sign = list(_perm), 1
    for _i in range(4):
        while _p[_i] != _i:
            _j = _p[_i]
            _p[_i], _p[_j] = _p[_j], _p[_i]
            _sign = -_sign
    EPS[_perm] = -_sign

MPI, MPI0, MK, MK0 = model.M_PI, model.M_PI0, model.M_K, model.M_K0
R_D, R_RES = 5.0, 3.0


def dot(a, b):
    return np.einsum("...i,ij,...j->...", a, G, b)


def lower(v):
    return v @ G


def p1(p):
    return -G + p[..., :, None] * p[..., None, :] / dot(p, p)[..., None, None]


def t1(pa, pb, pc):
    r = pb - pc
    return r - pa * (dot(pa, r) / dot(pa, pa))[..., None]


def t2(pa, pb, pc):
    rp = t1(pa, pb, pc)
    return rp[..., :, None] * rp[..., None, :] + dot(rp, rp)[..., None, None] / 3 * p1(
        pa
    )


def contract_vtv(a, t, b):
    return np.einsum("...i,...ij,...j->...", lower(a), t, lower(b))


def contract_tt(x, y):
    return np.einsum("...ij,ik,jl,...kl->...", x, G, G, y)


def p2_contract(p, x, y):
    """P^(2)_{mu nu mu' nu'}(p) x^{mu nu} y^{mu' nu'}, without simplifications."""
    q = G @ p1(p) @ G
    a = np.einsum("...ac,...ab,...bd,...cd->...", q, x, q, y)
    b = np.einsum("...ad,...ab,...bc,...cd->...", q, x, q, y)
    tx = np.einsum("...ab,...ab->...", q, x)
    ty = np.einsum("...ab,...ab->...", q, y)
    return (a + b) / 2 - tx * ty / 3


def eps4(a, b, c, d):
    return np.einsum("ijkl,...i,...j,...k,...l->...", EPS, a, b, c, d)


def mass2(p):
    return dot(p, p)


def q2_two_body(s, m1sq, m2sq):
    m1, m2 = np.sqrt(m1sq), np.sqrt(m2sq)
    return (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4 * s)


def q2_parent(parent, d1, d2):
    return q2_two_body(mass2(parent), mass2(d1), mass2(d2))


def barrier(L, q2, radius):
    qr2 = 1 / radius**2
    if L == 0:
        return np.ones_like(q2)
    if L == 1:
        return np.sqrt(2 / (q2 + qr2))
    return np.sqrt(13 / (q2**2 + 3 * q2 * qr2 + 9 * qr2**2))


def rbw(s, m0, width):
    return 1 / (m0**2 - s - 1j * np.sqrt(s) * width)


def f2_ls(p1_, p2_):
    s = mass2(p1_ + p2_)
    m0, g0 = model.RESONANCES["f2(1270)"]
    q2 = q2_two_body(s, MPI**2, MPI**2)
    q02 = q2_two_body(m0**2, MPI**2, MPI**2)
    width = (
        g0
        * (q2 / q02) ** 2.5
        * m0**2
        / s
        * (barrier(2, q2, R_RES) / barrier(2, q02, R_RES)) ** 2
    )
    return rbw(s, m0, width) * barrier(2, q2, R_RES)


def rho_ls(p_plus, p_minus, name="rho(770)"):
    s = mass2(p_plus + p_minus)
    m0, g0 = model.RESONANCES[name]
    q2 = q2_two_body(s, MPI**2, MPI**2)
    q, m = np.sqrt(q2), np.sqrt(s)
    q0 = np.sqrt(q2_two_body(m0**2, MPI**2, MPI**2))

    def h(mm, qq):
        return 2 / np.pi * qq / mm * np.log((mm + 2 * qq) / (2 * MPI))

    dh = h(m0, q0) * (1 / (8 * q0**2) - 1 / (2 * m0**2)) + 1 / (2 * np.pi * m0**2)
    f = g0 * m0**2 / q0**3 * (q**2 * (h(m, q) - h(m0, q0)) + (m0**2 - s) * q0**2 * dh)
    d = (
        3 / np.pi * MPI**2 / q0**2 * np.log((m0 + 2 * q0) / (2 * MPI))
        + m0 / (2 * np.pi * q0)
        - MPI**2 * m0 / (np.pi * q0**3)
    )
    gs = (1 + d * g0 / m0) / (m0**2 - s + f - 1j * m0 * g0 * (q / q0) ** 3 * (m0 / m))
    return gs * barrier(1, q2, R_RES)


def flatte_f0(s):
    m0, g_pipi, ratio = model.F0_FLATTE

    def rho(m1, m2):
        return np.sqrt((1 - (m1 + m2) ** 2 / s) * (1 - (m1 - m2) ** 2 / s) + 0j)

    rho_pipi = 2 / 3 * rho(MPI, MPI) + 1 / 3 * rho(MPI0, MPI0)
    rho_kk = (rho(MK, MK) + rho(MK0, MK0)) / 2
    return 1 / (m0**2 - s - 1j * g_pipi * (rho_pipi + ratio * rho_kk))


# --- K-matrix
KM_MASSES = np.array([0.651, 1.2036, 1.55817, 1.21, 1.82206])
KM_G = np.array([
    [0.22889, -0.55377, 0, -0.39899, -0.34639],
    [0.94128, 0.55095, 0, 0.39065, 0.31503],
    [0.36856, 0.23888, 0.55639, 0.1834, 0.18681],
    [0.3365, 0.40907, 0.85679, 0.19906, -0.00984],
    [0.18171, -0.17558, -0.79658, -0.00355, 0.22358],
])
KM_FSCATT = np.array([0.23399, 0.15044, -0.20545, 0.32825, 0.35412])
KM_MPI, KM_MK, KM_META, KM_METAP = 0.13957, 0.493677, 0.547862, 0.95778
S0_PROD = -5.0


def _phsp(s, m1, m2):
    """BaBar's phase-space factor sqrt(1 - (m1 + m2)^2/s), +i sqrt(...) below threshold."""
    return np.sqrt(1 - (m1 + m2) ** 2 / s + 0j)


def kmatrix_row(s):
    """Row pipi of (1 - i K rho)^-1, shape (n, 5)."""
    s = np.asarray(s, dtype=float)
    adler = (1 - -0.15) * (s - KM_MPI**2 / 2) / (s + 0.15)
    k = (
        KM_G[None, :, :, None]
        * KM_G[None, :, None, :]
        / (KM_MASSES[None, :, None, None] ** 2 - s[:, None, None, None])
    ).sum(axis=1)
    scatt = (1 + 3.92637) / (s + 3.92637)
    k[:, 0, :] += KM_FSCATT[None, :] * scatt[:, None]
    k[:, 1:, 0] += KM_FSCATT[None, 1:] * scatt[:, None]
    k *= adler[:, None, None]
    poly = np.polyval(
        [1.07885, 0.13655, -0.29744, -0.20840, 0.13851, -0.01933, 0.00051], s
    )
    four_pi = np.where(s > 1, _phsp(s, 2 * KM_MPI, 2 * KM_MPI), poly + 0j)
    rho = np.stack(
        [
            _phsp(s, KM_MPI, KM_MPI),
            _phsp(s, KM_MK, KM_MK),
            four_pi,
            _phsp(s, KM_META, KM_META),
            _phsp(s, KM_META, KM_METAP),
        ],
        axis=-1,
    )
    m = np.eye(5)[None] - 1j * k * rho[:, None, :]
    return np.linalg.inv(m)[:, 0, :]


def pole(s, a):
    return kmatrix_row(s) @ KM_G[a] / (KM_MASSES[a] ** 2 - s)


def prod(s, channel):
    return kmatrix_row(s)[:, channel] * (1 - S0_PROD) / (s - S0_PROD)


def swave(s, terms):
    return sum(
        c * (pole(s, int(t[4:])) if t.startswith("pole") else prod(s, int(t[4:])))
        for t, c in terms.items()
    )


def running_propagator(s, name):
    m0, g0 = model.RESONANCES[name]
    table_s, table_ratio = model.running_width_table(name)
    return 1 / (m0**2 - s - 1j * np.sqrt(s) * g0 * np.interp(s, table_s, table_ratio))


# --- amplitudes
def a1_vector(pa, pp, pm, pb, rho_sign, f_sign, decays):  # ruff: ignore[too-many-positional-arguments]
    p_iso = pp + pm
    t_iso = t1(p_iso, pp, pm)
    q2a = q2_parent(pa, p_iso, pb)
    rho = rho_ls(pp, pm)
    out = np.zeros(pa.shape, dtype=complex)
    if rho_sign:
        out += (
            rho_sign
            * decays["rho_S"]
            * np.einsum("...ij,...j->...i", p1(pa), lower(t_iso))
            * rho[..., None]
        )
        if "rho_D" in decays:
            vec = np.einsum("...ij,...j->...i", t2(pa, p_iso, pb), lower(t_iso))
            out += (
                rho_sign
                * decays["rho_D"]
                * vec
                * (rho * barrier(2, q2a, R_RES))[..., None]
            )
    if f_sign and "f2_P" in decays:
        t_a, tt = t1(pa, p_iso, pb), t2(p_iso, pp, pm)
        pm_ = p1(pa) @ G
        x = pm_ @ tt @ np.swapaxes(pm_, -1, -2)
        trace = np.einsum("...ab,ac,bd,...cd->...", p1(pa), G, G, tt)
        vec = (
            np.einsum("...ij,...j->...i", x, lower(t_a))
            - np.einsum("...ij,...j->...i", p1(pa), lower(t_a)) * trace[..., None] / 3
        )
        out += (
            f_sign
            * decays["f2_P"]
            * vec
            * (f2_ls(pp, pm) * barrier(1, q2a, R_RES))[..., None]
        )
    terms = {k[2:]: v for k, v in decays.items() if k.startswith("S_")}
    if f_sign and terms:
        out += (
            f_sign
            * t1(pa, p_iso, pb)
            * (swave(mass2(p_iso), terms) * barrier(1, q2a, R_RES))[..., None]
        )
    return out


def pi1300(pa, pp, pm, pb, rho_sign):
    p_iso = pp + pm
    amp = (
        rho_sign
        * dot(t1(pa, p_iso, pb), t1(p_iso, pp, pm))
        * rho_ls(pp, pm)
        * barrier(1, q2_parent(pa, p_iso, pb), R_RES)
    )
    terms = {k[2:]: v for k, v in model.PI1300_DECAYS.items() if k.startswith("S_")}
    return amp + swave(mass2(p_iso), terms)


def chains(p):  # ruff: ignore[too-many-locals, too-many-statements]
    """Unit-coupling chains of aman_examples.d024pi at momenta p (n, 4, 4), (pi+, pi-, pi+, pi-)."""
    plus1, minus1, plus2, minus2 = (p[:, i] for i in range(4))
    pd = plus1 + minus1 + plus2 + minus2
    out = {}

    def add(key, value):
        out[key] = out.get(key, 0) + value

    for a, b, c, d in [
        (plus1, minus1, plus2, minus2),
        (plus2, minus1, plus1, minus2),
        (plus1, minus2, plus2, minus1),
        (plus2, minus2, plus1, minus1),
    ]:
        for charge, (pp, pm, pb, pdb, rs) in {
            "+": (a, b, c, d, 1),
            "-": (a, b, d, c, -1),
        }.items():
            other = "-" if charge == "+" else "+"
            pa = pp + pm + pb
            s_a = mass2(pa)
            t_d = t1(pd, pa, pdb)
            bd1 = barrier(1, q2_parent(pd, pa, pdb), R_D)
            vec = a1_vector(pa, pp, pm, pb, rs, 1, model.A1_DECAYS)
            add(
                f"a1(1260){charge} pi{other}",
                dot(t_d, vec) * bd1 * running_propagator(s_a, "a1(1260)"),
            )
            vec = a1_vector(pa, pp, pm, pb, rs, 0, {"rho_S": 1.0})
            add(
                f"a1(1640){charge} pi{other}",
                dot(t_d, vec) * bd1 * running_propagator(s_a, "a1(1640)"),
            )
            if charge == "+":
                t_a = t1(pa, pp + pm, pb)
                add(
                    "a1(1420)+ pi-",
                    dot(t_d, t_a)
                    * barrier(1, q2_parent(pa, pp + pm, pb), R_RES)
                    * flatte_f0(mass2(pp + pm))
                    * bd1
                    * rbw(s_a, *model.RESONANCES["a1(1420)"]),
                )
            prho = pp + pm
            w = np.einsum("...ij,jk,...k->...i", p1(pa), G, t1(prho, pp, pm))
            t_mixed = np.einsum("...lk,kb->...lb", t2(pa, prho, pb), G)
            y_low = np.einsum("alsr,...r,...s,...lb->...ab", EPS, pa, w, t_mixed)
            y_up = np.einsum("ia,...ab,bj->...ij", G, y_low, G)
            spin = p2_contract(pa, t2(pd, pa, pdb), y_up)
            add(
                f"a2(1320){charge} pi{other}",
                rs
                * spin
                * barrier(2, q2_parent(pd, pa, pdb), R_D)
                * barrier(2, q2_parent(pa, prho, pb), R_RES)
                * rho_ls(pp, pm)
                * rbw(s_a, *model.RESONANCES["a2(1320)"]),
            )
            add(
                f"pi(1300){charge} pi{other}",
                pi1300(pa, pp, pm, pb, rs) * running_propagator(s_a, "pi(1300)"),
            )
        r1, r2 = a + b, c + d
        s1, s2 = mass2(r1), mass2(r2)
        q2d = q2_parent(pd, r1, r2)
        t_r1, t_r2 = t1(r1, a, b), t1(r2, c, d)
        rho1, rho2, rho2h = rho_ls(a, b), rho_ls(c, d), rho_ls(c, d, "rho(1450)")
        t_d = t1(pd, r1, r2)
        p_spin = eps4(t_d, t_r1, t_r2, pd) * barrier(1, q2d, R_D)
        d_spin = contract_vtv(t_r1, t2(pd, r1, r2), t_r2) * barrier(2, q2d, R_D)
        add("rho rho [S]", 0.5 * dot(t_r1, t_r2) * rho1 * rho2)
        add("rho rho [P]", 0.5 * p_spin * rho1 * rho2)
        add("rho rho [D]", 0.5 * d_spin * rho1 * rho2)
        add("rho rho(1450) [P]", p_spin * rho1 * rho2h)
        add("rho rho(1450) [D]", d_spin * rho1 * rho2h)
        vs = dot(t_d, t_r1) * barrier(1, q2d, R_D) * rho1
        add("rho S [beta1]", vs * pole(s2, 0))
        add("rho S [f_pipi]", vs * prod(s2, 0))
        add("rho S [f_KK]", vs * prod(s2, 1))
        add("S S [a11]", pole(s1, 0) * pole(s2, 0))
        add("S S [a12]", 0.5 * (pole(s1, 0) * pole(s2, 1) + pole(s1, 1) * pole(s2, 0)))
        add(
            "S S [b2_pipi]",
            0.5 * (prod(s1, 0) * pole(s2, 1) + prod(s2, 0) * pole(s1, 1)),
        )
        add("S S [c_pipi_pipi]", 0.5 * prod(s1, 0) * prod(s2, 0))
        add(
            "S S [c_pipi_KK]",
            0.5 * (prod(s1, 0) * prod(s2, 1) + prod(s1, 1) * prod(s2, 0)),
        )
        ts = (
            contract_tt(t2(pd, r1, r2), t2(r1, a, b))
            * barrier(2, q2d, R_D)
            * f2_ls(a, b)
        )
        add("f2 S [f_pipi]", ts * prod(s2, 0))
        add("f2 S [f_KK]", ts * prod(s2, 1))
    return out
