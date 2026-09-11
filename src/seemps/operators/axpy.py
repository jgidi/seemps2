from __future__ import annotations

import numpy as np
import scipy.linalg

from ..state import MPS
from ..typing import Weight
from .mpo import MPO


def axpy_norm(A: MPO, x: MPS, y: MPS, alpha: Weight = 1.0) -> float:
    """Norm-2 of `A @ x + alpha * y`, without building or simplifying `A @ x`.

    ----------
    A : MPO
        Operator applied to `x`.
    x : MPS
        State acted upon by `A`.
    y : MPS
        State added to `A @ x`.
    alpha : Weight, default = 1.0
        Weight of `y`, e.g. `-1` for the residual of `A x = y`.

    Returns
    -------
    float
        The norm :math:`\\Vert A x + \\alpha y\\Vert_2`.
    """
    RA = RY = np.ones((1, 1))
    size = x.size
    for i in range(size):
        Ai, Xi, Yi = A[i], x[i], y[i]
        Dl, so, si, Dr = Ai.shape
        xl, _, xr = Xi.shape
        r = RA.shape[0]
        # R(r, Dl, xl) * X(xl, si, xr) * A(Dl, so, si, Dr) -> (r, so, Dr * xr)
        t = (RA.reshape(r * Dl, xl) @ Xi.reshape(xl, si * xr)).reshape(r, Dl, si, xr)
        t = np.tensordot(t, Ai, axes=([1, 2], [0, 2]))
        mA = t.transpose(0, 2, 3, 1).reshape(r, so, Dr * xr)
        mY = (RY @ Yi.reshape(Yi.shape[0], -1)).reshape(r, so, Yi.shape[2])
        if i == 0:
            mY = alpha * mY
        if i == size - 1:
            return float(np.linalg.norm(mA + mY))
        M = np.concatenate([mA, mY], axis=2).reshape(r * so, -1)
        R = scipy.linalg.qr(M, mode="r", overwrite_a=True, check_finite=False)[0]
        R = R[: min(R.shape)]
        k = Dr * xr
        RA, RY = R[:, :k], R[:, k:]


__all__ = ["axpy_norm"]
