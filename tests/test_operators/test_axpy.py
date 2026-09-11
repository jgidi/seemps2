import numpy as np
from seemps.analysis.derivatives import finite_differences_mpo
from seemps.analysis.mesh import QuantizedInterval
from seemps.analysis.operators import id_mpo
from seemps.analysis.polynomials import mps_from_polynomial
from seemps.operators import MPO, axpy_norm
from seemps.state import MPS, random_mps
from ..tools import SeeMPSTestCase


class TestAxpyNorm(SeeMPSTestCase):
    def test_axpy_norm_matches_dense(self):
        n = 6
        A = MPO(
            [
                self.rng.normal(size=(Dl, 2, 2, Dr))
                + 1j * self.rng.normal(size=(Dl, 2, 2, Dr))
                for Dl, Dr in zip([1, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 1])
            ]
        )
        x = random_mps([2] * n, D=4, complex=True, rng=self.rng)
        y = random_mps([2] * n, D=3, complex=True, rng=self.rng)
        for alpha in (1.0, -1.0, 0.5 - 2j):
            with self.subTest(alpha=alpha):
                exact = np.linalg.norm(
                    A.to_matrix() @ x.to_vector() + alpha * y.to_vector()
                )
                self.assertAlmostEqual(
                    axpy_norm(A, x, y, alpha) / exact, 1.0, places=10
                )

    def test_axpy_norm_resolves_cancellation(self):
        # `A x` and `b` are ~1e6 times larger than the residual `A x - b`.
        n = 8
        interval = QuantizedInterval(0.0, 1.0, qubits=n)
        L = finite_differences_mpo(order=2, filter=3, interval=interval)
        A = (L + (4 / interval.step**2) * id_mpo(n)).join()
        b = mps_from_polynomial(np.asarray([0.5, 1.0]), interval)
        Ad = A.to_matrix()
        x = MPS.from_vector(
            np.linalg.solve(Ad, b.to_vector()) * (1 + 1e-9), [2] * n, normalize=False
        )
        exact = np.linalg.norm(Ad @ x.to_vector() - b.to_vector())
        self.assertAlmostEqual(axpy_norm(A, x, b, -1.0) / exact, 1.0, places=5)
