import numpy as np
from .tools import TestCase
from seemps.analysis.mesh import *


class TestIntervals(TestCase):
    def test_regular_closed_interval_constructor(self):
        I = RegularClosedInterval(0, 1, 3)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 3)
        self.assertEqual(len(I), 3)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 0.5)
        self.assertEqual(I[2], 1.0)

        self.assertEqual([I[0], I[1], I[2]], list(I))
        self.assertEqual([I[0], I[1], I[2]], [x for x in I])

    def test_regular_half_open_interval_constructor(self):
        I = RegularHalfOpenInterval(0, 1, 2)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 0.5)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])

    def test_regular_chebyshev_zeros_interval_constructor(self):
        I = ChebyshevZerosInterval(-1, 1, 2)
        self.assertEqual(I.start, -1)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertAlmostEqual(I[0], -np.sqrt(2.0) / 2.0)
        self.assertAlmostEqual(I[1], np.sqrt(2.0) / 2.0)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])


class TestMesh(TestCase):
    def test_mesh_constructor_1d(self):
        I0 = RegularClosedInterval(0, 1, 3)
        m = Mesh([I0])
        self.assertEqual(len(m.intervals), 1)
        self.assertEqual(m.intervals[0], I0)
        self.assertEqual(m.dimension, 1)
        self.assertEqual(m.dimensions, (3,))

    def test_mesh_1d_sequence_access(self):
        m = Mesh([RegularClosedInterval(0, 1, 3)])
        self.assertEqual(m[[0]], 0.0)
        self.assertEqual(m[[1]], 0.5)
        self.assertEqual(m[[2]], 1.0)
        with self.assertRaises(IndexError):
            m[[-1]]
        with self.assertRaises(IndexError):
            m[[3]]
        with self.assertRaises(IndexError):
            m[2, 0]

    def test_mesh_1d_integer_access(self):
        m = Mesh([RegularClosedInterval(0, 1, 3)])
        self.assertEqual(m[0], 0.0)
        self.assertEqual(m[1], 0.5)
        self.assertEqual(m[2], 1.0)

    def test_mesh_1d_checks_bounds(self):
        m = Mesh([RegularClosedInterval(0, 1, 3)])
        with self.assertRaises(IndexError):
            m[-1]
        with self.assertRaises(IndexError):
            m[3]
        with self.assertRaises(IndexError):
            m[3, 0]

    def test_mesh_1d_integer_access(self):
        m = Mesh([RegularClosedInterval(0, 1, 3)])
        self.assertEqual(m[0], 0.0)
        self.assertEqual(m[1], 0.5)
        self.assertEqual(m[2], 1.0)

    def test_mesh_1d_to_tensor(self):
        m = Mesh([RegularClosedInterval(0, 1, 3)])
        self.assertSimilar(m.to_tensor(), [(0,), (0.5,), (1.0,)])

    def test_mesh_constructor_2d(self):
        I0 = RegularClosedInterval(0, 1, 3)
        I1 = RegularHalfOpenInterval(0, 1, 2)
        m = Mesh([I0, I1])
        self.assertEqual(len(m.intervals), 2)
        self.assertEqual(m.intervals[0], I0)
        self.assertEqual(m.intervals[1], I1)
        self.assertEqual(m.dimension, 2)
        self.assertEqual(m.dimensions, (3, 2))

    def test_mesh_2d_tuple_access(self):
        m = Mesh([RegularClosedInterval(0, 1, 3), RegularHalfOpenInterval(0, 1, 2)])
        self.assertSimilar(m[0, 0], [0.0, 0.0])
        self.assertSimilar(m[1, 0], [0.5, 0.0])
        self.assertSimilar(m[0, 1], [0.0, 0.5])
        self.assertSimilar(m[1, 1], [0.5, 0.5])
        self.assertSimilar(m[2, 0], [1.0, 0.0])
        self.assertSimilar(m[2, 1], [1.0, 0.5])

    def test_mesh_2d_checks_bounds(self):
        m = Mesh([RegularClosedInterval(0, 1, 3), RegularHalfOpenInterval(0, 1, 2)])
        with self.assertRaises(IndexError):
            m[3, 0]
        with self.assertRaises(IndexError):
            m[-1, 0]
        with self.assertRaises(IndexError):
            m[0, -1]
        with self.assertRaises(IndexError):
            m[0]

    def test_mesh_2d_to_tensor(self):
        m = Mesh([RegularClosedInterval(0, 1, 3), RegularHalfOpenInterval(0, 1, 2)])
        self.assertSimilar(
            m.to_tensor(),
            [
                [(0.0, 0.0), (0.0, 0.5)],
                [(0.5, 0.0), (0.5, 0.5)],
                [(1.0, 0.0), (1.0, 0.5)],
            ],
        )
