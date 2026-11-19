import os
from types import SimpleNamespace
import unittest

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from scipy.optimize import brentq

from contextual_gain import ContextualGain, contexts
from controlled_task import FrozenSystem, target_from_iteration, teacher_gains
from data import load_split
from paired_kan import CubicModel, PairedBlock
from splines import CubicSpline


class GateTests(unittest.TestCase):
    def test_zero_strength_recovers_unit_gains(self):
        rng = np.random.default_rng(33)
        x = rng.normal(size=(9, 4))
        gate = ContextualGain(4, 3, "signed", 42)
        np.testing.assert_array_equal(gate.forward(x, 0.), np.ones((9, 6)))

    def test_signed_product_remains_bounded(self):
        g = ContextualGain(3, 5, "signed", 17)
        g.bias[5:] = np.linspace(-1e3, 1e3, 5)
        x = np.random.default_rng(341).normal(0, 30, (200, 3))
        for strength in (0., .125, .5, 1.):
            s = g.forward(x, strength)
            self.assertTrue((s[:, 0::2] > 0).all())
            self.assertTrue((np.abs(s[:, 0::2] * s[:, 1::2]) <= 1 + 3e-16).all())
        self.assertTrue((g.forward(x, 1.)[:, 1::2] < 0).any())

    def test_gain_derivatives(self):
        rng = np.random.default_rng(33)
        x, upstream = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
        for kind in ("constant", "directional", "signed"):
            g = ContextualGain(6, 3, kind, 29)
            a = .75
            _, cache = g.forward(x, a, cache=True)
            g.backward(upstream, cache, regularization=0)
            for p, grad in g.parameters():
                i = np.unravel_index(np.abs(grad).argmax(), grad.shape)
                v, delta = p[i], 1e-6
                p[i] = v + delta
                plus = (g.forward(x, a) * upstream).sum()
                p[i] = v - delta
                minus = (g.forward(x, a) * upstream).sum()
                p[i] = v
                self.assertAlmostEqual((plus - minus) / (2 * delta), grad[i], places=6)

    def test_signed_pair_derivatives(self):
        rng = np.random.default_rng(33)
        block = PairedBlock(8, rng)
        b = rng.uniform(-.8, .8, (3, 8))
        scales = rng.uniform(.3, 1.6, b.shape)
        scales[:, 1::2] = -.6 / scales[:, 0::2]
        weight = rng.normal(size=b.shape)
        _, cache = block.forward(b, scales, cache=True)
        db, ds = block.backward(weight, cache, accumulate=False)
        for arg, expected in ((b, db), (scales, ds)):
            for i in np.ndindex(arg.shape):
                v, eps = arg[i], 1e-6
                arg[i] = v + eps
                plus = (block.forward(b, scales) * weight).sum()
                arg[i] = v - eps
                minus = (block.forward(b, scales) * weight).sum()
                arg[i] = v
                self.assertAlmostEqual((plus - minus) / (2 * eps), expected[i], places=6)

    def test_breakpoint_root_matches_independent_brentq_for_sign_flips(self):
        model = FrozenSystem()
        stage = model.block.early
        rng = np.random.default_rng(234)
        h = rng.uniform(-4., 4., (30, 2 * stage.count))
        p = np.tile(np.arange(stage.count), len(h))
        a, b = h.reshape(-1, 2).T
        s = rng.uniform(.4, 3., len(p))
        t = rng.uniform(-.95, .95, len(p)) / s
        calculated = stage._enumerate(p, a, b, s, t)
        reference = []
        for i, u, v, sg, tg in zip(p, a, b, s, t):
            phi, psi = stage.coef[i]
            reference.append(brentq(lambda x: x - v - tg * np.interp(
                u + sg * np.interp(x, stage.grid, phi), stage.grid, psi,
            ), v - abs(tg) * np.max(np.abs(psi)) - 1., v + abs(tg) * np.max(np.abs(psi)) + 1., xtol=5e-15))
        np.testing.assert_allclose(calculated, reference, atol=2e-13, rtol=0)

    def test_cross_stage_and_readout_intervals(self):
        rng = np.random.default_rng(20)
        model = CubicModel(4, 3, 90, "fixed", reference_width=6)
        model.weight[:] = rng.normal(size=model.weight.shape)
        model.block.cross.coef[:] *= 4
        x = rng.normal(size=(7, 4))
        h = model.encode(x)
        gate = ContextualGain(h.shape[1] + x.shape[1], h.shape[1] // 2, "signed", 29)
        gate.bias[gate.pairs:] = -.85
        scales = gate.forward(contexts(x, h), 1.)
        nominal = model.readout.forward(h)
        scores, result, z = model.from_features(h, nominal, scales, certify=True)
        expected_z = target_from_iteration(SimpleNamespace(block=model.block), h, scales)
        self.assertTrue((np.abs(z - expected_z) <= result.error).all())
        expected = nominal + ((expected_z - h) @ model.weight + model.bias)
        lo, hi = model.output_bounds(scores, h, z, result.error, nominal)
        self.assertTrue(((lo <= expected) & (expected <= hi)).all())

    def test_teacher_certifies_a_gain_envelope_that_fails_when_pooled(self):
        m = FrozenSystem()
        h = np.zeros((21, 12))
        regimes = np.column_stack([np.linspace(-2., 2., len(h)), np.linspace(1.2, -1.2, len(h))])
        g = teacher_gains(regimes, h).reshape(len(h), -1, 2)
        rates = np.concatenate([m.block.early.rate(), m.block.late.rate()])
        bounds = rates * np.abs(g)
        self.assertTrue((bounds.prod(axis=2) < 1).all())
        self.assertGreater(bounds.max(axis=0).prod(axis=1).max(), 1.)

    def test_violation_is_rejected(self):
        m = FrozenSystem()
        h = np.ones((3, 12))
        with self.assertRaisesRegex(ValueError, "Uncertified loop"):
            m.block.certify(h, np.full_like(h, 2))


class BaselineAndDataTests(unittest.TestCase):
    def test_cubic_input_gradient(self):
        rng = np.random.default_rng(87)
        m = CubicSpline(3, 2, rng, 1.)
        x, g = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))
        _, c = m.forward(x, True)
        dx = m.backward(g, c)
        for index in np.ndindex(x.shape):
            a, b = x.copy(), x.copy()
            a[index] += 1e-6
            b[index] -= 1e-6
            truth = ((m.forward(a) - m.forward(b)) * g).sum() / 2e-6
            self.assertAlmostEqual(truth, dx[index], places=7)

    def test_letter_group_split(self):
        d = load_split("letters", 17)
        groups = [set(d.groups[t]) for t in (d.train, d.valid, d.test)]
        self.assertEqual(sum(map(len, (d.train, d.valid, d.test))), 20000)
        self.assertTrue(groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2])
                        and groups[1].isdisjoint(groups[2]))

    def test_bike_split_is_ordered(self):
        d = load_split("bike", 17)
        self.assertEqual(d.x.shape, (17379, 17))
        self.assertEqual(len(d.test), 3476)
        self.assertLess(d.train.max(), d.valid.min())
        self.assertLess(d.valid.max(), d.test.min())


if __name__ == "__main__":
    unittest.main()
