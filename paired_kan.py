"""Direct spline equilibria with fixed, reciprocal, or externally supplied gains."""

from dataclasses import dataclass

import numpy as np

from splines import EPS, TINY, CrossSpline, CubicSpline, down, positive_product, up


@dataclass
class Certificate:
    state: np.ndarray
    error: np.ndarray
    q: np.ndarray
    fallbacks: np.ndarray


class PairStage:
    def __init__(self, count, rng, original_gate=False):
        self.count = count
        self.grid = np.linspace(-2., 2., 9)
        rates = rng.normal(0, .20, (count, 2, 1))
        rates[:, 0] = rng.uniform(.2, .6, (count, 1))
        self.coef = rates * self.grid + rng.normal(0, .025, (count, 2, 9))
        self.grad = np.zeros_like(self.coef)
        self.local = np.column_stack([np.full(count, .65), np.zeros(count)]) if original_gate else None
        self.local_grad = np.zeros_like(self.local) if original_gate else None
        self.project()

    def parameters(self):
        pairs = [(self.coef, self.grad)]
        if self.local is not None:
            pairs.append((self.local, self.local_grad))
        return pairs

    def project(self):
        left = self.coef[:, 0]
        inc = np.clip(np.diff(left, axis=1), .015, 1.25)
        monotone = np.column_stack([np.zeros(len(left)), np.cumsum(inc, axis=1)])
        left[:] = monotone + (left.mean(axis=1) - monotone.mean(axis=1))[:, None]
        gains = (2 * np.abs(np.diff(self.coef, axis=2)).max(axis=2)).prod(axis=1)
        self.coef[:, 1] *= np.minimum(1., .72 / np.maximum(gains, 1e-30))[:, None]

    def rate(self):
        diff = np.diff(self.coef, axis=2)
        if not np.isfinite(self.coef).all() or not (diff[:, 0] > 0).all():
            raise ValueError("The first spline of each pair must increase over its knot span")
        return np.where(diff == 0, 0.0, up(2 * up(np.abs(diff)))).max(axis=2)

    def gains(self, external):
        if self.local is None:
            return np.ones_like(external), None
        contrast = 0.5 * (external[:, 0::2] - external[:, 1::2])
        y = np.tanh(contrast * self.local[:, 0] + self.local[:, 1])
        a = np.exp(np.log(3.) * y)
        return np.stack([a, 1 / a], axis=2).reshape(external.shape), (contrast, y)

    def _spline(self, channel, pairs, x, derivative=False):
        ids = pairs.reshape((-1,) + (1,) * (x.ndim - 1))
        raw = np.searchsorted(self.grid, x, side="right") - 1
        k = np.clip(raw, 0, 7)
        t = np.clip(2 * (np.clip(x, -2., 2.) - self.grid[k]), 0., 1.)
        a, b = self.coef[ids, channel, k], self.coef[ids, channel, k + 1]
        y = a + t * (b - a)
        if derivative:
            return y, np.where((raw >= 0) & (raw < 8), 2 * (b - a), 0.0), k, t
        return y

    def _enumerate(self, p, a, b, s, t):
        knots = self.coef[p, 0]
        target = (self.grid[None] - a[:, None]) / s[:, None]
        k = np.clip((target[:, :, None] >= knots[:, None]).sum(axis=2) - 1, 0, 7)
        lo = np.take_along_axis(knots, k, axis=1)
        hi = np.take_along_axis(knots, k + 1, axis=1)
        preimage = self.grid[k] + .5 * np.clip((target - lo) / (hi - lo), 0., 1.)
        x = np.concatenate([np.broadcast_to(self.grid, preimage.shape), preimage], axis=1)
        f = b[:, None] + t[:, None] * self._spline(1, p, a[:, None] + s[:, None] * self._spline(0, p, x))
        residual = x - f
        left = np.where(residual <= 0., x, -np.inf)
        right = np.where(residual >= 0., x, np.inf)
        rows = np.arange(len(a))
        li, ri = left.argmax(axis=1), right.argmin(axis=1)
        l, r = left[rows, li], right[rows, ri]
        y = np.where(np.isfinite(l), f[:, 8], f[:, 0])
        bracketed = np.isfinite(l) & np.isfinite(r)
        y[bracketed] = l[bracketed]
        j = rows[bracketed & (l < r)]
        u, v = residual[j, li[j]], residual[j, ri[j]]
        y[j] = l[j] + (r[j] - l[j]) * (-u / (v - u))
        return y

    def roots(self, bias, gains):
        p = np.tile(np.arange(self.count), len(bias))
        a, b = bias.reshape(-1, 2).T
        s, t = gains.reshape(-1, 2).T
        phi, dphi, _, _ = self._spline(0, p, b, True)
        psi, dpsi, _, _ = self._spline(1, p, a + s * phi, True)
        u = b + t * psi / (1 - (s * dphi) * (t * dpsi))
        v, residual = np.empty_like(u), np.empty_like(u)
        active = np.arange(len(b))
        for k in range(2):
            if not len(active):
                break
            phi, dphi, _, _ = self._spline(0, p[active], u[active], True)
            aa = a[active] + s[active] * phi
            psi, dpsi, _, _ = self._spline(1, p[active], aa, True)
            raw = (b[active] + t[active] * psi) - u[active]
            done = np.abs(raw) <= 16 * EPS * (1 + np.abs(u[active]) + np.abs(b[active]))
            v[active[done]], residual[active[done]] = aa[done], np.abs(raw[done])
            if k == 0:
                u[active[~done]] += (raw / (1 - (s[active] * dphi) * (t[active] * dpsi)))[~done]
            active = active[~done]
        if len(active):
            p0, a0, b0, s0, t0 = p[active], a[active], b[active], s[active], t[active]
            u[active] = self._enumerate(p0, a0, b0, s0, t0)
            v[active] = a0 + s0 * self._spline(0, p0, u[active])
            residual[active] = np.abs((b0 + t0 * self._spline(1, p0, v[active])) - u[active])
        state = np.column_stack([v, u]).reshape(bias.shape)
        raw = np.column_stack([np.zeros_like(residual), residual]).reshape(bias.shape)
        return state, raw, np.bincount(active // self.count, minlength=len(bias))

    def errors(self, bias, state, raw, scales, bias_error=0.):
        scales = np.abs(scales)
        magnitudes = scales.reshape(len(bias), self.count, 2)
        A = up(magnitudes * self.rate()[None])
        gain = up(A[:, :, 0] * A[:, :, 1])
        if not (gain < 1).all():
            raise ValueError("Uncertified loop gain")
        peaks = up(np.abs(self.coef).max(axis=2).reshape(-1) * scales)
        amplitude = up(up(np.abs(bias) + np.abs(state)) + peaks)
        slack = up(up(112 * EPS * amplitude) + 16 * TINY)
        r = up(up(raw + slack) + bias_error)
        ra, rb = r[:, 0::2], r[:, 1::2]
        denominator = down(1 - gain)
        ea = up(up(ra + up(A[:, :, 0] * rb)) / denominator)
        eb = up(up(rb + up(A[:, :, 1] * ra)) / denominator)
        return np.stack([ea, eb], axis=2).reshape(bias.shape), up(np.sqrt(gain)).max(axis=1)

    def backward(self, gradient, z, scales, local_cache, accumulate):
        pairs = np.tile(np.arange(self.count), len(z))
        a, b = z.reshape(-1, 2).T
        s, t = scales.reshape(-1, 2).T
        phi, dphi, k, fk = self._spline(0, pairs, b, True)
        psi, dpsi, l, fl = self._spline(1, pairs, a, True)
        gu, gv = gradient.reshape(-1, 2).T
        f, g = s * dphi, t * dpsi
        denom = 1 - f * g
        da, db = (gu + g * gv) / denom, (gv + f * gu) / denom
        ds, dt = da * phi, db * psi
        if accumulate:
            np.add.at(self.grad, (pairs, 0, k), da * s * (1 - fk))
            np.add.at(self.grad, (pairs, 0, k + 1), da * s * fk)
            np.add.at(self.grad, (pairs, 1, l), db * t * (1 - fl))
            np.add.at(self.grad, (pairs, 1, l + 1), db * t * fl)
        from_input = np.zeros_like(z)
        if local_cache is not None:
            contrast, tanh = local_cache
            q = (ds * s - dt * t).reshape(-1, self.count) * np.log(3.) * (1 - tanh ** 2)
            if accumulate:
                self.local_grad[:, 0] += (q * contrast).sum(axis=0)
                self.local_grad[:, 1] += q.sum(axis=0)
            tmp = .5 * q * self.local[:, 0]
            from_input[:, 0::2], from_input[:, 1::2] = tmp, -tmp
        return np.column_stack([da, db]).reshape(z.shape), from_input, np.column_stack([ds, dt]).reshape(z.shape)


class PairedBlock:
    def __init__(self, width, rng, original_gate=False):
        self.width = width
        self.split = 2 * (width // 4)
        self.early = PairStage(self.split // 2, rng, original_gate)
        self.late = PairStage((width - self.split) // 2, rng, original_gate)
        self.cross = CrossSpline(self.split, width - self.split, rng)

    def parameters(self):
        return self.early.parameters() + self.late.parameters() + self.cross.parameters()

    def project(self):
        self.early.project()
        self.late.project()

    def _forward(self, injection, scales=None, cache=False, certify=False):
        a, b = injection[:, :self.split], injection[:, self.split:]
        if scales is None:
            s, u = self.early.gains(a)
            t, v = self.late.gains(b)
        else:
            s, t = scales[:, :self.split], scales[:, self.split:]
            u = v = None
        if certify and (not np.isfinite(injection).all() or not np.isfinite(s).all() or not np.isfinite(t).all()
                        or (s[:, 0::2] <= 0).any() or (t[:, 0::2] <= 0).any()):
            raise ValueError("Expected finite injection and positive first directions")
        first, r1, f1 = self.early.roots(a, s)
        if cache:
            cross, track = self.cross.forward(first, True)
        else:
            cross = self.cross.forward(first)
        bias = b + cross
        second, r2, f2 = self.late.roots(bias, t)
        out = np.concatenate([first, second], axis=1)
        if cache:
            return out, (first, second, s, t, u, v, track)
        if not certify:
            return out
        ea, qa = self.early.errors(a, first, r1, s)
        error_in = self.cross.propagated_error(ea, b)
        eb, qb = self.late.errors(bias, second, r2, t, error_in)
        error = np.concatenate([ea, eb], axis=1)
        if not np.isfinite(error).all() or not np.isfinite(out).all():
            raise FloatingPointError("Nonfinite equilibrium or error bound")
        return Certificate(out, error, np.maximum(qa, qb), f1 + f2)

    def forward(self, injection, scales=None, cache=False):
        return self._forward(injection, scales, cache)

    def certify(self, injection, scales=None):
        return self._forward(injection, scales, certify=True)

    def backward(self, gradient, cache, accumulate=True):
        first, second, s, t, u, v, cross = cache
        a, b = gradient[:, :self.split], gradient[:, self.split:]
        db, extra2, dt = self.late.backward(b, second, t, v, accumulate)
        da, extra1, ds = self.early.backward(a + self.cross.backward(db, cross, accumulate),
                                             first, s, u, accumulate)
        return np.concatenate([da + extra1, db + extra2], axis=1), np.concatenate([ds, dt], axis=1)


class CubicModel:
    """Six-layer cubic KAN with an optional affine correction from paired equilibria."""

    def __init__(self, inputs, outputs, seed, method, reference_width=24):
        self.method = method
        self.inputs, self.outputs = inputs, outputs
        rng = np.random.default_rng(seed)
        n = reference_width
        reference = 9 * (inputs * n + 8 * n * n + outputs * n) + 7 * n + outputs

        def gap(width):
            basic = 8 * (inputs * width + 4 * width ** 2 + width * outputs) + 5 * width + outputs
            if method == "standard":
                return abs(basic - reference)
            left = 2 * (width // 4)
            branch = 9 * (width + left * (width - left)) + width - left
            return abs(basic + branch + width * outputs + outputs - reference)

        choices = range(4, 257) if method == "standard" else range(4, 257, 2)
        self.width = min(choices, key=gap)
        dims = [inputs, *([self.width] * 5), outputs]
        self.layers = [CubicSpline(a, b, rng, 1.45 if i < 5 else .85)
                       for i, (a, b) in enumerate(zip(dims, dims[1:]))]
        self.readout = self.layers[-1]
        self.block = None if method == "standard" else PairedBlock(self.width, rng, method == "reciprocal")
        if self.block is not None:
            self.weight = np.zeros((self.width, outputs))
            self.bias = np.zeros(outputs)
            self.weight_grad = np.zeros_like(self.weight)
            self.bias_grad = np.zeros_like(self.bias)

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        if self.block is not None:
            params += self.block.parameters() + [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]
        return params

    @property
    def parameter_count(self):
        return sum(p.size for p, _ in self.parameters())

    def snapshot(self):
        return [p.copy() for p, _ in self.parameters()]

    def restore(self, state):
        params = self.parameters()
        if len(params) != len(state):
            raise ValueError("Checkpoint parameter count differs")
        for (p, _), value in zip(params, state):
            if p.shape != value.shape:
                raise ValueError("Checkpoint parameter shape differs")
            p[:] = value

    def project(self):
        if self.block is not None:
            self.block.project()

    def encode(self, x, cache=False):
        tracks = []
        for layer in self.layers[:-1]:
            if cache:
                x, buf = layer.forward(x, True)
                tracks.append(buf)
            else:
                x = layer.forward(x)
        return (x, tracks) if cache else x

    def from_features(self, h, base=None, scales=None, certify=False):
        if base is None:
            base = self.readout.forward(h)
        if self.block is None:
            return base, None, None
        solved = self.block.certify(h, scales) if certify else None
        z = solved.state if certify else self.block.forward(h, scales)
        scores = base + ((z - h) @ self.weight + self.bias)
        return scores, solved, z

    def predict(self, x):
        return self.from_features(self.encode(x), certify=True)

    def output_bounds(self, scores, h, z, error, base):
        weight = np.abs(self.weight)
        radius = positive_product(error, weight)
        amplitude = positive_product(up(np.abs(z) + np.abs(h)), weight)
        amplitude = up(up(amplitude + np.abs(base)) + np.abs(self.bias))
        n = self.width + 5
        allowance = up(up(64 * n * EPS * amplitude) + 16 * n * TINY)
        radius = up(radius + allowance)
        return down(scores - radius), up(scores + radius)

    def training_forward(self, x):
        h, trunk = self.encode(x, True)
        base, out = self.readout.forward(h, True)
        if self.block is None:
            return base, (trunk, out, None, None)
        z, block = self.block.forward(h, cache=True)
        delta = z - h
        return base + (delta @ self.weight + self.bias), (trunk, out, block, delta)

    def backward(self, gradient, cache):
        trunk, out, block, delta = cache
        h_gradient = self.readout.backward(gradient, out)
        if self.block is not None:
            self.weight_grad += delta.T @ gradient
            self.bias_grad += gradient.sum(axis=0)
            correction = gradient @ self.weight.T
            b, _ = self.block.backward(correction, block)
            h_gradient += b - correction
        for layer, state in zip(reversed(self.layers[:-1]), reversed(trunk)):
            h_gradient = layer.backward(h_gradient, state)
