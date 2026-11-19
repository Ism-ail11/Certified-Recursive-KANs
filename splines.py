"""NumPy spline layers, directed numerical bounds, and the training optimizer."""

import numpy as np
from scipy.special import expit, logsumexp


EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny


def up(x):
    return np.nextafter(x, np.inf)


def down(x):
    return np.nextafter(x, -np.inf)


def positive_product(vector, matrix):
    """Outward bound on a nonnegative row-by-matrix product in binary64."""
    n = vector.shape[-1]
    product = vector @ matrix
    return up(up(product * (1 + 8 * (n + 2) * EPS)) + 8 * (n + 2) * TINY)


def loss_gradient(scores, target, task):
    if task == "classification":
        normalized = scores - logsumexp(scores, axis=1, keepdims=True)
        rows = np.arange(len(scores))
        gradient = np.exp(normalized)
        gradient[rows, target] -= 1
        return float(-normalized[rows, target].mean()), gradient / len(target)
    target = target.reshape(scores.shape)
    residual = scores - target
    return float(np.mean(residual ** 2)), 2 * residual / residual.size


class AdamW:
    def __init__(self, parameters, weight_decay=1e-4):
        self.parameters = parameters
        self.first = [np.zeros_like(p) for p, _ in parameters]
        self.second = [np.zeros_like(p) for p, _ in parameters]
        self.time = 0
        self.weight_decay = weight_decay

    def update(self, lr):
        self.time += 1
        norm = np.sqrt(sum(np.sum(g ** 2) for _, g in self.parameters))
        if not np.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        clip = min(1., 5. / max(norm, 1e-30))
        scale = np.sqrt(1 - 0.999 ** self.time)
        alpha = lr * scale / (1 - 0.9 ** self.time)
        for (p, grad), a, b in zip(self.parameters, self.first, self.second):
            g = clip * grad
            a *= .9
            a += .1 * g
            b *= .999
            b += .001 * g ** 2
            p *= 1 - lr * self.weight_decay
            p -= alpha * a / (np.sqrt(b) + 1e-8 * scale)


class CubicSpline:
    def __init__(self, inputs, outputs, rng, gain):
        self.extent = 3.
        self.grid = np.arange(-3, 8) * 1.5 - self.extent
        self.outputs = outputs
        self.coef = rng.normal(0.0, 0.06 / np.sqrt(inputs), (inputs, 7, outputs))
        self.base = rng.normal(0.0, gain / np.sqrt(inputs), (inputs, outputs))
        self.bias = np.zeros(outputs)
        self.coef_grad = np.zeros_like(self.coef)
        self.base_grad = np.zeros_like(self.base)
        self.bias_grad = np.zeros_like(self.bias)

    def parameters(self):
        return [(self.coef, self.coef_grad), (self.base, self.base_grad), (self.bias, self.bias_grad)]

    def basis(self, x, derivative):
        t = np.clip(x, -self.extent, self.extent)[:, :, None]
        knots = self.grid
        value = ((t >= knots[:-1]) & (t < knots[1:])).astype(np.float64)
        for degree in range(1, 4):
            d0 = knots[degree:-1] - knots[:-(degree + 1)]
            d1 = knots[degree + 1:] - knots[1:-degree]
            left = (t - knots[:-(degree + 1)]) / d0
            right = (knots[degree + 1:] - t) / d1
            if degree == 3 and derivative:
                slope = (degree / d0) * value[:, :, :-1] - (degree / d1) * value[:, :, 1:]
                slope *= ((x >= -self.extent) & (x < self.extent))[:, :, None]
            value = left * value[:, :, :-1] + right * value[:, :, 1:]
        return (value, slope) if derivative else value

    def forward(self, x, cache=False):
        if cache:
            values, slopes = self.basis(x, True)
        else:
            values = self.basis(x, False)
        basis = values.reshape(len(x), -1)
        sigmoid = expit(x)
        base = x * sigmoid
        y = basis @ self.coef.reshape(-1, self.outputs) + base @ self.base + self.bias
        if cache:
            return y, (basis, slopes, base, sigmoid * (1 + x * (1 - sigmoid)))
        return y

    def backward(self, dy, cache):
        basis, slopes, base, base_slope = cache
        self.coef_grad += (basis.T @ dy).reshape(self.coef.shape)
        self.base_grad += base.T @ dy
        self.bias_grad += dy.sum(axis=0)
        dz = (dy @ self.coef.reshape(-1, self.outputs).T).reshape(slopes.shape)
        return np.sum(dz * slopes, axis=-1) + (dy @ self.base.T) * base_slope


class CrossSpline:
    def __init__(self, inputs, outputs, rng):
        self.grid = np.linspace(-2., 2., 9)
        self.coef = rng.normal(0, .50 / np.sqrt(inputs), (inputs, 1, outputs)) * self.grid[None, :, None]
        self.coef += rng.normal(0, .025 / np.sqrt(inputs), self.coef.shape)
        self.bias = np.zeros(outputs)
        self.grad = np.zeros_like(self.coef)
        self.bias_grad = np.zeros_like(self.bias)

    def parameters(self):
        return [(self.coef, self.grad), (self.bias, self.bias_grad)]

    def forward(self, x, cache=False):
        segment = np.clip(np.searchsorted(self.grid, x, side="right") - 1, 0, len(self.grid) - 2)
        t = np.clip((np.clip(x, -2., 2.) - self.grid[segment]) * 2, 0., 1.)
        basis = np.zeros((x.size, len(self.grid)))
        row = np.arange(x.size)
        k = segment.ravel()
        basis[row, k], basis[row, k + 1] = 1 - t.ravel(), t.ravel()
        flat = basis.reshape(len(x), -1)
        y = flat @ self.coef.reshape(-1, len(self.bias)) + self.bias
        return (y, (flat, row, k, (x >= -2.) & (x < 2.))) if cache else y

    def backward(self, dy, cache, accumulate):
        basis, row, k, inside = cache
        if accumulate:
            self.grad += (basis.T @ dy).reshape(self.coef.shape)
            self.bias_grad += dy.sum(axis=0)
        g = (dy @ self.coef.reshape(-1, len(self.bias)).T).reshape(len(row), -1)
        return 2 * (g[row, k + 1] - g[row, k]).reshape(inside.shape) * inside

    def propagated_error(self, x_error, injection):
        slopes = up(up(np.abs(np.diff(self.coef, axis=1))) * 2).max(axis=1)
        peak = np.max(np.abs(self.coef), axis=1)
        n = self.coef.shape[0] * self.coef.shape[1] + 4
        amplitude = np.broadcast_to(np.abs(self.bias), injection.shape).copy()
        for source in peak:
            amplitude = up(amplitude + source)
        amplitude = up(amplitude + np.abs(injection))
        rounding = up(up(64 * n * EPS * amplitude) + 16 * n * TINY)
        return up(positive_product(x_error, slopes) + rounding)
