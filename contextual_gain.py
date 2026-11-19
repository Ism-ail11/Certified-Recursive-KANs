"""Contextual directional gains for a frozen paired-KAN predictor."""

import numpy as np

from splines import AdamW, loss_gradient


MODES = ("constant", "directional", "signed")
STRENGTHS = (0., .125, .25, .5, .75, 1.)


class ContextualGain:
    """The first direction is positive. The second may damp or reverse."""

    def __init__(self, inputs, pairs, mode, seed):
        self.mode, self.pairs = mode, pairs
        self.center, self.spread = np.zeros(inputs), np.ones(inputs)
        rng = np.random.default_rng(seed)
        n = pairs if mode == "directional" else 2 * pairs
        self.bias = np.zeros(n)
        if mode != "directional":
            self.bias[pairs:] = 1.8
        self.bias_grad = np.zeros(n)
        if mode != "constant":
            self.projection = rng.normal(0., .55 / np.sqrt(inputs), (inputs, 12))
            self.project_bias = np.zeros(12)
            self.weight = rng.normal(0., .015, (12, n))
            self.projection_grad = np.zeros_like(self.projection)
            self.project_bias_grad = np.zeros_like(self.project_bias)
            self.weight_grad = np.zeros_like(self.weight)

    def parameters(self):
        params = [(self.bias, self.bias_grad)]
        if self.mode != "constant":
            params += [(self.projection, self.projection_grad), (self.project_bias, self.project_bias_grad),
                       (self.weight, self.weight_grad)]
        return params

    @property
    def parameter_count(self):
        return sum(p.size for p, _ in self.parameters())

    def snapshot(self):
        return [p.copy() for p, _ in self.parameters()]

    def restore(self, values):
        params = self.parameters()
        if len(params) != len(values):
            raise ValueError("Gain checkpoint has a different number of arrays")
        for (p, _), v in zip(params, values):
            if p.shape != v.shape:
                raise ValueError("Gain checkpoint array shape mismatch")
            p[:] = v

    def set_standardizer(self, context):
        self.center[:] = context.mean(axis=0)
        deviation = context.std(axis=0)
        self.spread[:] = np.where(deviation > 1e-5, deviation, 1.)

    def forward(self, context, strength, cache=False):
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Gate strength must be in [0, 1]")
        if strength == 0 and not cache:
            return np.ones((len(context), 2 * self.pairs))
        if self.mode == "constant":
            hidden, norm = None, None
            raw = np.broadcast_to(self.bias, (len(context), len(self.bias)))
        else:
            norm = np.clip((context - self.center) / self.spread, -6, 6)
            hidden = np.tanh(norm @ self.projection + self.project_bias)
            raw = hidden @ self.weight + self.bias
        direction = np.tanh(raw[:, :self.pairs])
        shift = strength * np.log(3.) * direction
        s, inv = np.exp(shift), np.exp(-shift)
        if self.mode == "directional":
            loop, response = np.ones_like(shift), None
        else:
            response = np.tanh(raw[:, self.pairs:])
            loop = 1 - strength + strength * response
        t = inv * loop
        values = np.stack([s, t], axis=2).reshape(len(context), -1)
        if cache:
            return values, (norm, hidden, direction, response, s, t, inv, loop, shift, strength)
        return values

    def backward(self, gradients, cache, regularization=.001):
        norm, h, direction, response, s, t, inv, loop, shift, strength = cache
        gu, gv = gradients[:, 0::2], gradients[:, 1::2]
        du = gu * s - gv * t + 2 * regularization * shift / shift.size
        du *= strength * np.log(3.) * (1 - direction ** 2)
        if response is None:
            d = du
        else:
            dloop = gv * inv + 2 * regularization * (loop - 1) / loop.size
            d = np.concatenate([du, dloop * strength * (1 - response ** 2)], axis=1)
        self.bias_grad += d.sum(axis=0)
        if h is not None:
            self.weight_grad += h.T @ d
            activation = (d @ self.weight.T) * (1 - h ** 2)
            self.projection_grad += norm.T @ activation
            self.project_bias_grad += activation.sum(axis=0)


def contexts(x, h):
    return np.concatenate([x, h], axis=1)


def scores_at(model, h, nominal, context, gate, strength, batch=256):
    return np.concatenate([
        model.from_features(h[i:i + batch], nominal[i:i + batch],
                            gate.forward(context[i:i + batch], strength))[0]
        for i in range(0, len(h), batch)
    ])


def fit_gain(model, h, nominal, context, target, train, valid, task, mode, seed, epochs):
    gate = ContextualGain(context.shape[1], h.shape[1] // 2, mode, seed)
    gate.set_standardizer(context[train])
    params = gate.parameters()
    optimizer = AdamW(params, weight_decay=5e-4)
    rng = np.random.default_rng(6973 + seed)
    best = np.inf
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train)
        train_loss = 0.
        for offset in range(0, len(order), 256):
            ids = order[offset:offset + 256]
            for _, g in params:
                g.fill(0.)
            scale, cache = gate.forward(context[ids], 1., cache=True)
            z, block_cache = model.block.forward(h[ids], scale, cache=True)
            score = nominal[ids] + ((z - h[ids]) @ model.weight + model.bias)
            loss, derivative = loss_gradient(score, target[ids], task)
            _, dg = model.block.backward(derivative @ model.weight.T, block_cache, accumulate=False)
            gate.backward(dg, cache)
            optimizer.update(.012 * (.35 + .65 * (1 + np.cos(np.pi * (epoch - 1) / epochs)) / 2))
            train_loss += len(ids) * loss
        validation = scores_at(model, h[valid], nominal[valid], context[valid], gate, 1.)
        loss = loss_gradient(validation, target[valid], task)[0]
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite calibration loss")
        history.append({"epoch": epoch, "train_loss": train_loss / len(train), "validation_loss": loss})
        if loss < best - 1e-8:
            best, best_epoch, selected = loss, epoch, gate.snapshot()
        if epoch - best_epoch >= 8:
            break
    gate.restore(selected)
    val_curve = {}
    for alpha in STRENGTHS:
        val = scores_at(model, h[valid], nominal[valid], context[valid], gate, alpha)
        val_curve[alpha] = loss_gradient(val, target[valid], task)[0]
    best_loss = min(val_curve.values())
    slack = 0.002 * val_curve[0.]
    alpha = min(strength for strength, loss in val_curve.items() if loss <= best_loss + slack)
    return gate, alpha, history, val_curve
