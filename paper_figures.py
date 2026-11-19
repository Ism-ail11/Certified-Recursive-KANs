#!/usr/bin/env python3
"""Create vector figures and a rotatable offline 3D view from executed results."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_hex
import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

from controlled_task import teacher_gains
from contextual_gain import contexts
from data import load_split
from fit_baselines import load_model
from gate_study import anchor_hash, csv_write, load_gate
from paper_tables import DATASETS, METHODS, groups, numbers, read_csv


PALETTE = {
    "standard": "#3a444f", "fixed": "#7983a8", "reciprocal": "#aa604a",
    "constant": "#a1884c", "directional": "#3278a4", "signed_full": "#9b6ea2", "signed": "#087e71",
}
SHAPES = {"standard": "o", "fixed": "s", "reciprocal": "v", "constant": "P",
          "directional": "^", "signed_full": "D", "signed": "*"}
LABELS = {**METHODS, "signed_full": "Signed context (full)", "signed": "Signed context (calibrated)"}
INK = "#253448"
LIGHT = "#d7e1e6"


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelcolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.edgecolor": "#8c9ba7", "axes.spines.right": False, "axes.spines.top": False,
        "axes.labelsize": 9.5, "legend.frameon": False, "legend.fontsize": 8.2,
        "pdf.fonttype": 42, "mathtext.fontset": "stix", "figure.facecolor": "white",
    })


def save(fig, out, name, *, tight=True):
    bounds = {"bbox_inches": "tight", "pad_inches": .10} if tight else {}
    fig.savefig(out / f"{name}.pdf", **bounds)
    fig.savefig(out / f"{name}.png", dpi=300, **bounds)
    plt.close(fig)


def tradeoff(real, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    grouped = groups(real, "dataset", "method")
    for ax, dataset in zip(axes, DATASETS):
        signs = -1 if dataset == "bike" else 1
        certified = []
        for method in METHODS:
            rows = grouped[dataset, method]
            ms = numbers(rows, "ms_per_example")
            y = numbers(rows, "rmse" if dataset == "bike" else "accuracy")
            y = np.asarray(y) * (1 if dataset == "bike" else 100)
            if numbers(rows, "certified_fraction"):
                certified.append((mean(ms), mean(y)))
            ax.errorbar(mean(ms), mean(y), xerr=stdev(ms), yerr=stdev(y),
                        marker=SHAPES[method], markersize=10 if method == "signed" else 6.2,
                        color=PALETTE[method], elinewidth=1, capsize=3, markeredgecolor="white",
                        linestyle="none", label=LABELS[method], zorder=3)
        frontier, best = [], -np.inf
        for x, y in sorted(certified):
            if signs * y > best:
                best = signs * y
                frontier.append((x, y))
        if len(frontier) > 1:
            a, b = np.asarray(frontier).T
            ax.step(a, b, where="post", lw=1, ls="--", c="#9bb5b1", zorder=1)
        ax.set_title(DATASETS[dataset])
        ax.set_xlabel("Warm CPU time (ms / example)")
        ax.set_ylabel("RMSE (lower is better)" if dataset == "bike" else "Accuracy (%)")
        ax.grid(c=LIGHT, alpha=.61)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("Predictive quality and certified inference cost", fontsize=15, weight="bold")
    save(fig, out, "01_quality_latency")


def strength_figure(rows, out):
    variants = [("constant", "Learned constant", PALETTE["constant"]),
                ("directional", "Contextual reciprocal", PALETTE["directional"]),
                ("signed", "Contextual signed", PALETTE["signed"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), layout="constrained")
    for col, dataset in enumerate(DATASETS):
        all_data = [r for r in rows if r["dataset"] == dataset]
        for variant, title, color in variants:
            r = [a for a in all_data if a["mode"] == variant]
            by_seed = groups(r, "seed")
            by_strength = defaultdict(list)
            for series in by_seed.values():
                start = next(e for e in series if float(e["strength"]) == 0)
                baseline = float(start["validation_loss"])
                col_key = "rmse" if dataset == "bike" else "accuracy"
                test0 = float(start[col_key])
                for row in series:
                    strength = float(row["strength"])
                    val = 100 * (float(row["validation_loss"]) / baseline - 1)
                    test_delta = (float(row[col_key]) - test0) * (1 if dataset == "bike" else 100)
                    by_strength[strength].append((val, test_delta, row["selected"] == "True"))
                    if variant == "signed" and row["selected"] == "True":
                        axes[0, col].scatter(strength, val, s=23, c="#202d39", alpha=.66, zorder=5)
            a = sorted(by_strength)
            for line, ax in enumerate(axes[:, col]):
                values = [[e[line] for e in by_strength[s]] for s in a]
                y = np.array([mean(e) for e in values])
                spread = np.array([stdev(e) for e in values])
                ax.plot(a, y, marker="o", ms=4, lw=1.8, color=color, label=title)
                ax.fill_between(a, y - spread, y + spread, color=color, lw=0, alpha=.13)
        for ax in axes[:, col]:
            ax.axhline(0, color="#687b89", lw=1, ls="--", zorder=0)
            ax.grid(axis="y", color=LIGHT, alpha=.66)
            ax.set_xlim(-.035, 1.035)
            ax.set_xticks([0, .125, .25, .5, .75, 1.])
        axes[0, col].set_title(DATASETS[dataset])
        axes[0, col].set_ylabel("Validation loss change (%)")
        axes[1, col].set_ylabel("Test RMSE change" if dataset == "bike" else "Test accuracy change (pp)")
        axes[1, col].set_xlabel("Gate strength λ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color="#202d39", marker="o", ls="none", ms=4,
                                          label="Selected signed strength"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    fig.suptitle("Validation-selected gate strength and held-out response", fontsize=15, weight="bold")
    save(fig, out, "02_strength_and_response")


def frozen_bundle(dataset, seed, baseline_root, gain_root):
    path = baseline_root / dataset / "fixed" / f"{seed}.npz"
    model, _ = load_model(path)
    gate, strength = load_gate(gain_root / dataset / str(seed) / "signed.npz", anchor_hash(path))
    return model, gate, strength, load_split(dataset, seed)


def geometry(out, tables, baseline_root, gain_root, synthetic):
    model, gate, strength, d = frozen_bundle("bike", 17, baseline_root, gain_root)
    ids = d.test[np.linspace(0, len(d.test)-1, min(640, len(d.test)), dtype=int)]
    x = d.x[ids]
    h = model.encode(x)
    scale = gate.forward(contexts(x, h), strength).reshape(len(h), -1, 2)
    rates = np.concatenate([model.block.early.rate(), model.block.late.rate()], axis=0)
    gain = np.abs(scale) * rates
    candidates = np.linspace(0, gain.shape[0] * gain.shape[1] - 1, 560, dtype=int)
    sample = gain.reshape(-1, 2)[candidates]
    echo = scale.reshape(-1, 2)[candidates, 1] < 0
    factor = np.sqrt(sample[:, 0] * sample[:, 1])
    records = [dict(A=float(a), B=float(b), q=float(c), reversed=bool(e))
               for (a, b), c, e in zip(sample, factor, echo)]
    csv_write(tables / "figure_gain_points.csv", records)
    w, q = load_gate(synthetic / "switching_signed_17.npz", "FrozenSystem")
    ticks = np.linspace(-2., 2., 45)
    gx, gy = np.meshgrid(ticks, ticks)
    regime = np.column_stack([gx.ravel(), gy.ravel()])
    h0 = np.zeros((len(regime), 12))
    truth = teacher_gains(regime, h0)
    estimated = w.forward(np.column_stack([regime, h0]), q)
    ref = (truth[:, 0] * truth[:, 1]).reshape(gx.shape)
    estimate = (estimated[:, 0] * estimated[:, 1]).reshape(gx.shape)
    np.savez_compressed(tables / "figure_controlled_surface.npz", x=gx, y=gy, truth=ref, prediction=estimate)

    fig = plt.figure(figsize=(12.6, 5.8), layout="constrained")
    fig.get_layout_engine().set(rect=(.055, .025, .925, .963))
    ax = fig.add_subplot(121, projection="3d", proj_type="ortho")
    lo = 0.002
    a = np.linspace(lo, max(1.5, sample[:, 0].max() * 1.06), 60)
    b_max = max(1.15, sample[:, 1].max() * 1.10)
    A, fraction = np.meshgrid(a, np.linspace(0., 1., len(a)))
    B = lo + fraction * (np.minimum(b_max, 1. / A) - lo)
    C = np.sqrt(A * B)
    ax.plot_surface(A, B, C, cmap="PuBuGn", alpha=.24, linewidth=0)
    limit = np.linspace(max(lo, 1. / b_max), min(a[-1], 1. / lo), 90)
    ax.plot(limit, 1 / limit, np.ones_like(limit), color="#64364a", lw=1.8, label="Contraction boundary")
    ax.scatter(sample[~echo, 0], sample[~echo, 1], factor[~echo], s=10, c="#0e8272", alpha=.62,
               depthshade=False, label="Positive second gain")
    if echo.any():
        ax.scatter(sample[echo, 0], sample[echo, 1], factor[echo], s=11, c="#c47043", alpha=.72,
                   depthshade=False, label="Reversed second gain")
    ax.set(xlabel="First-direction slope bound A", ylabel="Second-direction slope bound B", zlabel=r"Loop factor $\sqrt{AB}$")
    ax.set_title("A. Measured gains and the convergence surface")
    ax.view_init(25, -126)
    ax.set_box_aspect((1, 1, .68))
    ax.legend(loc="upper left")
    ax2 = fig.add_subplot(122, projection="3d", proj_type="ortho")
    surf = ax2.plot_surface(gx, gy, estimate, cmap="coolwarm", norm=TwoSlopeNorm(0., -1, 1), alpha=.87,
                           linewidth=0, antialiased=True)
    ax2.plot_wireframe(gx, gy, ref, rstride=7, cstride=7, color="#293b50", linewidth=.65,
                       alpha=.70, label="Teacher")
    ax2.set(xlabel="Regime coordinate 1", ylabel="Regime coordinate 2", zlabel="Signed loop multiplier")
    ax2.set_title("B. Recovered gain in a controlled switching system")
    ax2.view_init(23, -60)
    ax2.set_box_aspect((1, 1, .75))
    ax2.legend(loc="upper left")
    fig.colorbar(surf, ax=ax2, shrink=.52, pad=.08, label="Learned multiplier")
    fig.suptitle("Directional flexibility within a certified loop", fontsize=15, weight="bold")
    save(fig, out, "03_3d_gate_geometry", tight=False)
    scene = dict(name="Measured slope geometry", xyz=[[r["A"], r["B"], r["q"]] for r in records],
                 colors=["#bd6b44" if r["reversed"] else "#087e71" for r in records],
                 axes=["A slope bound", "B slope bound", "Loop factor"], note="Orange marks a reversed second direction.")
    return scene, strength


def embeddings(out, tables, baseline_root, gain_root):
    model, gate, strength, d = frozen_bundle("letters", 17, baseline_root, gain_root)
    rng = np.random.default_rng(9211)
    chosen = np.concatenate([rng.choice(d.test[d.y[d.test] == k], 22, replace=False) for k in range(26)])
    label = d.y[chosen]
    h = model.encode(d.x[chosen])
    before = model.block.forward(h)
    proposed = model.block.forward(h, gate.forward(contexts(d.x[chosen], h), 1.))
    joint = StandardScaler().fit_transform(np.concatenate([before, proposed]))
    computed = []
    for dimension in (2, 3):
        coordinates = TSNE(n_components=dimension, perplexity=32, max_iter=1100,
                           learning_rate="auto", init="pca", random_state=7354, n_jobs=1).fit_transform(joint)
        computed.append((coordinates, float(trustworthiness(joint, coordinates, n_neighbors=12))))
    (xy, trust2), (xyz, trust3) = computed
    colors = [to_hex(plt.get_cmap("tab20")(i)) for i in range(20)] + [
        "#84614f", "#557d47", "#543c78", "#b83167", "#154a5b", "#9c6a1f"]
    paint = np.asarray(colors)[label]
    n = len(chosen)
    pairs = np.concatenate([np.flatnonzero(label == k)[::9] for k in range(26)])
    bounds = np.quantile(xy, (.001, .999), axis=0)
    limits = bounds + np.array([[-3, -3], [3, 3]])
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), layout="constrained")
    for a, points, title in zip(axes[:2], (xy[:n], xy[n:]), ("A. Fixed gains", "B. Signed gate, full strength")):
        a.scatter(*points.T, s=10, c=paint, alpha=.79, linewidth=0, rasterized=True)
        a.set_title(title)
    ax = axes[2]
    lines = np.stack([xy[pairs], xy[pairs + n]], axis=1)
    ax.add_collection(LineCollection(lines, colors=paint[pairs], linewidths=.75, alpha=.56))
    ax.scatter(*xy[pairs].T, s=12, facecolor="white", edgecolors=paint[pairs], linewidths=.6)
    ax.scatter(*xy[pairs + n].T, s=11, c=paint[pairs], linewidths=0)
    ax.set_title("C. Paired displacement")
    for a in axes:
        a.set(xlim=limits[:, 0], ylim=limits[:, 1], xlabel="Shared t-SNE coordinate 1")
        a.set_yticks([])
        a.set_xticks([])
        a.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Shared t-SNE coordinate 2")
    legend = [matplotlib.lines.Line2D([], [], color=colors[i], marker="o", linestyle="none",
                                     label=chr(65 + i), markersize=5) for i in range(26)]
    fig.legend(handles=legend, loc="outside lower center", ncol=13, columnspacing=1, handletextpad=.22)
    fig.suptitle(f"Shared t-SNE of matched Letter equilibria  |  trustworthiness {trust2:.3f}", fontsize=15, weight="bold")
    save(fig, out, "04_tsne_paired_2d")

    fig = plt.figure(figsize=(11.8, 5.5), layout="constrained")
    fig.get_layout_engine().set(rect=(.025, .015, .935, .975))
    c = np.quantile(xyz, (.001, .999), axis=0) + np.array([[-2, -2, -2], [2, 2, 2]])
    for index, tag in enumerate(("A. Fixed gains", "B. Signed gate, full strength")):
        a = fig.add_subplot(1, 2, index + 1, projection="3d", proj_type="ortho")
        place = xyz[index * n: (index + 1) * n]
        a.scatter(*place.T, c=paint, s=9, alpha=.74, depthshade=False, linewidth=0, rasterized=True)
        a.set(title=tag, xlim=c[:, 0], ylim=c[:, 1], zlim=c[:, 2],
              xlabel="Coordinate 1", ylabel="Coordinate 2", zlabel="Coordinate 3")
        a.view_init(21, -62)
        a.set_box_aspect(np.ptp(c, axis=0))
    fig.legend(handles=legend, loc="outside lower center", ncol=13, columnspacing=1, handletextpad=.22)
    fig.suptitle(f"Three-dimensional shared t-SNE  |  trustworthiness {trust3:.3f}", fontsize=15, weight="bold")
    save(fig, out, "05_tsne_paired_3d", tight=False)
    csv_write(tables / "figure_tsne_coordinates.csv", [
        dict(test_index=int(chosen[i % n]), label=chr(65 + label[i % n]), state="fixed" if i < n else "signed_full",
             x2=float(xy[i, 0]), y2=float(xy[i, 1]), x3=float(xyz[i, 0]), y3=float(xyz[i, 1]), z3=float(xyz[i, 2]))
        for i in range(2 * n)
    ])
    scene = dict(name="Letter equilibria t-SNE", xyz=xyz.tolist(),
                 colors=["#a6aeb8"] * n + paint.tolist(),
                 axes=["t-SNE 1", "t-SNE 2", "t-SNE 3"], note="Gray shows fixed gains. Color shows the full-strength gate and letter class.")
    return scene, dict(seed=17, samples=n, representations=2 * n, perplexity=32,
                       trustworthiness_2d=trust2, trustworthiness_3d=trust3,
                       calibrated_strength_for_context=strength)


def paired_effect(tables, out):
    raw = read_csv(tables / "paired_seed_differences.csv")
    confidence = {r["dataset"]: r for r in read_csv(tables / "paired_bootstrap.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        r = sorted((row for row in raw if row["dataset"] == dataset and row["method"] == "signed"),
                   key=lambda a: int(a["seed"]))
        val = np.asarray(numbers(r, "delta"))
        y = np.arange(len(r))
        ax.axvline(0., color="#667989", lw=1, ls="--")
        ax.hlines(y, 0, val, colors=["#087e71" if (p < 0 if dataset == "bike" else p > 0)
                                   else "#a36a52" for p in val], lw=1.5, alpha=.72)
        ax.scatter(val, y, s=55, color=PALETTE["signed"], edgecolors="white", lw=.7, zorder=3)
        c = confidence[dataset]
        ax.plot([float(c["low"]), float(c["high"])], [0, 0], color=INK, linewidth=3, alpha=.50, zorder=1)
        ax.scatter([mean(val)], [len(y)], marker="D", color="#d39043", s=42, zorder=3)
        ax.set_yticks([*y, len(y)], [*(f"Seed {row['seed']}" for row in r), "Mean"])
        ax.invert_yaxis()
        ax.set_xlabel("Paired RMSE change" if dataset == "bike" else "Paired accuracy change (pp)")
        ax.set_title(DATASETS[dataset])
        ax.grid(axis="x", alpha=.55, color=LIGHT)
    fig.suptitle("Gate effect with the same fitted anchor", fontsize=15, weight="bold")
    save(fig, out, "06_paired_seed_effect")


def controlled_figure(synthetic, out):
    rows = read_csv(synthetic / "controlled_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), layout="constrained")
    ax, curves = axes
    modes = ("fixed", "constant", "directional", "signed_full", "signed")
    for i, mode in enumerate(modes):
        values = numbers([r for r in rows if r["scenario"] == "switching" and r["method"] == mode], "rmse")
        ax.scatter(values, np.full(len(values), i), s=23, alpha=.66, c=PALETTE[mode], zorder=2)
        ax.scatter(mean(values), i, marker="D", s=41, c=PALETTE[mode], edgecolors="white", zorder=3)
    ax.set_xscale("log")
    ax.set(yticks=range(len(modes)), yticklabels=[LABELS[m] for m in modes], xlabel="Held-out RMSE (log scale)")
    ax.set_title("A. Recovering sign-changing feedback")
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT, alpha=.65)
    for scenario, style in (("switching", "-"), ("constant", "--")):
        for mode in ("constant", "directional", "signed"):
            raw = json.loads((synthetic / f"{scenario}_{mode}_17.json").read_text())
            hist = raw["history"]
            label = ({"constant": "Constant", "directional": "Reciprocal", "signed": "Signed"}[mode]
                     + (" (switching)" if scenario == "switching" else " (fixed target)"))
            curves.plot([a["epoch"] for a in hist], [a["validation_loss"] for a in hist],
                        color=PALETTE[mode], linestyle=style, lw=1.5, label=label)
    curves.set_yscale("log")
    curves.set_title("B. Full-strength validation loss, seed 17")
    curves.set(xlabel="Gate-fit epoch", ylabel="Validation MSE (log scale)")
    curves.grid(axis="y", color=LIGHT, alpha=.60)
    fig.legend(*curves.get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("Controlled tests of conditional gains", fontsize=15, weight="bold")
    save(fig, out, "07_controlled_learning")


def interact(path, scenes):
    source = json.dumps(scenes, separators=(",", ":"), allow_nan=False)
    html = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Certified KAN 3D figures</title>
<style>
body{font-family:system-ui,sans-serif;color:#233447;margin:0 auto;padding:28px;max-width:960px;background:#fff}
h1{font-size:24px;margin:0 0 14px}p{font-size:14px}fieldset{border:0;padding:0;margin:0 0 12px}
select,button{font:inherit;padding:7px 9px;background:#fff;border:1px solid #a8b7c2;border-radius:5px}
canvas{width:100%;height:540px;display:block;border:1px solid #dce3e8;touch-action:none;cursor:grab}
</style>
<h1>Rotate the measured 3D figures</h1>
<fieldset><label for="scene">View </label><select id="scene"></select> <button id="reset">Reset camera</button></fieldset>
<p id="note"></p><canvas id="plot" aria-label="Rotatable 3D scatter plot"></canvas>
<p>Drag to rotate. Use the mouse wheel to zoom.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.querySelector('#data').textContent), picker=document.querySelector('#scene');
const canvas=document.querySelector('#plot'), ctx=canvas.getContext('2d');
data.forEach((s,i)=>picker.add(new Option(s.name,i)));
let angleX=.49,angleY=-.65,zoom=1,last=null,scene,values,limits;
function load(){
 scene=data[Number(picker.value)]; const columns=[0,1,2].map(i=>scene.xyz.map(p=>p[i]));
 limits=columns.map(col=>[Math.min(...col),Math.max(...col)]);
 values=scene.xyz.map(row=>row.map((v,i)=>2*(v-limits[i][0])/(limits[i][1]-limits[i][0]||1)-1));
 document.querySelector('#note').textContent=scene.note; draw();
}
function rotated([x,y,z]){
 const dx=Math.cos(angleY)*x+Math.sin(angleY)*z, dz=-Math.sin(angleY)*x+Math.cos(angleY)*z;
 return [dx,Math.cos(angleX)*y-Math.sin(angleX)*dz,Math.sin(angleX)*y+Math.cos(angleX)*dz];
}
function draw(){
 const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;
 if(!w||!h||!scene)return;
 canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 const scale=Math.min(w,h)*.255*zoom, view=p=>[w/2+scale*p[0],h/2-scale*p[1]];
 const dots=values.map((p,i)=>[rotated(p),scene.colors[i]]).sort((a,b)=>a[0][2]-b[0][2]);
 const origin=view(rotated([-1,-1,-1]));
 for(let axis=0;axis<3;axis++){
  const tip=[-1,-1,-1];tip[axis]=1.18;const p=view(rotated(tip));
  ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...p);ctx.strokeStyle='#80909e';ctx.lineWidth=1;ctx.stroke();
  ctx.fillStyle='#233447';ctx.font='13px system-ui';ctx.fillText(scene.axes[axis],Math.min(w-100,p[0]+4),p[1]-5);
 }
 for(const [point,color] of dots){const [x,y]=view(point);ctx.beginPath();ctx.arc(x,y,2.8,0,Math.PI*2);ctx.fillStyle=color;ctx.globalAlpha=.73;ctx.fill();}
 ctx.globalAlpha=1;
}
canvas.onpointerdown=e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(last){angleY+=(e.clientX-last[0])*.009;angleX+=(e.clientY-last[1])*.009;last=[e.clientX,e.clientY];draw();}};
canvas.onpointerup=canvas.onpointercancel=()=>last=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
picker.onchange=load;document.querySelector('#reset').onclick=()=>{angleX=.49;angleY=-.65;zoom=1;draw();};
window.onresize=draw;load();
</script>
"""
    path.write_text(html.replace("__DATA__", source), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=Path("results/baselines"))
    parser.add_argument("--gating", type=Path, default=Path("results/gating"))
    parser.add_argument("--controlled", type=Path, default=Path("results/controlled"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    out, tables = args.paper / "figures", args.paper / "tables"
    out.mkdir(parents=True, exist_ok=True)
    setup()
    tradeoff(read_csv(args.gating / "real_per_seed.csv"), out)
    strength_figure(read_csv(args.gating / "strength_curves.csv"), out)
    s, bike_strength = geometry(out, tables, args.baselines, args.gating, args.controlled)
    t, details = embeddings(out, tables, args.baselines, args.gating)
    paired_effect(tables, out)
    controlled_figure(args.controlled, out)
    interact(out / "interactive_3d.html", [s, t])
    (out / "figure_metadata.json").write_text(json.dumps({
        "bike_geometry_seed": 17, "bike_geometry_strength": bike_strength,
        "letter_tsne": details, "png_dpi": 300,
    }, indent=2) + "\n")
    print("Created seven figures in PDF and PNG plus a standalone 3D viewer.")


if __name__ == "__main__":
    main()
