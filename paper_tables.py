#!/usr/bin/env python3
"""Build editable LaTeX/CSV tables and a typeset table preview."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak


METHODS = {
    "standard": "Standard cubic KAN", "fixed": "Fixed-gain anchor",
    "reciprocal": "Original reciprocal gate", "constant": "Learned constant gains",
    "directional": "Contextual reciprocal gains", "signed_full": "Signed context, full strength",
    "signed": "Calibrated signed context",
}
DATASETS = {"bike": "Bike Sharing", "letters": "Letter Recognition"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) + sorted(set().union(*(r.keys() for r in rows)) - set(rows[0]))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def groups(rows, *keys):
    bins = defaultdict(list)
    for row in rows:
        bins[tuple(row[key] for key in keys)].append(row)
    return bins


def numbers(rows, name):
    return [float(r[name]) for r in rows if r.get(name, "") != ""]


def summary_text(series, digits=2):
    if not series:
        return "—"
    center = f"{mean(series):.{digits}f}"
    return center if len(series) == 1 else f"{center} ± {stdev(series):.{digits}f}"


def prepare(real, controlled, gating_dir):
    matched = groups(real, "dataset", "method")
    main_rows, numeric, times, ablations, paired = [], [], [], [], []
    lookup = {(r["dataset"], r["seed"], r["method"]): r for r in real}
    for name, label in METHODS.items():
        bikes, letters = [matched[d, name] for d in ("bike", "letters")]
        r = numbers(bikes, "rmse")
        a = [100 * x for x in numbers(letters, "accuracy")]
        f = [100 * x for x in numbers(letters, "macro_f1")]
        c = [100 * x for x in numbers(bikes + letters, "certified_fraction")]
        main_rows.append([label, summary_text(r), summary_text(a), summary_text(f),
                          f"{min(c):.1f}" if c else "—"])
        times.append([label, summary_text(numbers(bikes, "ms_per_example"), 3),
                      summary_text(numbers(letters, "ms_per_example"), 3),
                      f"{int(mean(numbers(bikes, 'parameters'))):,}",
                      f"{int(mean(numbers(letters, 'parameters'))):,}"])
        for d, scores, field, cohort in (("bike", r, "RMSE", bikes), ("letters", a, "Accuracy (%)", letters)):
            cover = numbers(cohort, "certified_fraction")
            ms = numbers(cohort, "ms_per_example")
            numeric.append({
                "dataset": d, "method": name, "metric": field,
                "mean_score": mean(scores), "sd_score": stdev(scores) if len(scores) > 1 else None,
                "mean_cpu_ms": mean(ms), "sd_cpu_ms": stdev(ms) if len(ms) > 1 else None,
                "min_certified_fraction": min(cover) if cover else None,
                "params": int(mean(numbers(cohort, "parameters"))), "seeds": len(cohort),
            })
        if name in {"constant", "directional", "signed_full", "signed"}:
            by_set = []
            for records, key, factor in ((bikes, "rmse", 1.), (letters, "accuracy", 100.)):
                scores = []
                for row in records:
                    ref = lookup[row["dataset"], row["seed"], "fixed"]
                    delta = factor * (float(row[key]) - float(ref[key]))
                    scores.append(delta)
                    paired.append({"dataset": row["dataset"], "seed": int(row["seed"]), "method": name,
                                   "strength": float(row["strength"]), "delta": delta})
                by_set += [summary_text(scores, 3), f"{sum(x > 0 for x in numbers(records, 'strength'))}/{len(records)}"]
            ablations.append([label, *by_set])

    synthetic_rows = []
    m = groups(controlled, "scenario", "method")
    names = {"fixed": "Fixed gains", "constant": "Learned constant gains",
             "directional": "Contextual reciprocal gains", "signed_full": "Signed context, full strength",
             "signed": "Calibrated signed context"}
    for name, label in names.items():
        pos, neg = m["switching", name], m["constant", name]
        sign = [100 * x for x in numbers(pos, "sign_agreement")]
        strengths = numbers(neg, "strength")
        nogain = "0 (roundoff)" if max(numbers(neg, "rmse")) < 1e-10 else summary_text(numbers(neg, "rmse"), 4)
        synthetic_rows.append([label, summary_text(numbers(pos, "rmse"), 4),
                               summary_text(sign), nogain, summary_text(strengths, 3)])
    boot = [bootstrap_delta(gating_dir, d, 17) for d in DATASETS]
    return main_rows, numeric, ablations, times, synthetic_rows, paired, boot


def bootstrap_delta(root, dataset, seed):
    with np.load(root / dataset / str(seed) / "paired_predictions.npz", allow_pickle=False) as data:
        _, codes = np.unique(data["group"], return_inverse=True)
        a, b = data["loss_fixed"], data["loss_signed"]
    size = np.bincount(codes)
    x, y = np.bincount(codes, weights=a), np.bincount(codes, weights=b)
    rng = np.random.default_rng(128972)
    draw = rng.integers(0, len(size), (2400, len(size)))
    count = size[draw].sum(axis=1)
    u, v = x[draw].sum(axis=1) / count, y[draw].sum(axis=1) / count
    if dataset == "bike":
        distribution = np.sqrt(v) - np.sqrt(u)
        point = np.sqrt(b.mean()) - np.sqrt(a.mean())
        unit, metric = "Calendar week", "RMSE"
    else:
        distribution = 100 * (u - v)
        point = 100 * (a.mean() - b.mean())
        unit, metric = "Feature group", "Accuracy points"
    lo, hi = np.quantile(distribution, (.025, .975))
    return dict(dataset=dataset, seed=seed, metric=metric, observed=point, low=lo, high=hi,
                resampling_unit=unit, n_groups=len(size), replications=len(draw))


def tex_escape(text):
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "—": r"\textemdash{}",
                    "±": r"\ensuremath{\pm}", "Δ": r"\ensuremath{\Delta}"}
    return r"\ensuremath{10^{-5}}".join(
        "".join(replacements.get(c, c) for c in part) for part in str(text).split("10^-5")
    )


def tex_table(path, label, title, headers, rows, note):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{" + tex_escape(title) + "}",
             r"\label{tab:" + label + "}",
             r"\begin{tabular}{@{}l" + "r" * (len(headers) - 1) + "@{}}", r"\toprule",
             " & ".join(r"\textbf{" + tex_escape(x) + "}" for x in headers) + r" \\",
             r"\midrule"]
    lines += [" & ".join(tex_escape(x) for x in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
              r"\begin{minipage}{0.98\linewidth}\footnotesize " + tex_escape(note) + r"\end{minipage}",
              r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pdf_tables(path, tables):
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular, bold = "PaperSans", "PaperSansBold"
    pdfmetrics.registerFont(TTFont(regular, str(fonts / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(bold, str(fonts / "DejaVuSans-Bold.ttf")))
    document = SimpleDocTemplate(str(path), pagesize=landscape(letter), leftMargin=43,
                                 rightMargin=43, topMargin=36, bottomMargin=32,
                                 title="Certified gating benchmark tables")
    navy = colors.HexColor("#243347")
    ink = colors.HexColor("#303b45")
    teal = colors.HexColor("#0b6e63")
    rules = colors.HexColor("#cbd5dc")
    title_style = ParagraphStyle("title", fontName=bold, fontSize=23, leading=27, textColor=navy)
    lead = ParagraphStyle("lead", fontName=regular, fontSize=10, leading=14, textColor=ink)
    heading = ParagraphStyle("heading", fontName=bold, fontSize=12, leading=16,
                             textColor=navy, alignment=TA_LEFT)
    note_style = ParagraphStyle("note", fontName=regular, fontSize=8.7, leading=11, textColor=ink)
    body = [Paragraph("Certified gating benchmark", title_style), Spacer(1, 9),
            Paragraph("Publication table preview  |  Larger UCI tasks and controlled feedback systems", lead),
            Spacer(1, 18)]
    for i, (_, title, headers, rows, note) in enumerate(tables):
        if i in (2, 4):
            body.append(PageBreak())
        body.append(Paragraph(f"Table {i+1}. {title}", heading))
        body.append(Spacer(1, 5))
        width = float(document.width)
        first = 197
        relative = [first, *[(width - first) / (len(headers) - 1)] * (len(headers) - 1)]
        matrix = [headers, *rows]
        # Letter-sized columns and compact cells keep all figures on a single line.
        view = Table(matrix, colWidths=relative, hAlign="LEFT", repeatRows=1)
        view.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTNAME", (0, 1), (-1, -1), regular),
            ("FONTSIZE", (0, 0), (-1, -1), 8.6),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5")),
            ("TEXTCOLOR", (0, 0), (-1, -1), ink),
            ("TEXTCOLOR", (0, 0), (-1, 0), navy),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.6),
            ("LINEABOVE", (0, 0), (-1, 0), 1, teal),
            ("LINEBELOW", (0, 0), (-1, 0), .6, rules),
            ("LINEBELOW", (0, -1), (-1, -1), .9, navy),
        ]))
        body.extend([view, Spacer(1, 7), Paragraph(note.replace("10^-5", "10<super>−5</super>"), note_style),
                     Spacer(1, 19)])

    def footer(canvas, doc):
        canvas.setStrokeColor(rules)
        canvas.line(43, 28, landscape(letter)[0] - 43, 28)
        canvas.setFillColor(ink)
        canvas.setFont(regular, 8)
        canvas.drawString(43, 16, "Paired equilibrium KAN  |  benchmark tables")
        canvas.drawRightString(landscape(letter)[0] - 43, 16, str(doc.page))

    document.build(body, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gating", type=Path, default=Path("results/gating"))
    parser.add_argument("--controlled", type=Path, default=Path("results/controlled"))
    parser.add_argument("--output", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    real = read_csv(args.gating / "real_per_seed.csv")
    control = read_csv(args.controlled / "controlled_per_seed.csv")
    main, numeric, ablations, runtime, synthetic, paired, boot = prepare(real, control, args.gating)
    boot_view = [[DATASETS[r["dataset"]], r["metric"], f"{r['observed']:+.3f}", f"[{r['low']:+.3f}, {r['high']:+.3f}]",
                  f"{r['resampling_unit']} (n={r['n_groups']})"] for r in boot]
    notes = [
        "Larger-task means ± sample standard deviations across five initialization or split seeds. The final column gives the lowest test certification rate across both tasks at a hidden-state error tolerance of 10^-5.",
        "Each difference uses the same frozen predictor. Negative RMSE differences and positive accuracy differences favor the gate. Gate on counts seeds with a nonzero selected strength, or a forced full-strength evaluation.",
        "Latencies include feature extraction, solver, and bounds, and use warm, single-thread CPU batches of up to 128. Values show the mean ± standard deviation of per-seed medians from three repeats on up to 1,024 test rows.",
        "Targets use known fixed or sign-changing feedback. These are controlled mechanism tests. Inputs and equations are held fixed across gate variants. Mean ± sample standard deviation over three seeds.",
        "Seed 17 only, with 2,400 paired resamples of weeks or identical-feature groups. Intervals describe sampling variation in this split. Repeated-split variability appears in the other tables.",
    ]
    tables = [
        ("main_results", "Held-out predictive performance and certification",
         ["Model", "Bike RMSE", "Letter acc. (%)", "Letter F1 (%)", "Min. certified (%)"], main, notes[0]),
        ("gating_ablation", "Paired differences against the fixed-gain anchor",
         ["Gate variant", "Bike Δ RMSE", "Bike gate on", "Letter Δ pp", "Letter gate on"], ablations, notes[1]),
        ("cpu_and_parameters", "CPU inference and trainable parameter counts",
         ["Model", "Bike ms / row", "Letters ms / row", "Bike parameters", "Letter parameters"], runtime, notes[2]),
        ("controlled_feedback", "Recovery of known directional feedback",
         ["Gate variant", "Switching RMSE", "Sign agreement (%)", "Fixed RMSE", "Fixed strength"], synthetic, notes[3]),
        ("paired_bootstrap", "Paired bootstrap for the calibrated signed gate",
         ["Benchmark", "Metric", "Difference", "95% interval", "Resampling unit"], boot_view, notes[4]),
    ]
    for key, title, head, rows, note in tables:
        tex_table(args.output / f"{key}.tex", key, title, head, rows, note)
    write_csv(args.output / "main_results_numeric.csv", numeric)
    write_csv(args.output / "paired_seed_differences.csv", paired)
    write_csv(args.output / "paired_bootstrap.csv", boot)
    write_csv(args.output / "controlled_feedback_per_seed.csv", control)
    pdf_tables(args.output / "paper_tables.pdf", tables)
    print(f"Created {len(tables)} LaTeX tables and a PDF preview.")


if __name__ == "__main__":
    main()
