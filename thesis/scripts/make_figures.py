"""Publication figures for the diploma article. CPU only.

Okabe–Ito fills, one sans everywhere, 300 dpi PNG + vector PDF/SVG.
No in-figure titles (captions carry the title). SFT = blue, GRPO = vermillion.
Both series are solid: color carries the distinction, labels stay one dark ink.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "thesis" / "data" / "canonical_results.json"
FIG = ROOT / "thesis" / "figures"

WONG = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#7A7A7A",
}
C = {
    "sft": WONG["blue"],
    "grpo": WONG["vermillion"],
    "in_domain": WONG["green"],
    "transfer": WONG["sky"],
    "goal_red": WONG["purple"],
    "map8": WONG["blue"],
    "map16": WONG["orange"],
    "left": WONG["blue"],
    "right": WONG["orange"],
    "forward": WONG["green"],
    "mix": WONG["sky"],
}
HATCH = {key: "" for key in ("sft", "grpo", "in_domain", "transfer", "goal_red", "map8", "map16", "left", "right", "forward", "mix")}

INK = "#1A1A1A"
GRID = "#E6E8EC"
BAR_EDGE = "#FFFFFF"
W_COL = 3.35
W_WIDE = 7.05
H = 2.65


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.axisbelow": True,
            "axes.grid": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg", ".png"):
        path = FIG / f"{stem}{ext}"
        tmp = FIG / f"{stem}.tmp{ext}"
        try:
            fig.savefig(tmp, bbox_inches="tight", pad_inches=0.14)
            try:
                tmp.replace(path)
            except OSError:
                print(f"left {tmp} (could not replace {path})")
                continue
            if ext == ".png":
                print(path)
        except OSError as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            print(f"skip {path}: {exc}")
    plt.close(fig)


def dress(ax: plt.Axes, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_ylabel(ylabel, color=INK)
    ax.tick_params(axis="both", colors=INK, length=2.8, width=0.6)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, color=GRID, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
        ax.spines[spine].set_linewidth(0.6)
    if ylim is not None:
        ax.set_ylim(*ylim)


def legend_above(ax: plt.Axes, ncol: int) -> None:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=False,
        handlelength=0.95,
        handleheight=0.7,
        handletextpad=0.4,
        columnspacing=1.15,
        borderaxespad=0.35,
        labelcolor=INK,
    )


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.0,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
        zorder=6,
    )


def _parts(item: tuple) -> tuple[str, list[float], str, str]:
    if len(item) == 4:
        return item[0], item[1], item[2], item[3]
    return item[0], item[1], item[2], ""


def _label_gap(ax: plt.Axes, points: float = 2.5) -> float:
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    height_pt = bbox.height * 72.0 / ax.figure.dpi
    if height_pt < 1.0:
        return (y1 - y0) * 0.045
    return (y1 - y0) * (points / height_pt)


def _label_y(height: float, err: float, signed: bool, gap: float) -> tuple[float, str, str, str]:
    text = f"{height:+.1f}" if signed else f"{height:.1f}"
    if height < 0:
        return height - err - gap, "top", INK, text
    return height + err + gap, "bottom", INK, text


def annotate_bars(
    ax: plt.Axes,
    container: BarContainer,
    *,
    signed: bool = False,
    ylim: tuple[float, float] | None = None,
    yerr: list[float] | np.ndarray | None = None,
) -> None:
    gap = _label_gap(ax)
    for i, bar in enumerate(container):
        h = float(bar.get_height())
        err = float(yerr[i]) if yerr is not None else 0.0
        y, va, color, text = _label_y(h, err, signed, gap)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            text,
            ha="center",
            va=va,
            fontsize=7.5,
            color=color,
            clip_on=False,
            zorder=5,
        )
    separate_value_labels(ax)


def _padded(bbox, px: float, py: float):
    return bbox.expanded(1, 1).from_extents(bbox.x0 - px, bbox.y0 - py, bbox.x1 + px, bbox.y1 + py)


def separate_value_labels(ax: plt.Axes) -> None:
    texts = [t for t in ax.texts if t.get_text()[:1] in "+-0123456789"]
    if len(texts) < 2:
        return
    fig = ax.figure
    for _ in range(12):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        extents = [_padded(t.get_window_extent(renderer), 5, 1) for t in texts]
        origin = ax.transData.transform((0.0, 0.0))[1]
        unit = ax.transData.transform((0.0, 1.0))[1]
        px_per_data = abs(unit - origin)
        if px_per_data < 1e-6:
            return
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not extents[i].overlaps(extents[j]):
                    continue
                overlap_px = min(extents[i].y1, extents[j].y1) - max(extents[i].y0, extents[j].y0)
                delta = (overlap_px + 1.0) / px_per_data
                yi = texts[i].get_position()[1]
                yj = texts[j].get_position()[1]
                if yi >= 0 and yj >= 0:
                    victim = texts[i] if yi >= yj else texts[j]
                    direction = 1.0
                else:
                    victim = texts[i] if yi <= yj else texts[j]
                    direction = -1.0
                x, y = victim.get_position()
                victim.set_position((x, y + direction * delta))
                moved = True
                break
            if moved:
                break
        if not moved:
            return


def grouped_bars(
    ax: plt.Axes,
    categories: list[str],
    series: list[tuple],
    *,
    yerrs: list[list[float]] | None = None,
    signed: bool = False,
    ylim: tuple[float, float] | None = None,
    ylabel: str,
    legend: bool = True,
    annotate: bool = True,
) -> None:
    n_cat = len(categories)
    n_ser = len(series)
    x = np.arange(n_cat, dtype=float)
    width = min(0.76 / n_ser, 0.34)
    offset = (n_ser - 1) * width / 2.0
    for i, item in enumerate(series):
        label, values, color, hatch = _parts(item)
        err = None if yerrs is None else yerrs[i]
        xs = x - offset + i * width
        del hatch
        bars = ax.bar(
            xs,
            values,
            label=label,
            color=color,
            width=width,
            edgecolor=BAR_EDGE,
            linewidth=0.0,
            zorder=3,
        )
        vis_err = None
        if err is not None:
            err_arr = np.asarray(err, dtype=float)
            vis_err = err_arr
            mask = err_arr > 1e-6
            if np.any(mask):
                ax.errorbar(
                    xs[mask],
                    np.asarray(values, dtype=float)[mask],
                    yerr=err_arr[mask],
                    fmt="none",
                    elinewidth=0.7,
                    capsize=2.0,
                    capthick=0.7,
                    ecolor=INK,
                    zorder=4,
                )
        if annotate:
            annotate_bars(ax, bars, signed=signed, ylim=ylim, yerr=vis_err)
    ax.set_xticks(x, categories)
    ax.set_xlim(-0.55, n_cat - 0.45)
    dress(ax, ylabel, ylim)
    if legend:
        legend_above(ax, ncol=n_ser)


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    lo = max(0.0, (centre - half) * 100.0)
    hi = min(100.0, (centre + half) * 100.0)
    return lo, hi


T_CRIT_95_DF2 = 4.30265  # two-sided 95%, three eval seeds


def mean_t_ci_half(sd: float, n: int = 3, t_crit: float = T_CRIT_95_DF2) -> float:
    if n <= 1:
        return 0.0
    return t_crit * float(sd) / math.sqrt(n)


def fig_sign_flip(c: dict) -> None:
    epochs = [1, 5, 9]
    id_delta = [c["handoff_8x8_in_domain"]["by_epoch"][str(e)]["delta"] for e in epochs]
    id_delta[2] = c["handoff_8x8_in_domain"]["e9_three_train_seed"]["delta"]
    tr = [c["c1_ood"]["transfer"][str(e)]["delta"] for e in epochs]
    red = [c["c1_ood"]["goal_red"][str(e)]["delta"] for e in epochs]
    tr_err = [c["c1_ood"]["transfer"][str(e)]["delta_sd"] for e in epochs]
    red_err = [c["c1_ood"]["goal_red"][str(e)]["delta_sd"] for e in epochs]
    fig, axes = plt.subplots(1, 2, figsize=(W_WIDE, H), sharey=True)
    grouped_bars(
        axes[0],
        ["e1", "e5", "e9*"],
        [
            ("In-domain 8×8", id_delta, C["in_domain"], HATCH["in_domain"]),
            ("Перенос 8→16", tr, C["transfer"], HATCH["transfer"]),
            ("Goal-red", red, C["goal_red"], HATCH["goal_red"]),
        ],
        yerrs=[[0.0, 0.0, 0.0], tr_err, red_err],
        signed=True,
        ylim=(-50, 28),
        ylabel="Δ успех, п.п. (GRPO − SFT)",
        legend=False,
    )
    axes[0].axhline(0, color=INK, linewidth=0.7, zorder=2)
    axes[0].set_xlabel("Эпоха SFT  ·  лестница C1")

    ts = c["c1_ood_e9_trainseed"]
    seeds = ["42", "123", "456"]
    tr_b = [ts["transfer"]["by_train_seed"][s]["delta"] for s in seeds]
    red_b = [ts["goal_red"]["by_train_seed"][s]["delta"] for s in seeds]
    tr_b_err = [ts["transfer"]["by_train_seed"][s]["delta_sd"] for s in seeds]
    red_b_err = [ts["goal_red"]["by_train_seed"][s]["delta_sd"] for s in seeds]
    grouped_bars(
        axes[1],
        ["42", "123", "456"],
        [
            ("Перенос 8→16", tr_b, C["transfer"], HATCH["transfer"]),
            ("Goal-red", red_b, C["goal_red"], HATCH["goal_red"]),
        ],
        yerrs=[tr_b_err, red_b_err],
        signed=True,
        ylim=(-50, 28),
        ylabel="",
        legend=False,
    )
    axes[1].axhline(0, color=INK, linewidth=0.7, zorder=2)
    axes[1].set_xlabel("Train seed GRPO  ·  e9")
    axes[1].tick_params(labelleft=False)
    handles = [
        Patch(facecolor=C["in_domain"], edgecolor="none", label="In-domain"),
        Patch(facecolor=C["transfer"], edgecolor="none", label="Перенос 8→16"),
        Patch(facecolor=C["goal_red"], edgecolor="none", label="Goal-red"),
    ]
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.35,
        handleheight=0.85,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.2,
        fontsize=7.5,
    )
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.subplots_adjust(wspace=0.16)
    save(fig, "fig1_sign_flip")


def fig_prior_flip(c: dict) -> None:
    p8 = c["probes"]["empty8"]
    p16 = c["probes"]["empty16"]
    keys = ["visible", "facing_miss", "far"]
    fig, ax = plt.subplots(figsize=(W_COL, H))
    grouped_bars(
        ax,
        ["visible", "facing_miss", "far"],
        [
            ("Empty 8×8", [100 * p8[k] for k in keys], C["map8"], HATCH["map8"]),
            ("Empty 16×16", [100 * p16[k] for k in keys], C["map16"], HATCH["map16"]),
        ],
        ylim=(0, 72),
        ylabel="Доля стартов, %",
    )
    save(fig, "fig2_start_prior")


def fig_memoryless_bayes(c: dict) -> None:
    b = c["memoryless_bayes"]
    fig, ax = plt.subplots(figsize=(W_WIDE, H))
    spin_err = [mean_t_ci_half(b["8"]["spin_sd"]), mean_t_ci_half(b["16"]["spin_sd"])]
    search_err = [mean_t_ci_half(b["8"]["search_sd"]), mean_t_ci_half(b["16"]["search_sd"])]
    grouped_bars(
        ax,
        ["Empty 8×8  (cap 12)", "Empty 16×16  (cap 40)"],
        [
            ("spin (left при пустом FOV)", [b["8"]["spin"], b["16"]["spin"]], C["map8"], HATCH["map8"]),
            ("search (forward при пустом FOV)", [b["8"]["search"], b["16"]["search"]], C["map16"], HATCH["map16"]),
        ],
        yerrs=[spin_err, search_err],
        ylim=(0, 118),
        ylabel="Успех, %",
    )
    save(fig, "fig8_memoryless_bayes")


def fig_strata(c: dict) -> None:
    g = c["geometry_e9_transfer"]
    keys = ["visible", "facing_miss", "far"]
    fig, ax = plt.subplots(figsize=(W_COL, H))
    grouped_bars(
        ax,
        keys,
        [
            ("SFT", [g[k]["sft"] for k in keys], C["sft"], HATCH["sft"]),
            ("GRPO", [g[k]["grpo"] for k in keys], C["grpo"], HATCH["grpo"]),
        ],
        ylim=(0, 112),
        ylabel="Успех, %",
    )
    save(fig, "fig3_transfer_strata")


def fig_paint_see(c: dict) -> None:
    p = c["paint"]
    s = c["see_far"]
    fig, axes = plt.subplots(1, 2, figsize=(W_WIDE, H), sharey=True)
    grouped_bars(
        axes[0],
        ["G1", "PAINT"],
        [
            ("SFT", [p["g1_sft"], p["paint_sft"]], C["sft"], HATCH["sft"]),
            ("GRPO", [p["g1_grpo"], p["paint_grpo"]], C["grpo"], HATCH["grpo"]),
        ],
        ylim=(0, 112),
        ylabel="Успех, %",
        legend=False,
    )
    grouped_bars(
        axes[1],
        ["G1", "SEE"],
        [
            ("SFT", [s["sft_g1"], s["sft_see"]], C["sft"], HATCH["sft"]),
            ("GRPO", [s["grpo_g1"], s["grpo_see"]], C["grpo"], HATCH["grpo"]),
        ],
        ylim=(0, 112),
        ylabel="",
        legend=False,
    )
    axes[0].set_xlabel("Goal-red")
    axes[1].set_xlabel("Far, перенос 8→16")
    axes[1].tick_params(labelleft=False)
    handles = [
        Patch(facecolor=C["sft"], edgecolor="none", label="SFT"),
        Patch(facecolor=C["grpo"], edgecolor="none", label="GRPO"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handlelength=0.95,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.subplots_adjust(wspace=0.18)
    save(fig, "fig4_paint_see")


def fig_three_mdp(c: dict) -> None:
    f = c["far_honest"]
    fig, ax = plt.subplots(figsize=(W_WIDE, H))
    grouped_bars(
        ax,
        ["C1  8→16", "Native 16×16", "Far-reset\n3 eval-seed"],
        [
            ("SFT", [f["c1_far_sft"], f["native_far_sft"], f["farreset_far_sft_3seed"]], C["sft"], HATCH["sft"]),
            (
                "GRPO",
                [f["c1_far_grpo"], f["native_far_grpo"], f["farreset_far_grpo_3seed"]],
                C["grpo"],
                HATCH["grpo"],
            ),
        ],
        ylim=(0, 100),
        ylabel="Успех на far, %",
    )
    fig.subplots_adjust(bottom=0.16)
    save(fig, "fig5_far_three_mdp")


def fig_color_mix(c: dict) -> None:
    m = c["color_mix"]
    c1_id = c["handoff_8x8_in_domain"]["e9_three_train_seed"]["grpo"]
    fig, ax = plt.subplots(figsize=(W_WIDE, H))
    grouped_bars(
        ax,
        ["In-domain (зелёная цель)", "Goal-red"],
        [
            ("SFT e9", [m["in_domain_sft"], m["goal_red_sft"]], C["sft"], HATCH["sft"]),
            ("C1 GRPO (только green)", [c1_id, m["c1_goal_red_grpo"]], C["grpo"], HATCH["grpo"]),
            ("Color-mix GRPO", [m["in_domain_grpo"], m["goal_red_grpo"]], C["mix"], HATCH["mix"]),
        ],
        ylim=(0, 112),
        ylabel="Успех, %",
    )
    save(fig, "fig6_color_mix")


def maybe_expert_figure() -> None:
    path = ROOT / "thesis" / "data" / "expert_stratum_audit.json"
    if not path.exists():
        print("skip expert figure: audit json missing")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(W_WIDE, 3.05), sharey=True)
    action_colors = [C["left"], C["right"], C["forward"]]
    action_labels = ["left", "right", "forward"]
    for ax, key, title in (
        (axes[0], "8", "Empty 8×8"),
        (axes[1], "16", "Empty 16×16"),
    ):
        block = report[key]["expert_first_action_on_empty_start"]
        labels = ["facing_miss", "far"]
        ns = [block[k]["n"] for k in labels]
        n_max = max(ns)
        small_n = 80
        widths = [0.20 + 0.40 * math.sqrt(n) / math.sqrt(n_max) for n in ns]
        xtick = [f"{k}\nn={n}" for k, n in zip(labels, ns)]
        shares = {
            "left": [100 * block[k]["share"]["left"] for k in labels],
            "right": [100 * block[k]["share"]["right"] for k in labels],
            "forward": [100 * block[k]["share"]["forward"] for k in labels],
        }
        x = np.arange(len(labels), dtype=float)
        bottom = np.zeros(len(labels))
        for lab, color in zip(action_labels, action_colors):
            vals = shares[lab]
            bars = ax.bar(
                x,
                vals,
                bottom=bottom,
                color=color,
                width=widths,
                edgecolor=BAR_EDGE,
                linewidth=0.6,
                label=lab,
                zorder=3,
            )
            bottom = bottom + np.array(vals)
        note = title
        for lab, n in zip(labels, ns):
            if n >= small_n:
                continue
            k_fwd = int(round(block[lab]["share"]["forward"] * n))
            lo, hi = wilson_ci(k_fwd, n)
            note = f"{title}\n{lab}: Wilson P(forward) [{lo:.0f}; {hi:.0f}]"
        ax.set_xticks(x, xtick)
        ax.set_xlim(-0.6, 1.6)
        dress(ax, "Доля первого действия, %" if ax is axes[0] else "", (0, 100))
        ax.set_xlabel(note, labelpad=10, color=INK)
        if ax is axes[1]:
            ax.tick_params(labelleft=False)
    handles = [
        Patch(facecolor=C["left"], edgecolor="none", label="left"),
        Patch(facecolor=C["right"], edgecolor="none", label="right"),
        Patch(facecolor=C["forward"], edgecolor="none", label="forward"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=0.95,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.subplots_adjust(wspace=0.16, bottom=0.22)
    fig.align_xlabels()
    save(fig, "fig7_expert_empty_start")


def fig_lu_schematic() -> None:
    fig, ax = plt.subplots(figsize=(W_WIDE, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.15, 2.05)
    ax.axis("off")
    boxes = [
        (0.12, 0.55, 1.72, 1.15, "Скрытая цель\n(x, y, dir)\nне в RGB", WONG["purple"]),
        (2.08, 0.55, 1.72, 1.15, "Пустой 7×7\nalias:\nmiss ≈ far", WONG["orange"]),
        (4.04, 0.55, 1.72, 1.15, "GRPO G = 16\nодин layout\nP(far) ≈ 3%", WONG["sky"]),
        (6.00, 0.55, 1.72, 1.15, "far + spin\nвсе R = 0\nA = 0", WONG["vermillion"]),
        (7.96, 0.55, 1.72, 1.15, "Градиент\nс facing_miss\nзаостряет поворот", WONG["blue"]),
    ]
    for x, y, w, h, text, facecolor in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=facecolor,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.28,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK, linespacing=1.25)
    for x0, x1 in ((1.84, 2.08), (3.80, 4.04), (5.76, 6.00), (7.72, 7.96)):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.12),
                (x1, 1.12),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color=INK,
            )
        )
    ax.text(
        5.0,
        0.28,
        "Следствие для градиента: far редок по частоте (рис. 10a).\nA = 0 — вторичный канал, если политика уже spin. Знак vs SFT из схемы не следует.",
        ha="center",
        va="center",
        fontsize=8,
        color=INK,
    )
    save(fig, "fig9_lu_filter")


def fig_p_std_zero(c: dict) -> None:
    block = c["p_std_zero_stratum"]
    join = block["join_pooled"]
    cpu = block["cpu_spin_prior"]
    n_total = int(block.get("n_groups", 300))
    fig, axes = plt.subplots(1, 2, figsize=(W_WIDE, H + 0.4))

    order = ["visible", "facing_miss", "far"]
    ns = [join[k]["n"] for k in order]
    colors = [WONG["grey"], WONG["sky"], C["grpo"]]
    x_a = np.arange(3, dtype=float)
    bars_a = axes[0].bar(
        x_a,
        ns,
        color=colors,
        edgecolor=BAR_EDGE,
        linewidth=0.0,
        width=0.62,
        zorder=3,
    )
    for rect, n in zip(bars_a, ns):
        axes[0].text(
            rect.get_x() + rect.get_width() / 2.0,
            n + 4,
            str(n),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    axes[0].set_xticks(x_a, ["visible", "facing_miss", "far"])
    dress(axes[0], f"Число групп из {n_total}", ylim=(0, 190))
    axes[0].set_xlabel("страта старта")
    panel_letter(axes[0], "a")

    cats = ["facing_miss", "far"]
    join_vals = [join[k]["p_all_fail"] for k in cats]
    cpu_vals = [cpu[k]["p_all_fail"] for k in cats]
    miss_j, far_j = join["facing_miss"], join["far"]
    miss_c, far_c = cpu["facing_miss"], cpu["far"]
    grouped_bars(
        axes[1],
        cats,
        [
            (
                "GRPO join",
                join_vals,
                C["grpo"],
                HATCH["grpo"],
            ),
            (
                "CPU spin",
                cpu_vals,
                C["sft"],
                HATCH["sft"],
            ),
        ],
        ylim=(0, 112),
        ylabel="Доля all-fail групп, %",
        legend=False,
        annotate=False,
    )
    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(0.18, 1.0),
        frameon=False,
        handlelength=0.95,
        handleheight=0.7,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelcolor=INK,
        fontsize=7.5,
    )
    n_ser = 2
    width = min(0.76 / n_ser, 0.34)
    offset = (n_ser - 1) * width / 2.0
    x_b = np.arange(2, dtype=float)
    join_lo = np.array([join[k]["ci"][0] for k in cats], dtype=float)
    join_hi = np.array([join[k]["ci"][1] for k in cats], dtype=float)
    cpu_lo = np.array([cpu[k]["ci"][0] for k in cats], dtype=float)
    cpu_hi = np.array([cpu[k]["ci"][1] for k in cats], dtype=float)
    for i, (lo, hi, vals) in enumerate(
        (
            (join_lo, join_hi, np.array(join_vals, dtype=float)),
            (cpu_lo, cpu_hi, np.array(cpu_vals, dtype=float)),
        )
    ):
        xs = x_b - offset + i * width
        axes[1].errorbar(
            xs,
            vals,
            yerr=np.vstack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)]),
            fmt="none",
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            ecolor=INK,
            zorder=4,
        )
        gap = _label_gap(axes[1])
        for x_i, val, top in zip(xs, vals, hi):
            axes[1].text(
                float(x_i),
                float(top) + gap,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK,
                clip_on=False,
                zorder=5,
            )
    mixed = int(far_j.get("n_mixed", far_j["n"] - far_j["n_all_fail"]))
    axes[1].set_xticks(
        x_b,
        [
            f"facing_miss\n{miss_j['n']} / {miss_c['n']}",
            f"far\n{far_j['n']} / {far_c['n']}",
        ],
    )
    axes[1].set_xlabel(
        f"n: GRPO join / CPU spin\nfar join: {far_j['n_all_fail']}/{far_j['n']} all-fail, {mixed}/{far_j['n']} mixed",
        color=INK,
    )
    panel_letter(axes[1], "b")
    fig.subplots_adjust(bottom=0.36, wspace=0.34)
    save(fig, "fig10_p_std_zero_stratum")


def main() -> None:
    style()
    canon = json.loads(DATA.read_text(encoding="utf-8"))
    fig_sign_flip(canon)
    fig_prior_flip(canon)
    fig_memoryless_bayes(canon)
    fig_strata(canon)
    fig_paint_see(canon)
    fig_three_mdp(canon)
    fig_color_mix(canon)
    maybe_expert_figure()
    fig_lu_schematic()
    fig_p_std_zero(canon)


if __name__ == "__main__":
    main()
