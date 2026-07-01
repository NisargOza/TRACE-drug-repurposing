"""
Shared publication-figure style module for the TRACE manuscript figure set.

Consolidates the figure-style skill's helper functions (apply_figure_style,
bar_with_points, end_of_line_labels, focal_palette, goodness_arrow,
panel_crops, panel_letter, set_frame, strip_with_median, two_tier_label)
plus a small IPF/RA-specific colour palette used consistently across all
13 figures in make_figures.py.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.text

# ── Shared semantic palette ─────────────────────────────────────────────────
BLUE    = "#2166ac"   # Net-TRACE / primary method
TEAL    = "#1b7a72"   # VAE-TRACE
RED     = "#d6604d"   # nintedanib / positive control 1
ORANGE  = "#f4a582"   # pirfenidone / positive control 2
GRAY    = "#aaaaaa"   # non-focal / baseline
DGRAY   = "#555555"
GREEN   = "#1a9641"   # significant / validated
PURPLE  = "#7b2d8b"   # RA / dual-disease
GOLD    = "#b8860b"   # CRISPR / genetic evidence
META_GREY = "#888888"


# ===== apply_figure_style =====
def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    """Set matplotlib rcParams for publication-grade output. Call once before plotting.

    frame : 'open' (bottom+left spines, default) | 'boxed' (all four) | 'none'
    font  : sans-serif family name; None = system default sans-serif
    sizes : (base, secondary, tick) — titles/axis-labels, legend/annotation, ticks
    grid  : whether to draw axes.grid (default False)
    """
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob
        import matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


# ===== bar_with_points =====
def bar_with_points(ax, x, ymat, labels, colors, jitter=0.08, show_points=True,
                    errorbar=None, point_alpha=0.5, point_size=8):
    """bar = mean; optionally overlay raw points or draw an interval."""
    means = np.array([np.mean(y) for y in ymat], float)
    err = None
    if errorbar and not show_points:
        if errorbar == "sd":
            err = np.array([np.std(y, ddof=1) if np.asarray(y).size > 1 else 0 for y in ymat])
        elif errorbar == "ci95":
            from scipy.stats import t
            def _hw(y):
                n = np.asarray(y).size
                return t.ppf(0.975, n - 1) * np.std(y, ddof=1) / np.sqrt(n) if n > 1 else 0
            err = np.array([_hw(y) for y in ymat])
    ax.bar(x, means, color=colors, width=0.7, edgecolor="none",
           yerr=err, error_kw={"elinewidth": 0.8, "capsize": 0})
    if show_points:
        for xi, ys in zip(x, ymat):
            ys = np.asarray(ys)
            if ys.ndim and ys.size > 1:
                jit = (np.random.rand(ys.size) - 0.5) * 2 * jitter
                ax.scatter(np.full(ys.size, xi) + jit, ys, s=point_size, color="black",
                           alpha=point_alpha, zorder=3, linewidths=0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    return ax


# ===== end_of_line_labels =====
def end_of_line_labels(ax, xs, ys, labels, colors=None, dx=0.01, fontsize=None):
    """Label each line series at its right end instead of a legend box."""
    if fontsize is None:
        fontsize = plt.rcParams["font.size"]
    if colors is None:
        colors = [None] * len(labels)
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for x, y, lab, c in zip(xs, ys, labels, colors):
        ax.text(x[-1] + dx * span, y[-1], lab, color=c, va="center", ha="left", fontsize=fontsize)


# ===== focal_palette =====
def focal_palette(labels, focal, focal_color, other="muted", base_colors=None):
    """Map labels -> colours with the focal series visually dominant."""
    focal_set = {focal} if isinstance(focal, str) else set(focal)
    n = len(labels)
    if not focal_set & set(labels):
        raise ValueError(f"focal {focal!r} not found in labels")
    if base_colors is None:
        base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#444444"])
    base_colors = [base_colors[i % len(base_colors)] for i in range(n)]
    if other == "grey":
        rest = ["#BCBCBC"] * n
    elif other == "ordinal":
        nf = max(1, n - len(focal_set))
        ramp = [mcolors.to_hex((v, v, v)) for v in
                ([0.55] if nf == 1 else [0.80 - 0.35 * i / (nf - 1) for i in range(nf)])]
        rest, k = [], 0
        for l in labels:
            rest.append(ramp[min(k, nf - 1)]); k += (l not in focal_set)
    else:  # 'muted'
        def mute(c):
            r, g, b = mcolors.to_rgb(c)
            m = (r + g + b) / 3
            return mcolors.to_hex((0.3 * r + 0.7 * m, 0.3 * g + 0.7 * m, 0.3 * b + 0.7 * m))
        rest = [mute(c) for c in base_colors]
    return [focal_color if l in focal_set else rest[i] for i, l in enumerate(labels)]


# ===== goodness_arrow =====
def goodness_arrow(ax, text="higher = better", loc="upper left", axis="y", fontsize=None):
    """Small upright direction-of-goodness cue in the margin."""
    if fontsize is None:
        fontsize = plt.rcParams["legend.fontsize"]
    pos = {"upper left": (0.02, 0.98), "upper right": (0.98, 0.98),
           "lower left": (0.02, 0.02), "lower right": (0.98, 0.02)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    arrow = "\u2191 " if axis == "y" else "\u2192 "
    ax.text(pos[0], pos[1], arrow + text, transform=ax.transAxes,
            fontsize=fontsize, color=META_GREY, ha=ha, va=va)


# ===== panel_crops =====
def panel_crops(fig, dpi=None, pad_px=6, bbox_inches=None, pad_inches=None):
    """Pixel-space crop boxes for each lettered panel in the SAVED PNG."""
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", fig.dpi)
        if dpi == "figure":
            dpi = fig.dpi
    dpi = float(dpi)
    if bbox_inches is None:
        bbox_inches = mpl.rcParams.get("savefig.bbox")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    if bbox_inches == "tight":
        if pad_inches is None:
            pad_inches = mpl.rcParams.get("savefig.pad_inches", 0.1)
        tb = fig.get_tightbbox(r).padded(pad_inches)
        ox_in, oy_in = tb.x0, tb.y0
        W_in, H_in = tb.width, tb.height
    elif isinstance(bbox_inches, mpl.transforms.BboxBase):
        ox_in, oy_in = bbox_inches.x0, bbox_inches.y0
        W_in, H_in = bbox_inches.width, bbox_inches.height
    else:
        ox_in, oy_in = 0.0, 0.0
        W_in, H_in = fig.get_size_inches()
    W_px, H_px = int(round(W_in * dpi)), int(round(H_in * dpi))
    lettered = {}
    for ax in fig.axes:
        for t in ax.findobj(matplotlib.text.Text):
            s = (t.get_text() or "").strip()
            if len(s) == 1 and s.isalpha() and t.get_fontweight() in ("bold", 700):
                lettered[ax] = s
                break
    if not lettered:
        lettered = {ax: str(i) for i, ax in enumerate(fig.axes)}
    out = {}
    for ax, letter in lettered.items():
        bbs = [ax.get_tightbbox(r)]
        ss = ax.get_subplotspec()
        for sib in fig.axes:
            if sib is ax or sib in lettered:
                continue
            ssib = sib.get_subplotspec()
            same_row = ss is None or ssib is None or ss.rowspan == ssib.rowspan
            same_col = ss is None or ssib is None or ss.colspan == ssib.colspan
            if ((ax.get_shared_y_axes().joined(ax, sib) and same_row)
                    or (ax.get_shared_x_axes().joined(ax, sib) and same_col)):
                bbs.append(sib.get_tightbbox(r))
        bb = mpl.transforms.Bbox.union(bbs)
        bx0 = (bb.x0 / fig.dpi - ox_in) * dpi
        bx1 = (bb.x1 / fig.dpi - ox_in) * dpi
        by0 = H_px - (bb.y1 / fig.dpi - oy_in) * dpi
        by1 = H_px - (bb.y0 / fig.dpi - oy_in) * dpi
        out[letter] = (
            max(int(bx0) - pad_px, 0),
            max(int(by0) - pad_px, 0),
            min(int(bx1) + pad_px, W_px),
            min(int(by1) + pad_px, H_px),
        )
    return out


# ===== panel_letter =====
def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    """Bold panel letter outside top-left of axes. case in {'lower','upper'}."""
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


# ===== set_frame =====
def set_frame(ax, style="open"):
    """Set spine visibility on an existing axes. style in {'open','boxed','none'}."""
    show = {"open": (False, False, True, True),
            "boxed": (True, True, True, True),
            "none": (False, False, False, False)}[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


# ===== strip_with_median =====
def strip_with_median(ax, groups, values, colors=None, jitter=0.12):
    """Jittered points + bold horizontal median tick per group."""
    labs = list(groups)
    if colors is None:
        colors = ["#444444"] * len(labs)
    for i, (ys, c) in enumerate(zip(values, colors)):
        ys = np.asarray(ys)
        jit = (np.random.rand(ys.size) - 0.5) * 2 * jitter
        ax.scatter(np.full(ys.size, i) + jit, ys, s=10, color=c, alpha=0.6, linewidths=0, zorder=2)
        m = np.median(ys)
        ax.plot([i - 0.22, i + 0.22], [m, m], color="black", lw=1.6, zorder=3)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    return ax


# ===== two_tier_label =====
def two_tier_label(name, meta):
    """Two-line label string (name / metadata)."""
    return f"{name}\n{meta}"
