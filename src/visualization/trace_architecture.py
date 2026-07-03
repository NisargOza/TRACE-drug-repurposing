"""
TRACE VAE architecture diagram — styled after the CNN architecture illustration.

Blocks: height ∝ layer dimension. Multiple stacked blocks (back→front) convey
depth of feature representation (3 = wide layer, 1 = bottleneck). Colors:
  - blue  = encoder linear layers
  - teal  = μ / log σ² heads
  - gold  = latent z (128-d)
  - orange = decoder linear layers
  - gray  = input / output

Outputs:
  results/figures/trace_vae_architecture.png
  results/figures/trace_vae_architecture.svg
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mc
import numpy as np

ROOT    = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_ENC = "#4A7FCF"
C_MU  = "#3BB08F"
C_Z   = "#F0A500"
C_DEC = "#E07050"
C_IO  = "#8FA8C8"
C_DIS = "#9B59B6"
BG    = "#EAECF0"

def _mix(c, t, f):
    r, g, b = mc.to_rgb(c)
    tr, tg, tb = mc.to_rgb(t)
    return (r + (tr - r) * f, g + (tg - g) * f, b + (tb - b) * f)

def li(c, f=0.32): return _mix(c, "white", f)
def dk(c, f=0.22): return _mix(c, "black", f)

# ── 3-D block primitive ───────────────────────────────────────────────────────
def box3d(ax, x, y, w, h, d=0.22, c=C_ENC, zo=2):
    dx, dy = d, d * 0.48
    ax.add_patch(plt.Polygon(                                            # front
        [(x,y),(x+w,y),(x+w,y+h),(x,y+h)],
        closed=True, fc=c, ec="white", lw=0.8, zorder=zo))
    ax.add_patch(plt.Polygon(                                            # top
        [(x,y+h),(x+w,y+h),(x+w+dx,y+h+dy),(x+dx,y+h+dy)],
        closed=True, fc=li(c), ec="white", lw=0.8, zorder=zo))
    ax.add_patch(plt.Polygon(                                            # right
        [(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)],
        closed=True, fc=dk(c), ec="white", lw=0.8, zorder=zo))

def stacked(ax, cx, cy, w, h, n, c, d=0.22, zo=3):
    """n blocks offset back→front; front face centred at (cx, cy)."""
    for i in range(n - 1, -1, -1):
        ox = i * d * 0.45
        oy = i * d * 0.22
        box3d(ax, cx - w/2 + ox, cy - h/2 + oy, w, h, d=d, c=c, zo=zo + (n - i))

def arr(ax, x0, y0, x1, y1, c="#666", lw=1.5, dashed=False, zo=20):
    ls = (0, (5, 4)) if dashed else "-"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                linestyle=ls, mutation_scale=10,
                                connectionstyle="arc3,rad=0."), zorder=zo)

def lbl(ax, x, y, s, fs=7, c="#222", ha="center", va="center", bold=False, zo=25):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va,
            fontweight="bold" if bold else "normal",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)], zorder=zo)

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 16, 6.8
fig, ax = plt.subplots(figsize=(FW, FH), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
CY = FH / 2.0
BW = 0.62          # block width
MX = 3.80          # max block height (for 978-dim)
D  = 0.22          # 3-D depth offset

def h(dim): return max(0.42, dim / 978 * MX)

# ── X positions (block centres) ───────────────────────────────────────────────
XI  = 1.15          # input
XE1, XE2, XE3 = 2.65, 3.95, 5.20
XM  = 6.35          # μ / σ heads (same x, split y)
XZ  = 7.55          # latent z
XD1, XD2, XD3 = 8.80, 10.10, 11.40
XO  = 12.75         # output

# ── Draw blocks ───────────────────────────────────────────────────────────────

# Input
stacked(ax, XI, CY, BW, h(978), 3, C_IO, d=D)

# Encoder
stacked(ax, XE1, CY, BW, h(978), 3, C_ENC, d=D)
stacked(ax, XE2, CY, BW, h(512), 2, C_ENC, d=D)
stacked(ax, XE3, CY, BW, h(256), 1, C_ENC, d=D)

# μ and log σ² heads (vertically split at same x)
MH = h(128)
MU_Y  = CY + MH * 0.65
SIG_Y = CY - MH * 0.65
stacked(ax, XM, MU_Y,  BW, MH, 1, C_MU, d=D)
stacked(ax, XM, SIG_Y, BW, MH, 1, C_MU, d=D)

# z latent
stacked(ax, XZ, CY, BW, MH, 1, C_Z, d=D)

# Decoder
stacked(ax, XD1, CY, BW, h(256), 1, C_DEC, d=D)
stacked(ax, XD2, CY, BW, h(512), 2, C_DEC, d=D)
stacked(ax, XD3, CY, BW, h(978), 3, C_DEC, d=D)

# Output
stacked(ax, XO, CY, BW, h(978), 3, C_IO, d=D)

# ── Block labels — plain black, no stroke halo ────────────────────────────────
def blk(ax, x, y, s, fs=9):
    ax.text(x, y, s, fontsize=fs, color="black", ha="center", va="center",
            fontweight="bold", zorder=25)

for x, dim in [
    (XI, 978), (XE1, 978), (XE2, 512), (XE3, 256),
    (XD1, 256), (XD2, 512), (XD3, 978), (XO, 978),
]:
    blk(ax, x, CY, str(dim))

# Sub-labels below input and output blocks
_BY = CY - h(978)/2 - 0.18
ax.text(XI, _BY, "L1000 drug signatures\n(978 landmark genes)",
        fontsize=6.5, ha="center", va="top", color="#444", linespacing=1.4, zorder=22)
ax.text(XO, _BY, "Reconstructed\nsignature",
        fontsize=6.5, ha="center", va="top", color="#444", linespacing=1.4, zorder=22)

blk(ax, XM, MU_Y,  "μ",     fs=11)
blk(ax, XM, SIG_Y, "logσ²", fs=8.5)
blk(ax, XZ, CY,    "z",     fs=15)

# ── Arrows ────────────────────────────────────────────────────────────────────
G = 0.08  # gap from block edge to arrow tip

# Input → e1
arr(ax, XI  + BW/2 + G, CY, XE1 - BW/2 - G, CY)
# e1 → e2 → e3
arr(ax, XE1 + BW/2 + G, CY, XE2 - BW/2 - G, CY)
arr(ax, XE2 + BW/2 + G, CY, XE3 - BW/2 - G, CY)
# e3 forks to μ and σ
arr(ax, XE3 + BW/2 + G, CY, XM - BW/2 - G, MU_Y,  c=C_MU)
arr(ax, XE3 + BW/2 + G, CY, XM - BW/2 - G, SIG_Y, c=C_MU)
# μ, σ → z
arr(ax, XM + BW/2 + G, MU_Y,  XZ - BW/2 - G, CY + MH*0.18, c=C_Z)
arr(ax, XM + BW/2 + G, SIG_Y, XZ - BW/2 - G, CY - MH*0.18, c=C_Z)
# z → d1 → d2 → d3 → output
arr(ax, XZ  + BW/2 + G, CY, XD1 - BW/2 - G, CY, c=C_DEC)
arr(ax, XD1 + BW/2 + G, CY, XD2 - BW/2 - G, CY)
arr(ax, XD2 + BW/2 + G, CY, XD3 - BW/2 - G, CY)
arr(ax, XD3 + BW/2 + G, CY, XO  - BW/2 - G, CY)


# z label placed above the block (below interferes with IPF sig)
ax.text(XZ, CY + MH/2 + 0.30, "z = μ + σε", fontsize=14, color="#444",
        ha="center", va="bottom", zorder=25)

# ── Disease signature scoring path ────────────────────────────────────────────
DS_CX, DS_Y_top = XZ, 1.05
box3d(ax, DS_CX - BW/2, DS_Y_top - 0.30, BW, 0.60, d=0.16, c=C_DIS, zo=5)
ax.text(DS_CX, DS_Y_top - 0.01, "IPF sig", fontsize=6.5, color="#111",
        fontweight="bold", ha="center", va="center", zorder=25)

arr(ax, DS_CX, DS_Y_top + 0.30, DS_CX, CY - MH/2 - 0.08,
    c=C_DIS, dashed=True, lw=1.3)


# ── Section headers ───────────────────────────────────────────────────────────
HY = FH - 0.52
for x, label_, bold in [
    (XI,                    "Input",               False),
    ((XE1 + XE3) / 2,      "Encoder",             True),
    ((XM + XZ) / 2,        "Reparameterization",  False),
    ((XD1 + XD3) / 2,      "Decoder",             True),
    (XO,                    "Output",              False),
]:
    ax.text(x, HY, label_, fontsize=10.5, ha="center", color="#1a1a2e",
            fontweight="bold" if bold else "semibold", zorder=22)

# Bracket lines under Encoder and Decoder headers
def brace(x0, x1, y=HY - 0.22, c="#aaa"):
    ax.plot([x0, x0, x1, x1], [y, y+0.09, y+0.09, y], color=c, lw=1.1)

brace(XE1 - BW*0.8, XE3 + BW*0.8 + D)
brace(XD1 - BW*0.8, XD3 + BW*0.8 + 2*D)

plt.tight_layout(pad=0.4)
for ext in ("png", "svg"):
    out = FIG_DIR / f"trace_vae_architecture.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out.relative_to(ROOT)}")
plt.close()
print("Done.")

if __name__ == "__main__":
    pass
