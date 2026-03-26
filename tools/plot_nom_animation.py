"""
tools/plot_nom_animation.py

Interactive history viewer for TalarAI NOM optimization runs.

ACTION ITEMS IMPLEMENTED:
  [1] Video export -- renders all 250 iterations as an mp4 (or gif fallback)
      using matplotlib FuncAnimation. Each frame shows the foil shape
      changing plus the convergence curve with a moving cursor.

  [2] Interactive slider -- opens a matplotlib window with a scrubber so
      you can drag through any iteration forward or back, seeing:
        - Current foil shape vs baseline
        - CL, CD, L/D for that iter
        - Convergence curve with a moving vertical line

USAGE:
    # Interactive slider (default):
    python tools/plot_nom_animation.py

    # Export video (mp4 if ffmpeg installed, gif otherwise):
    python tools/plot_nom_animation.py --video

    # Export video at specific fps:
    python tools/plot_nom_animation.py --video --fps 30

    # Point at a different outputs folder:
    python tools/plot_nom_animation.py --outputs path/to/outputs

REQUIREMENTS:
    pip install matplotlib numpy
    (for mp4: ffmpeg must be on PATH -- https://ffmpeg.org/download.html)
    (gif fallback uses Pillow: pip install Pillow)

DATA REQUIREMENTS:
    outputs/nom_history.json  -- written by nom_driver.py (must include coords)
    outputs/nom_summary.json  -- written by nom_driver.py (for baseline info)
    outputs/best_coords_nom.csv -- for best foil overlay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------
BG_DARK   = "#1a1a2e"
BG_PANEL  = "#0f0f23"
GRID_COL  = "#1f1f44"
SPINE_COL = "#333366"
TEXT_COL  = "#c5cae9"
TICK_COL  = "#aaaacc"
FOIL_U    = "#4fc3f7"
FOIL_L    = "#81d4fa"
BASE_COL  = "#ff7043"
CONV_RAW  = "#4fc3f7"
CONV_BEST = "#ffd54f"
STAR_COL  = "#ffd54f"
N_POINTS  = 40


def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TICK_COL, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--")


def _load_data(outputs_dir: Path):
    hist_path = outputs_dir / "nom_history.json"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"nom_history.json not found in {outputs_dir}.\n"
            "Run nom_driver.py first (it must be the updated version that "
            "saves coords in nom_history.json).")

    with open(hist_path) as f:
        history = json.load(f)

    # Filter entries that have coords
    history = [h for h in history if h.get("coords") is not None]
    if not history:
        raise ValueError(
            "nom_history.json exists but has no entries with coords.\n"
            "Re-run nom_driver.py with the updated version.")

    summary_path = outputs_dir / "nom_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    best_path = outputs_dir / "best_coords_nom.csv"
    best_coords = (np.loadtxt(best_path, delimiter=",", skiprows=1)
                   if best_path.exists() else None)

    # Baseline coords from summary (stored by updated nom_driver)
    bl_coords_list = summary.get("baseline_coords")
    baseline_coords = np.array(bl_coords_list) if bl_coords_list else None

    return history, summary, best_coords, baseline_coords


def _coords_from_entry(entry: dict) -> np.ndarray:
    """Extract (80, 2) coords from a history entry."""
    return np.array(entry["coords"], dtype=np.float32)


def _build_best_curve(history: list) -> list[float]:
    """L/D best-so-far at each logged iteration."""
    best_curve = []
    best_ld = 0.0
    for h in history:
        CL = h.get("CL", 0)
        CD = h.get("CD", 1e-9)
        ld = CL / CD if CD > 0 else 0.0
        best_ld = max(best_ld, ld)
        best_curve.append(best_ld)
    return best_curve


# ---------------------------------------------------------------------------
# SHARED DRAWING HELPERS
# ---------------------------------------------------------------------------

def _draw_foil(ax, coords, color_u, color_l, label_u=None, label_l=None,
               lw=2.0, ls_l="--", alpha_fill=0.08):
    upper = coords[:N_POINTS][::-1]   # TE->LE to LE->TE
    lower = coords[N_POINTS:]
    lu, = ax.plot(upper[:, 0], upper[:, 1], color=color_u, lw=lw,
                  label=label_u)
    ll, = ax.plot(lower[:, 0], lower[:, 1], color=color_l, lw=lw, ls=ls_l,
                  label=label_l)
    xf  = np.linspace(0, 1, 200)
    yuf = np.interp(xf, upper[:, 0], upper[:, 1])
    ylf = np.interp(xf, lower[:, 0], lower[:, 1])
    ax.fill_between(xf, ylf, yuf, alpha=alpha_fill, color=color_u)
    return lu, ll


def _foil_axes_style(ax, title=""):
    _style_ax(ax)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c", color=TICK_COL, fontsize=8)
    ax.set_ylabel("y/c", color=TICK_COL, fontsize=8)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=9)


# ===========================================================================
# ACTION ITEM 2:  INTERACTIVE SLIDER VIEWER
# ===========================================================================

def interactive_viewer(outputs_dir: Path):
    """
    Opens a matplotlib window with a slider to scrub through all iterations.

    Layout:
      Top-left:  Foil shape (current iter vs baseline vs best)
      Top-right: Convergence (L/D best-so-far) with vertical cursor
      Bottom:    Slider + stats text + Prev / Next buttons
    """
    history, summary, best_coords, baseline_coords = _load_data(outputs_dir)
    best_curve = _build_best_curve(history)
    n_iters    = len(history)
    iters      = [h["iter"] for h in history]

    alpha  = summary.get("alpha", "?")
    Re     = summary.get("Re",    "?")
    bl_LD  = summary.get("baseline_LD", 0.0)
    bl_name = summary.get("baseline_foil_filename", "baseline")

    # ---- Figure ----
    fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
    fig.suptitle("TalarAI NOM  —  Iteration History Viewer",
                 color=TEXT_COL, fontsize=12, y=0.98)

    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[3.5, 0.25, 0.6],
        hspace=0.50, wspace=0.35,
        figure=fig,
        top=0.93, bottom=0.05, left=0.06, right=0.97,
    )
    ax_foil = fig.add_subplot(gs[0, 0])
    ax_conv = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[1, :])   # slider lives here
    ax_stat = fig.add_subplot(gs[2, :])   # stats text lives here

    for ax in [ax_foil, ax_conv]:
        _style_ax(ax)
    ax_ctrl.axis("off")
    ax_stat.axis("off")

    # ---- Convergence background (drawn once) ----
    _style_ax(ax_conv)
    ld_vals = [h["CL"] / h["CD"] if h.get("CD", 0) > 0 else 0.0 for h in history]
    ax_conv.plot(iters, ld_vals, color=CONV_RAW, alpha=0.25, lw=0.8,
                 label="L/D each iter")
    ax_conv.plot(iters, best_curve, color=CONV_BEST, lw=2.0,
                 label="Best so far")
    ax_conv.axhline(bl_LD, color=BASE_COL, lw=1.0, ls=":", alpha=0.6,
                    label=f"baseline ({bl_LD:.1f})")
    ax_conv.set_xlabel("Iteration", color=TICK_COL, fontsize=8)
    ax_conv.set_ylabel("L/D", color=TICK_COL, fontsize=8)
    ax_conv.set_title("Convergence — drag slider to explore",
                      color=TEXT_COL, fontsize=9)
    ax_conv.legend(fontsize=6.5, framealpha=0.3, facecolor=BG_PANEL,
                   edgecolor=SPINE_COL, labelcolor=TEXT_COL)

    # Vertical cursor on convergence plot
    [vline] = ax_conv.plot([iters[0], iters[0]],
                            ax_conv.get_ylim() or [0, 100],
                            color="#ffffff", lw=1.0, alpha=0.5, ls="--")

    # ---- Initial foil draw (mutable line objects) ----
    _foil_axes_style(ax_foil)

    # Baseline
    if baseline_coords is not None:
        _draw_foil(ax_foil, baseline_coords, BASE_COL, BASE_COL,
                   label_u=bl_name, lw=1.2, ls_l=":", alpha_fill=0.0)

    # Best foil (static)
    if best_coords is not None:
        _draw_foil(ax_foil, best_coords, "#a5d6a7", "#a5d6a7",
                   label_u="best", lw=1.0, ls_l=":", alpha_fill=0.04)

    # Current iter foil (will be redrawn)
    coords0 = _coords_from_entry(history[0])
    upper0  = coords0[:N_POINTS][::-1]
    lower0  = coords0[N_POINTS:]
    line_upper, = ax_foil.plot(upper0[:, 0], upper0[:, 1],
                               color=FOIL_U, lw=2.2, label="current upper")
    line_lower, = ax_foil.plot(lower0[:, 0], lower0[:, 1],
                               color=FOIL_L, lw=2.2, ls="--", label="current lower")
    fill_ref = [None]   # mutable container for fill_between

    ax_foil.legend(fontsize=6.5, framealpha=0.3, facecolor=BG_PANEL,
                   edgecolor=SPINE_COL, labelcolor=TEXT_COL, loc="upper right")

    # ---- Stats text ----
    stat_text = ax_stat.text(
        0.01, 0.95, "", transform=ax_stat.transAxes,
        fontsize=7.5, va="top", ha="left", color=TEXT_COL,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  alpha=0.85, edgecolor=SPINE_COL))

    # ---- Slider ----
    # Axes for slider: [left, bottom, width, height] in figure coords
    ax_slider = fig.add_axes([0.10, 0.165, 0.75, 0.025])
    ax_slider.set_facecolor(BG_PANEL)
    slider = Slider(
        ax=ax_slider,
        label="Iter",
        valmin=0,
        valmax=n_iters - 1,
        valinit=0,
        valstep=1,
        color="#4fc3f7",
    )
    slider.label.set_color(TEXT_COL)
    slider.valtext.set_color(CONV_BEST)

    # ---- Prev / Next buttons ----
    ax_prev = fig.add_axes([0.87, 0.155, 0.05, 0.04])
    ax_next = fig.add_axes([0.93, 0.155, 0.05, 0.04])
    btn_prev = Button(ax_prev, "◄", color=BG_PANEL, hovercolor=SPINE_COL)
    btn_next = Button(ax_next, "►", color=BG_PANEL, hovercolor=SPINE_COL)
    btn_prev.label.set_color(TEXT_COL)
    btn_next.label.set_color(TEXT_COL)

    # ---- Update function ----
    def update(idx: int):
        idx = int(np.clip(idx, 0, n_iters - 1))
        entry  = history[idx]
        coords = _coords_from_entry(entry)
        upper  = coords[:N_POINTS][::-1]
        lower  = coords[N_POINTS:]
        CL     = entry.get("CL", 0)
        CD     = entry.get("CD", 1e-9)
        ld     = CL / CD if CD > 0 else 0.0
        best_ld_here = best_curve[idx]
        it     = entry["iter"]
        imp    = " ★ BEST" if (idx > 0 and best_curve[idx] > best_curve[idx - 1]) else ""

        # Redraw current foil
        line_upper.set_xdata(upper[:, 0]); line_upper.set_ydata(upper[:, 1])
        line_lower.set_xdata(lower[:, 0]); line_lower.set_ydata(lower[:, 1])

        # Redo fill_between (must remove old one first)
        if fill_ref[0] is not None:
            fill_ref[0].remove()
        xf  = np.linspace(0, 1, 200)
        yuf = np.interp(xf, upper[:, 0], upper[:, 1])
        ylf = np.interp(xf, lower[:, 0], lower[:, 1])
        fill_ref[0] = ax_foil.fill_between(xf, ylf, yuf, alpha=0.10,
                                            color=FOIL_U)

        ax_foil.set_title(
            f"Iter {it}/{iters[-1]}  |  CL={CL:.4f}  CD={CD:.6f}  L/D={ld:.1f}{imp}",
            color=STAR_COL if imp else TEXT_COL, fontsize=8)

        # Move vertical cursor
        vline.set_xdata([it, it])
        ylims = ax_conv.get_ylim()
        vline.set_ydata(ylims)

        # Stats text
        pct = (ld - bl_LD) / bl_LD * 100 if bl_LD > 0 else 0.0
        stat_text.set_text(
            f"  Iter {it:>4}/{iters[-1]}   |   "
            f"CL={CL:.4f}   CD={CD:.6f}   L/D={ld:.1f}   "
            f"Best L/D so far={best_ld_here:.1f}   "
            f"vs baseline: {pct:+.1f}%   |   "
            f"alpha={alpha}°  Re={Re:.0e}"
        )

        fig.canvas.draw_idle()

    # Wire slider
    slider.on_changed(lambda val: update(int(val)))

    def on_prev(_):
        v = max(0, int(slider.val) - 1)
        slider.set_val(v)

    def on_next(_):
        v = min(n_iters - 1, int(slider.val) + 1)
        slider.set_val(v)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Draw initial frame
    update(0)

    print(f"Interactive viewer ready — {n_iters} iterations loaded.")
    print("Use the slider or ◄ ► buttons to scrub through the optimization.")
    plt.show()


# ===========================================================================
# ACTION ITEM 1:  VIDEO EXPORT
# ===========================================================================

def export_video(outputs_dir: Path, fps: int = 30, out_name: str = "nom_optimization"):
    """
    Renders all iterations to an mp4 (or gif fallback) using FuncAnimation.

    Each frame shows:
      - Left:  foil shape (current vs baseline vs best so far)
      - Right: convergence curve with a moving cursor dot

    Output: outputs/nom_optimization.mp4  (or .gif if ffmpeg not found)
    """
    history, summary, best_coords, baseline_coords = _load_data(outputs_dir)
    best_curve = _build_best_curve(history)
    n_frames   = len(history)
    iters      = [h["iter"] for h in history]
    ld_vals    = [h["CL"] / h["CD"] if h.get("CD", 0) > 0 else 0.0
                  for h in history]

    bl_LD   = summary.get("baseline_LD", 0.0)
    bl_name = summary.get("baseline_foil_filename", "baseline")
    alpha   = summary.get("alpha", "?")
    Re      = summary.get("Re",    "?")

    print(f"Rendering {n_frames} frames at {fps} fps...")

    matplotlib.use("Agg")  # non-interactive, fast
    fig = plt.figure(figsize=(12, 5), facecolor=BG_DARK)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                             left=0.06, right=0.97, top=0.88, bottom=0.12)
    ax_foil = fig.add_subplot(gs[0])
    ax_conv = fig.add_subplot(gs[1])

    for ax in [ax_foil, ax_conv]:
        _style_ax(ax)

    # ---- Static convergence background ----
    ax_conv.plot(iters, ld_vals, color=CONV_RAW, alpha=0.25, lw=0.8,
                 label="L/D each iter")
    ax_conv.plot(iters, best_curve, color=CONV_BEST, lw=2.0,
                 label="Best so far")
    ax_conv.axhline(bl_LD, color=BASE_COL, lw=1.0, ls=":", alpha=0.6,
                    label=f"baseline ({bl_LD:.1f})")
    ax_conv.set_xlabel("Iteration", color=TICK_COL, fontsize=8)
    ax_conv.set_ylabel("L/D", color=TICK_COL, fontsize=8)
    ax_conv.legend(fontsize=6.5, framealpha=0.3, facecolor=BG_PANEL,
                   edgecolor=SPINE_COL, labelcolor=TEXT_COL)

    # Cursor dot on conv plot
    cursor_dot, = ax_conv.plot([], [], "o", color="#ffffff", ms=6, zorder=5)

    # ---- Static foil elements ----
    _foil_axes_style(ax_foil)
    if baseline_coords is not None:
        _draw_foil(ax_foil, baseline_coords, BASE_COL, BASE_COL,
                   label_u=bl_name, lw=1.2, ls_l=":", alpha_fill=0.0)
    if best_coords is not None:
        _draw_foil(ax_foil, best_coords, "#a5d6a7", "#a5d6a7",
                   label_u="best", lw=1.0, ls_l=":", alpha_fill=0.04)

    # Mutable current-foil lines
    coords0 = _coords_from_entry(history[0])
    upper0  = coords0[:N_POINTS][::-1]
    lower0  = coords0[N_POINTS:]
    line_upper, = ax_foil.plot(upper0[:, 0], upper0[:, 1],
                               color=FOIL_U, lw=2.2, label="current upper")
    line_lower, = ax_foil.plot(lower0[:, 0], lower0[:, 1],
                               color=FOIL_L, lw=2.2, ls="--", label="current lower")
    fill_ref = [None]

    ax_foil.legend(fontsize=6.5, framealpha=0.3, facecolor=BG_PANEL,
                   edgecolor=SPINE_COL, labelcolor=TEXT_COL, loc="upper right")

    title = fig.suptitle("", color=TEXT_COL, fontsize=10, y=0.96)

    def animate(frame_idx):
        entry  = history[frame_idx]
        coords = _coords_from_entry(entry)
        upper  = coords[:N_POINTS][::-1]
        lower  = coords[N_POINTS:]
        CL     = entry.get("CL", 0)
        CD     = entry.get("CD", 1e-9)
        ld     = CL / CD if CD > 0 else 0.0
        it     = entry["iter"]
        best_here = best_curve[frame_idx]
        imp    = " ★" if (frame_idx > 0 and best_curve[frame_idx] > best_curve[frame_idx - 1]) else ""

        line_upper.set_xdata(upper[:, 0]); line_upper.set_ydata(upper[:, 1])
        line_lower.set_xdata(lower[:, 0]); line_lower.set_ydata(lower[:, 1])

        if fill_ref[0] is not None:
            fill_ref[0].remove()
        xf  = np.linspace(0, 1, 200)
        yuf = np.interp(xf, upper[:, 0], upper[:, 1])
        ylf = np.interp(xf, lower[:, 0], lower[:, 1])
        fill_ref[0] = ax_foil.fill_between(xf, ylf, yuf, alpha=0.10, color=FOIL_U)

        ax_foil.set_title(
            f"Iter {it}/{iters[-1]}  CL={CL:.4f}  CD={CD:.6f}  L/D={ld:.1f}{imp}",
            color=STAR_COL if imp else TEXT_COL, fontsize=8)

        cursor_dot.set_data([it], [ld_vals[frame_idx]])

        pct = (ld - bl_LD) / bl_LD * 100 if bl_LD > 0 else 0.0
        title.set_text(
            f"TalarAI NOM  |  iter {it}/{iters[-1]}  "
            f"L/D={ld:.1f}  best={best_here:.1f}  vs baseline: {pct:+.1f}%  "
            f"|  alpha={alpha}°  Re={Re:.0e}")

        return line_upper, line_lower, cursor_dot, title

    anim = animation.FuncAnimation(
        fig, animate,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
    )

    # Try mp4 first, fall back to gif
    mp4_path = outputs_dir / f"{out_name}.mp4"
    gif_path = outputs_dir / f"{out_name}.gif"

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        metadata={"title": "TalarAI NOM"})
        anim.save(str(mp4_path), writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": BG_DARK})
        print(f"Video saved: {mp4_path}")
    except Exception as e:
        print(f"  ffmpeg not available ({e})")
        print(f"  Falling back to GIF (may be large)...")
        try:
            writer_gif = animation.PillowWriter(fps=min(fps, 15))
            anim.save(str(gif_path), writer=writer_gif, dpi=100,
                      savefig_kwargs={"facecolor": BG_DARK})
            print(f"GIF saved: {gif_path}")
            print(f"  Tip: install ffmpeg for better quality mp4 output.")
        except Exception as e2:
            print(f"  GIF export also failed: {e2}")
            print(f"  Install Pillow:  pip install Pillow")
            print(f"  Install ffmpeg:  https://ffmpeg.org/download.html")

    plt.close(fig)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TalarAI NOM animation viewer / video exporter")
    parser.add_argument("--outputs", default="",
                        help="Path to outputs folder (default: auto-detect)")
    parser.add_argument("--video",   action="store_true",
                        help="Export video instead of interactive viewer")
    parser.add_argument("--fps",     type=int, default=30,
                        help="Frames per second for video export (default 30)")
    parser.add_argument("--name",    default="nom_optimization",
                        help="Output filename stem (default: nom_optimization)")
    args = parser.parse_args()

    # Resolve outputs dir
    if args.outputs:
        outputs_dir = Path(args.outputs)
    else:
        # Walk up from this script's location looking for outputs/
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / "outputs",
            script_dir.parent / "outputs",
            Path("outputs"),
        ]
        outputs_dir = next((p for p in candidates if p.exists()), Path("outputs"))

    print(f"Using outputs dir: {outputs_dir.resolve()}")

    if args.video:
        export_video(outputs_dir, fps=args.fps, out_name=args.name)
    else:
        interactive_viewer(outputs_dir)


if __name__ == "__main__":
    main()