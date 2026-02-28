# tools/plot_nom_results.py
"""
Plots the TalarAI optimized hydrofoil result.

2/26 ACTION ITEMS:
  ✓ Convergence plot Y-axis clipped — spike at iter 1 no longer dominates
  ✓ Thickness distribution line added to foil plot
  ✓ Multi-condition breakdown shown in stats panel
  ✓ All geometry params shown (thickness, camber, TE gap, limits)
  ✓ Baseline foil is always the actual one used (from nom_summary.json)
    NACA 0012 only appears if the .txt file genuinely can't be found
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _load_best_coords(outputs_dir="outputs"):
    csv_path = os.path.join(outputs_dir, "best_coords_nom.csv")
    npy_path = os.path.join(outputs_dir, "best_coords_nom.npy")
    if os.path.exists(csv_path):
        return np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def _load_summary(outputs_dir="outputs"):
    p = os.path.join(outputs_dir, "nom_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _load_history(outputs_dir="outputs"):
    p = os.path.join(outputs_dir, "nom_history.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _find_baseline_dat(foil_stem: str, script_dir: str) -> str | None:
    foil_stem = os.path.splitext(foil_stem)[0]
    candidates = [
        os.path.join(script_dir, "airfoils_txt", f"{foil_stem}.txt"),
        os.path.join(script_dir, f"{foil_stem}.txt"),
        os.path.join(script_dir, "..", "airfoils_txt", f"{foil_stem}.txt"),
        os.path.join(script_dir, "..", f"{foil_stem}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_baseline_foil(foil_stem: str | None, script_dir: str):
    """
    Load baseline foil from its .txt file (Selig format).
    Returns (x_upper, y_upper, x_lower, y_lower, label).
    Falls back to analytical NACA 0012 only if file genuinely not found.
    """
    dat_path = None
    label = foil_stem or "NACA 0012 (fallback)"

    if foil_stem:
        dat_path = _find_baseline_dat(foil_stem, script_dir)
        if dat_path is None:
            print(f"WARNING: could not find {foil_stem}.txt — falling back to NACA 0012")
            label = "NACA 0012 (fallback)"

    if dat_path is None:
        x  = np.linspace(0, 1, 40)
        yt = 0.12 * (0.2969*np.sqrt(x+1e-9) - 0.126*x
                     - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        return x, yt, x, -yt, label

    data        = np.loadtxt(dat_path, skiprows=1)
    lower_te2le = data[:40]
    upper_le2te = data[40:]
    lower_le2te = lower_te2le[::-1]

    return (upper_le2te[:, 0], upper_le2te[:, 1],
            lower_le2te[:, 0], lower_le2te[:, 1],
            label)


def main(
    coords_path="outputs/best_coords_nom.csv",
    outputs_dir="outputs",
    n_points=40,
    show_baseline=True,
    show_convergence=True,
):
    # ── Load data ────────────────────────────────────────────────────────
    coords  = _load_best_coords(outputs_dir)
    if coords is None:
        raise FileNotFoundError(f"Could not find best_coords_nom.csv in {outputs_dir}")

    summary = _load_summary(outputs_dir)
    history = _load_history(outputs_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    baseline_stem = None
    if summary is not None:
        baseline_stem = summary.get("baseline_foil_filename")

    xu_base, yu_base, xl_base, yl_base, baseline_label = load_baseline_foil(
        baseline_stem, script_dir
    )

    # ── Unpack optimized foil coords ────────────────────────────────────
    upper_le2te = coords[:n_points][::-1]       # x: 0→1
    lower_le2te = coords[n_points:2*n_points]   # x: 0→1

    # ── Geometry calculations ────────────────────────────────────────────
    xg          = np.linspace(0.05, 0.90, 500)
    yu_interp   = np.interp(xg, upper_le2te[:, 0], upper_le2te[:, 1])
    yl_interp   = np.interp(xg, lower_le2te[:, 0], lower_le2te[:, 1])
    thick       = yu_interp - yl_interp
    camber_line = 0.5 * (yu_interp + yl_interp)

    max_thick_val  = float(np.max(thick))
    max_thick_x    = float(xg[np.argmax(thick)])
    max_camber_val = float(np.max(np.abs(camber_line)))
    max_camber_x   = float(xg[np.argmax(np.abs(camber_line))])
    te_gap_val     = float(upper_le2te[-1, 1] - lower_le2te[-1, 1])

    print("\n=== GEOMETRY DIAGNOSTIC ===")
    print(f"Max thickness: {max_thick_val*100:.2f}%c at x={max_thick_x:.2f}c")
    print(f"Max camber:    {max_camber_val*100:.2f}%c at x={max_camber_x:.2f}c")
    print(f"TE gap:        {te_gap_val*100:.3f}%c")
    print(f"Baseline:      {baseline_label}")
    print("===========================\n")

    # ── Summary values ────────────────────────────────────────────────────
    CL        = summary.get("best_CL", 0)          if summary else 0
    CD        = summary.get("best_CD", 0)          if summary else 0
    LD        = CL / CD                             if CD > 0 else 0
    avg_LD    = summary.get("best_avg_LD", None)   if summary else None
    alpha_val = summary.get("alpha", "?")          if summary else "?"
    Re_val    = summary.get("Re",    0)            if summary else 0
    n_ep      = summary.get("tf_n_epochs", "?")    if summary else "?"
    lr_tf     = summary.get("tf_learning_rate", 0.0005) if summary else 0.0005
    n_cond    = len(summary.get("conditions", [])) if summary else 1

    has_conv = show_convergence and history is not None

    # ── Figure layout ─────────────────────────────────────────────────────
    # Row 0: foil plot (left, tall) | convergence (right, tall)
    # Row 1: full-width stats panel
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")

    if has_conv:
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[3.0, 1.2],
            width_ratios=[1.6, 1],
            hspace=0.40, wspace=0.30,
            figure=fig,
        )
        ax_foil  = fig.add_subplot(gs[0, 0])
        ax_conv  = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.2],
                               hspace=0.40, figure=fig)
        ax_foil  = fig.add_subplot(gs[0])
        ax_conv  = None
        ax_stats = fig.add_subplot(gs[1])

    for ax in [ax_foil, ax_stats] + ([ax_conv] if ax_conv else []):
        ax.set_facecolor("#0f0f23")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333366")

    ax_foil.grid(True, color="#1f1f44", linewidth=0.6, linestyle="--")
    if ax_conv:
        ax_conv.grid(True, color="#1f1f44", linewidth=0.6, linestyle="--")

    # ── Foil surfaces ──────────────────────────────────────────────────────
    ax_foil.plot(upper_le2te[:, 0], upper_le2te[:, 1],
                 color="#4fc3f7", lw=2.2, label="TalarAI upper")
    ax_foil.plot(lower_le2te[:, 0], lower_le2te[:, 1],
                 color="#81d4fa", lw=2.2, ls="--", label="TalarAI lower")

    xf  = np.linspace(0, 1, 300)
    yuf = np.interp(xf, upper_le2te[:, 0], upper_le2te[:, 1])
    ylf = np.interp(xf, lower_le2te[:, 0], lower_le2te[:, 1])
    ax_foil.fill_between(xf, ylf, yuf, alpha=0.12, color="#4fc3f7")

    # ── Thickness distribution line on foil plot ──────────────────────────
    # Plot as a scaled line centered at y=0 so it sits visually inside the foil.
    # Half-thickness on each side of the chord line.
    thick_half = thick / 2.0
    ax_foil.plot(xg,  thick_half, color="#a5d6a7", lw=0.9, ls=":",
                 alpha=0.65, label="½ thickness")
    ax_foil.plot(xg, -thick_half, color="#a5d6a7", lw=0.9, ls=":",
                 alpha=0.65)

    # Mark max thickness location
    ax_foil.axvline(max_thick_x, color="#a5d6a7", lw=0.7, ls="--", alpha=0.5)
    ax_foil.annotate(
        f"t_max={max_thick_val*100:.1f}%c\n@ x={max_thick_x:.2f}c",
        xy=(max_thick_x, max_thick_val/2),
        xytext=(max_thick_x + 0.07, max_thick_val/2 + 0.01),
        color="#a5d6a7", fontsize=6.5, fontfamily="monospace",
        arrowprops=dict(arrowstyle="->", color="#a5d6a7", lw=0.7),
    )

    # ── Baseline overlay ──────────────────────────────────────────────────
    if show_baseline:
        ax_foil.plot(xu_base, yu_base, color="#ff7043", lw=1.4, ls=":",
                     alpha=0.85, label=baseline_label)
        ax_foil.plot(xl_base, yl_base, color="#ff7043", lw=1.4, ls=":",
                     alpha=0.85)

    ax_foil.axhline(0, color="#444466", lw=0.8)

    all_y = np.concatenate([upper_le2te[:, 1], lower_le2te[:, 1]])
    y_pad = (all_y.max() - all_y.min()) * 0.35
    ax_foil.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax_foil.set_xlim(-0.02, 1.05)
    ax_foil.set_aspect("equal")

    ax_foil.set_title(f"TalarAI Optimized Hydrofoil  vs  {baseline_label}",
                      color="#c5cae9", fontsize=11, pad=8)
    ax_foil.set_xlabel("x/c", color="#aaaacc", fontsize=9)
    ax_foil.set_ylabel("y/c", color="#aaaacc", fontsize=9)
    ax_foil.legend(fontsize=7, framealpha=0.4, facecolor="#0f0f23",
                   edgecolor="#333366", labelcolor="#c5cae9",
                   loc="upper right", ncol=2)

    # ── Convergence plot — Y-axis clipped ────────────────────────────────
    # ACTION ITEM: lower Y-axis so spike at iter 1 doesn't dominate.
    # We clip to the 95th percentile of all objective values so the yellow
    # "best so far" curve is clearly visible and not squashed at the bottom.
    if ax_conv is not None and history:
        iters = [h["iter"]      for h in history]
        objs  = [h.get("avg_cd_cl", h.get("objective", 0)) for h in history]

        best_curve = []
        cur = float("inf")
        for o in objs:
            cur = min(cur, o)
            best_curve.append(cur)

        # Clip Y axis: show from 0 to 95th percentile so spike doesn't dominate
        y_cap = float(np.percentile([o for o in objs if np.isfinite(o)], 90))
        y_floor = 0.0

        ax_conv.plot(iters, objs,
                     color="#4fc3f7", alpha=0.20, lw=0.5, label="CD/CL each iter")
        ax_conv.plot(iters, best_curve,
                     color="#ffd54f", lw=1.8, label="Best so far")

        ax_conv.set_ylim(y_floor, y_cap * 1.05)
        ax_conv.set_title("NOM Search Convergence",
                          color="#c5cae9", fontsize=11, pad=8)
        ax_conv.set_xlabel("Iteration", color="#aaaacc", fontsize=9)
        ax_conv.set_ylabel("avg CD / CL  (minimize)", color="#aaaacc", fontsize=9)
        ax_conv.legend(fontsize=7.5, framealpha=0.3, facecolor="#0f0f23",
                       edgecolor="#333366", labelcolor="#c5cae9",
                       loc="upper right")

    # ── Stats panel ───────────────────────────────────────────────────────
    # Left column: aerodynamics at design point + avg across conditions
    # Mid column: geometry (all params)
    # Right column: per-condition breakdown

    avg_ld_str = f"{avg_LD:.1f}" if avg_LD else f"{LD:.1f}"

    col_left = (
        f"  AERODYNAMICS\n"
        f"  CL     {CL:.4f}         (design pt)\n"
        f"  CD     {CD:.6f}\n"
        f"  L/D    {LD:.1f}           (design pt)\n"
        f"  avg L/D {avg_ld_str}         ({n_cond} conditions)\n"
        f"  α      {alpha_val}°\n"
        f"  Re     {Re_val:.2e}"
    )

    col_mid = (
        f"  GEOMETRY\n"
        f"  Max thickness  {max_thick_val*100:.2f}%c  @ x={max_thick_x:.2f}c\n"
        f"  Max camber     {max_camber_val*100:.2f}%c  @ x={max_camber_x:.2f}c\n"
        f"  TE gap         {te_gap_val*100:.3f}%c\n"
        f"  Limit t_min    {(summary.get('min_thickness',0) if summary else 0)*100:.2f}%c\n"
        f"  Limit t_max    {(summary.get('max_thickness',0) if summary else 0)*100:.2f}%c\n"
        f"  Limit camber   ≤4%c\n"
        f"  Unified loop   {n_ep} iters  lr={lr_tf}"
    ) if summary else "  GEOMETRY\n  (no summary)"

    # Per-condition breakdown (right column)
    per_cond_list = summary.get("best_per_condition", []) if summary else []
    if per_cond_list:
        cond_lines = "  PER-CONDITION RESULTS\n"
        for c in per_cond_list:
            cl_s = f"{c['CL']:.4f}" if c["CL"] else "N/A "
            ld_s = f"{c['LD']:.1f}"  if c["LD"] else "N/A"
            cond_lines += f"  α={c['alpha']}° Re={c['Re']:.0e}  CL={cl_s}  L/D={ld_s}\n"
        cond_lines += (
            f"\n  NOM SEARCH\n"
            f"  Valid   {summary.get('valid_evals','?')}/{summary.get('n_iters','?')}\n"
            f"  Skipped {summary.get('skipped','?')}\n"
            f"  Base    {summary.get('baseline_foil_filename','?')}"
        )
        col_right = cond_lines
    else:
        col_right = (
            f"  NOM SEARCH\n"
            f"  Valid   {summary.get('valid_evals','?')}/{summary.get('n_iters','?')}\n"
            f"  Skipped {summary.get('skipped','?')}\n"
            f"  Base    {summary.get('baseline_foil_filename','?')}"
        ) if summary else "  NOM\n  (no summary)"

    ax_stats.axis("off")
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#0a0a1a",
                     alpha=0.85, edgecolor="#333366")

    ax_stats.text(0.00, 0.98, col_left,  transform=ax_stats.transAxes,
                  fontsize=7.5, va="top", ha="left", color="#e0e0ff",
                  fontfamily="monospace", bbox=box_style)
    ax_stats.text(0.33, 0.98, col_mid,   transform=ax_stats.transAxes,
                  fontsize=7.5, va="top", ha="left", color="#b2dfdb",
                  fontfamily="monospace", bbox=box_style)
    ax_stats.text(0.66, 0.98, col_right, transform=ax_stats.transAxes,
                  fontsize=7.5, va="top", ha="left", color="#ffe082",
                  fontfamily="monospace", bbox=box_style)

    out_png = os.path.join(outputs_dir, "nom_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent
    outputs      = project_root / "outputs"
    main(
        coords_path=str(outputs / "best_coords_nom.csv"),
        outputs_dir=str(outputs),
    )