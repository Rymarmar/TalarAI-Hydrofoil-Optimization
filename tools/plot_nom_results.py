# plot_nom_results.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _load_best_latent(outputs_dir="outputs"):
    csv_path = os.path.join(outputs_dir, "best_latent_nom.csv")
    npy_path = os.path.join(outputs_dir, "best_latent_nom.npy")
    if os.path.exists(csv_path):
        return np.loadtxt(csv_path, delimiter=",", skiprows=1).reshape(-1)
    if os.path.exists(npy_path):
        return np.load(npy_path).reshape(-1)
    return None


def _load_summary(outputs_dir="outputs"):
    p = os.path.join(outputs_dir, "nom_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _load_history(outputs_dir="outputs"):
    p = os.path.join(outputs_dir, "nom_history.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _find_baseline_dat(foil_stem: str, script_dir: str) -> str | None:
    """
    Given a foil filename stem (e.g. "hq358", "naca0012"), find the .txt file
    by searching the standard airfoils_txt/ locations relative to this script.

    Returns the path string if found, None otherwise.
    """
    # Strip any extension the caller may have passed (e.g. "hq358.txt" → "hq358")
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
    Load baseline foil coordinates from its .txt file (Selig format).

    HOW THIS WORKS:
      nom_driver.py saves the name of whichever baseline foil it used
      (e.g. "hq358") into nom_summary.json under the key
      'baseline_foil_filename'.  This function receives that stem,
      finds the matching .txt file in airfoils_txt/, and parses it.

      This means plot_nom_results.py NEVER needs a hardcoded path — it
      always overlays exactly the foil that was used in that particular run.

    ASSUMED FILE FORMAT (standard Selig):
      Row 0        = header line (skipped)
      Rows 0–39    = lower surface, TE→LE  (x: 1→0, y: negative)
      Rows 40–79   = upper surface, LE→TE  (x: 0→1, y: positive)

    FALLBACK:
      If foil_stem is None or the file can't be found, generates an
      analytical NACA 0012 so the plot always works.

    RETURNS: (x_upper, y_upper, x_lower, y_lower)  all in LE→TE order.
    """
    dat_path = None
    label    = foil_stem or "NACA 0012 (fallback)"

    if foil_stem:
        dat_path = _find_baseline_dat(foil_stem, script_dir)
        if dat_path is None:
            print(f"WARNING: could not find {foil_stem}.txt — falling back to NACA 0012")
            label = "NACA 0012 (fallback)"

    if dat_path is None:
        # Analytical NACA 0012 as a last resort
        x  = np.linspace(0, 1, 40)
        yt = 0.12 * (0.2969*np.sqrt(x+1e-9) - 0.126*x
                     - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        return x, yt, x, -yt, label

    data           = np.loadtxt(dat_path, skiprows=1)
    lower_te2le    = data[:40]          # x: 1→0, y: negative
    upper_le2te    = data[40:]          # x: 0→1, y: positive
    lower_le2te    = lower_te2le[::-1]  # flip to LE→TE for consistent plotting

    return (upper_le2te[:, 0], upper_le2te[:, 1],
            lower_le2te[:, 0], lower_le2te[:, 1],
            label)


def main(
    coords_path="outputs/best_coords_nom.csv",
    outputs_dir="outputs",
    n_points=40,
    show_baseline=True,    # renamed from show_naca — now loads whatever baseline was used
    show_convergence=True,
):
    """
    Plot the TalarAI optimized foil vs. the actual baseline used for that run.

    HOW THE BASELINE IS CHOSEN AUTOMATICALLY:
      nom_driver.py saves the name of the starting foil (e.g. "hq358") into
      outputs/nom_summary.json as 'baseline_foil_filename'.
      This function reads that key and loads the matching .txt file from
      airfoils_txt/.  No hardcoding needed — the plot always matches the run.

      If nom_summary.json doesn't have the key (older runs), it falls back
      to NACA 0012.
    """
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Could not find: {coords_path}")

    coords  = np.loadtxt(coords_path, delimiter=",", skiprows=1)
    summary = _load_summary(outputs_dir)
    history = _load_history(outputs_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Auto-detect baseline foil from nom_summary.json ────────────────
    # nom_driver.py writes 'baseline_foil_filename' (e.g. "hq358") into the
    # summary JSON every run.  We read it here so the correct foil is always
    # overlaid without any hardcoded paths.
    baseline_stem = None
    if summary is not None:
        baseline_stem = summary.get('baseline_foil_filename')  # e.g. "hq358" or None

    xu_base, yu_base, xl_base, yl_base, baseline_label = load_baseline_foil(
        baseline_stem, script_dir
    )

    # ── Unpack coords (talarai_pipeline.py convention) ─────────────────
    # coords[:40]  = upper surface  TE→LE  (x: 1→0)
    # coords[40:]  = lower surface  LE→TE  (x: 0→1)
    # Flip upper so both go LE→TE for plotting.
    upper_le2te = coords[:n_points][::-1]      # x: 0→1
    lower_le2te = coords[n_points:2*n_points]  # x: 0→1

    print("\n=== COORDS DIAGNOSTIC ===")
    print(f"Upper LE: x={upper_le2te[0,0]:.4f}  y={upper_le2te[0,1]:+.5f}")
    print(f"Upper TE: x={upper_le2te[-1,0]:.4f}  y={upper_le2te[-1,1]:+.5f}")
    print(f"Lower LE: x={lower_le2te[0,0]:.4f}  y={lower_le2te[0,1]:+.5f}")
    print(f"Lower TE: x={lower_le2te[-1,0]:.4f}  y={lower_le2te[-1,1]:+.5f}")
    xg = np.linspace(0.05, 0.90, 200)
    yu = np.interp(xg, upper_le2te[:,0], upper_le2te[:,1])
    yl = np.interp(xg, lower_le2te[:,0], lower_le2te[:,1])
    t  = yu - yl
    c  = 0.5 * (yu + yl)
    print(f"Max thickness: {np.max(t):.4f} ({np.max(t)*100:.1f}%c)")
    print(f"Max camber:    {np.max(np.abs(c)):.4f} ({np.max(np.abs(c))*100:.1f}%c)")
    print(f"Baseline foil: {baseline_label}")
    print("=========================\n")

    has_conv = show_convergence and history is not None

    # ── Layout: 2 rows ──────────────────────────────────────────────────
    # Row 0 (tall): foil plot | convergence plot   (nothing overlapping foil)
    # Row 1 (short): full-width stats panel
    fig = plt.figure(figsize=(13, 7.5))
    fig.patch.set_facecolor("#1a1a2e")

    if has_conv:
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[3.2, 1],
            width_ratios=[1.6, 1],
            hspace=0.38, wspace=0.3,
            figure=fig,
        )
        ax_foil  = fig.add_subplot(gs[0, 0])
        ax_conv  = fig.add_subplot(gs[0, 1])
        ax_stats = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1],
                               hspace=0.38, figure=fig)
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

    # ── Foil surfaces ───────────────────────────────────────────────────
    ax_foil.plot(upper_le2te[:,0], upper_le2te[:,1],
                 color="#4fc3f7", lw=2.2, label="TalarAI upper")
    ax_foil.plot(lower_le2te[:,0], lower_le2te[:,1],
                 color="#81d4fa", lw=2.2, ls="--", label="TalarAI lower")
    xf  = np.linspace(0, 1, 300)
    yuf = np.interp(xf, upper_le2te[:,0], upper_le2te[:,1])
    ylf = np.interp(xf, lower_le2te[:,0], lower_le2te[:,1])
    ax_foil.fill_between(xf, ylf, yuf, alpha=0.15, color="#4fc3f7")

    # ── Baseline overlay ────────────────────────────────────────────────
    if show_baseline:
        ax_foil.plot(xu_base, yu_base, color="#ff7043", lw=1.4, ls=":",
                     alpha=0.85, label=baseline_label)
        ax_foil.plot(xl_base, yl_base, color="#ff7043", lw=1.4, ls=":",
                     alpha=0.85)

    ax_foil.axhline(0, color="#444466", lw=0.8)

    all_y = np.concatenate([upper_le2te[:,1], lower_le2te[:,1]])
    y_pad = (all_y.max() - all_y.min()) * 0.4
    ax_foil.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax_foil.set_xlim(-0.02, 1.05)
    ax_foil.set_aspect("equal")

    ax_foil.set_title(f"TalarAI Optimized Hydrofoil vs {baseline_label}",
                      color="#c5cae9", fontsize=11, pad=8)
    ax_foil.set_xlabel("x/c", color="#aaaacc", fontsize=9)
    ax_foil.set_ylabel("y/c", color="#aaaacc", fontsize=9)
    ax_foil.legend(fontsize=7.5, framealpha=0.4, facecolor="#0f0f23",
                   edgecolor="#333366", labelcolor="#c5cae9", loc="upper right")

    # ── Convergence panel ───────────────────────────────────────────────
    if ax_conv is not None and history:
        iters = [h["iter"]      for h in history]
        objs  = [h["objective"] for h in history]
        best_curve = []
        cur = float("inf")
        for o in objs:
            cur = min(cur, o)
            best_curve.append(cur)
        ax_conv.plot(iters, objs,
                     color="#4fc3f7", alpha=0.20, lw=0.5, label="CD/CL each iter")
        ax_conv.plot(iters, best_curve,
                     color="#ffd54f", lw=1.8, label="Best so far")
        ax_conv.set_title("NOM Search Convergence",
                          color="#c5cae9", fontsize=11, pad=8)
        ax_conv.set_xlabel("Iteration", color="#aaaacc", fontsize=9)
        ax_conv.set_ylabel("CD / CL  (minimize)", color="#aaaacc", fontsize=9)
        ax_conv.legend(fontsize=7.5, framealpha=0.3, facecolor="#0f0f23",
                       edgecolor="#333366", labelcolor="#c5cae9", loc="upper right")

    # ── Full-width stats panel (row 1) ──────────────────────────────────
    # Compute geometry stats from actual foil coords
    xg2         = np.linspace(0.05, 0.90, 500)
    yu2         = np.interp(xg2, upper_le2te[:, 0], upper_le2te[:, 1])
    yl2         = np.interp(xg2, lower_le2te[:, 0], lower_le2te[:, 1])
    thick       = yu2 - yl2
    camber_line = 0.5 * (yu2 + yl2)

    max_thick_val  = float(np.max(thick))
    max_thick_x    = float(xg2[np.argmax(thick)])
    max_camber_val = float(np.max(np.abs(camber_line)))
    max_camber_x   = float(xg2[np.argmax(np.abs(camber_line))])
    te_gap_val     = float(upper_le2te[-1, 1] - lower_le2te[-1, 1])

    # Pull from summary
    CL        = summary.get("best_CL", 0)            if summary else 0
    CD        = summary.get("best_CD", 0)            if summary else 0
    LD        = CL / CD                               if CD > 0  else 0
    alpha_val = summary.get("alpha", "?")            if summary else "?"
    Re_val    = summary.get("Re",    0)              if summary else 0
    tf_ran_v  = summary.get("tf_ran",      False)    if summary else False
    tf_imp_v  = summary.get("tf_improved", False)    if summary else False
    tf_LD_v   = summary.get("tf_LD",       None)     if summary else None
    tf_CL_v   = summary.get("tf_CL",       None)     if summary else None
    tf_CD_v   = summary.get("tf_CD",       None)     if summary else None
    n_ep      = summary.get("tf_n_epochs", 400)      if summary else 400
    lr_tf     = summary.get("tf_learning_rate", 0.0005) if summary else 0.0005

    if not tf_ran_v:
        tf_status = "Did not run"
    elif tf_imp_v:
        tf_status = f"✓ Improved start  L/D={tf_LD_v:.1f}  CL={tf_CL_v:.4f}  CD={tf_CD_v:.6f}"
    else:
        tf_ld_str = f"L/D={tf_LD_v:.1f}" if tf_LD_v else "eval failed"
        tf_status = f"Did not improve on baseline ({tf_ld_str})"

    col_left = (
        f"  AERODYNAMICS (NOM result)\n"
        f"  CL     {CL:.4f}\n"
        f"  CD     {CD:.6f}\n"
        f"  L/D    {LD:.1f}\n"
        f"  α      {alpha_val}°\n"
        f"  Re     {Re_val:.2e}"
    )

    col_mid = (
        f"  GEOMETRY\n"
        f"  Max thickness  {max_thick_val*100:.2f}%c  at x={max_thick_x:.2f}c\n"
        f"  Max camber     {max_camber_val*100:.2f}%c  at x={max_camber_x:.2f}c\n"
        f"  TE gap         {te_gap_val*100:.3f}%c\n"
        f"  Limit t_min    {summary.get('min_thickness', 0)*100:.2f}%c\n"
        f"  Limit t_max    {summary.get('max_thickness', 0)*100:.2f}%c\n"
        f"  Limit camber   ≤4%c"
    ) if summary else "  GEOMETRY\n  (no summary)"

    col_right = (
        f"  TF TRAINING  ({n_ep} epochs, lr={lr_tf})\n"
        f"  {tf_status}\n"
        f"\n"
        f"  NOM SEARCH\n"
        f"  Valid evals   {summary.get('valid_evals','?')}/{summary.get('n_iters','?')}\n"
        f"  Hard rejects  {summary.get('skipped','?')}\n"
        f"  Baseline      {summary.get('baseline_foil_filename','?')}"
    ) if summary else "  TF / NOM\n  (no summary)"

    ax_stats.axis("off")
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#0a0a1a",
                     alpha=0.85, edgecolor="#333366")
    ax_stats.text(0.01, 0.95, col_left,  transform=ax_stats.transAxes,
                  fontsize=7.8, va="top", ha="left", color="#e0e0ff",
                  fontfamily="monospace", bbox=box_style)
    ax_stats.text(0.35, 0.95, col_mid,   transform=ax_stats.transAxes,
                  fontsize=7.8, va="top", ha="left", color="#b2dfdb",
                  fontfamily="monospace", bbox=box_style)
    ax_stats.text(0.68, 0.95, col_right, transform=ax_stats.transAxes,
                  fontsize=7.8, va="top", ha="left", color="#ffe082",
                  fontfamily="monospace", bbox=box_style)

    out_png = os.path.join(outputs_dir, "nom_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    import pathlib
    # plot_nom_results.py lives in tools/ — go up one level to project root,
    # then into outputs/ where nom_driver.py saves all its files.
    project_root = pathlib.Path(__file__).resolve().parent.parent
    outputs      = project_root / "outputs"
    main(
        coords_path=str(outputs / "best_coords_nom.csv"),
        outputs_dir=str(outputs),
    )