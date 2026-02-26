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
    fig = plt.figure(figsize=(13, 5) if has_conv else (9, 5))
    fig.patch.set_facecolor("#1a1a2e")

    if has_conv:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], figure=fig, wspace=0.3)
        ax_foil = fig.add_subplot(gs[0])
        ax_conv = fig.add_subplot(gs[1])
    else:
        ax_foil = fig.add_subplot(111)
        ax_conv = None

    for ax in [ax_foil] + ([ax_conv] if ax_conv else []):
        ax.set_facecolor("#0f0f23")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333366")
        ax.grid(True, color="#1f1f44", linewidth=0.6, linestyle="--")

    # -- foil surfaces --
    ax_foil.plot(upper_le2te[:,0], upper_le2te[:,1],
                 color="#4fc3f7", lw=2.2, label="TalarAI upper")
    ax_foil.plot(lower_le2te[:,0], lower_le2te[:,1],
                 color="#81d4fa", lw=2.2, ls="--", label="TalarAI lower")
    xf  = np.linspace(0, 1, 300)
    yuf = np.interp(xf, upper_le2te[:,0], upper_le2te[:,1])
    ylf = np.interp(xf, lower_le2te[:,0], lower_le2te[:,1])
    ax_foil.fill_between(xf, ylf, yuf, alpha=0.15, color="#4fc3f7")

    # ── Baseline overlay (auto-loaded from nom_summary.json) ───────────
    # Uses whatever foil nom_driver.py actually started from this run.
    # Label comes from load_baseline_foil() — e.g. "hq358" or "NACA 0012 (fallback)".
    if show_baseline:
        ax_foil.plot(xu_base, yu_base, color="#ff7043", lw=1.4, ls=":",
                     alpha=0.85, label=baseline_label)
        ax_foil.plot(xl_base, yl_base, color="#ff7043", lw=1.4, ls=":",
                     alpha=0.85)

    ax_foil.axhline(0, color="#444466", lw=0.8)

    # -- TIGHT y-axis (the fix: old plot used axis=equal with x-range 0..1
    #    which forced y to -0.5..+0.5 making the foil look tiny and the lower
    #    surface appear "flipped". Now we set y limits from actual foil data.) --
    all_y = np.concatenate([upper_le2te[:,1], lower_le2te[:,1]])
    y_pad = (all_y.max() - all_y.min()) * 0.4
    ax_foil.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax_foil.set_xlim(-0.02, 1.05)
    ax_foil.set_aspect("equal")

    if summary:
        CL = summary.get("best_CL", 0)
        CD = summary.get("best_CD", 0)
        LD = CL / CD if CD > 0 else 0
        ax_foil.text(0.97, 0.05,
                     f"CL = {CL:.4f}   CD = {CD:.5f}\n"
                     f"L/D = {LD:.1f}   α={summary.get('alpha','?')}°"
                     f"   Re={summary.get('Re',0):.0e}",
                     transform=ax_foil.transAxes,
                     fontsize=8, va="bottom", ha="right", color="#e0e0ff",
                     bbox=dict(boxstyle="round,pad=0.4",
                               facecolor="#0a0a1a", alpha=0.85,
                               edgecolor="#555588"))

    ax_foil.set_title(f"TalarAI Optimized Hydrofoil vs {baseline_label}",
                      color="#c5cae9", fontsize=11, pad=8)
    ax_foil.set_xlabel("x/c", color="#aaaacc", fontsize=9)
    ax_foil.set_ylabel("y/c", color="#aaaacc", fontsize=9)
    ax_foil.legend(fontsize=7.5, framealpha=0.4, facecolor="#0f0f23",
                   edgecolor="#333366", labelcolor="#c5cae9", loc="upper right")

    # -- convergence panel --
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
        ax_conv.set_title("Optimization Convergence",
                          color="#c5cae9", fontsize=11, pad=8)
        ax_conv.set_xlabel("Iteration", color="#aaaacc", fontsize=9)
        ax_conv.set_ylabel("CD / CL  (minimize)", color="#aaaacc", fontsize=9)
        ax_conv.legend(fontsize=7.5, framealpha=0.3, facecolor="#0f0f23",
                       edgecolor="#333366", labelcolor="#c5cae9", loc="upper right")
        if summary:
            ax_conv.text(0.03, 0.05,
                         f"Valid evals: {summary['valid_evals']}/{summary['n_iters']}\n"
                         f"Skipped (hard reject): {summary['skipped']}",
                         transform=ax_conv.transAxes,
                         fontsize=7.5, va="bottom", ha="left", color="#aaaacc",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="#0a0a1a", alpha=0.8,
                                   edgecolor="#333366"))

    out_png = os.path.join(outputs_dir, "nom_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()