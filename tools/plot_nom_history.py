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


def load_naca0012(dat_path: str | None = None):
    """
    Load NACA 0012 coords from the project's actual n0012.txt file.
    Returns (x_upper, y_upper, x_lower, y_lower) all in LE->TE order.

    n0012.txt is in standard Selig format:
      - rows 0..39  = lower surface, TE->LE  (x: 1->0, y negative)
      - rows 40..79 = upper surface, LE->TE  (x: 0->1, y positive)
    """
    # Search common locations relative to this script
    candidates = []
    if dat_path:
        candidates.append(dat_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(script_dir, "airfoils_txt", "n0012.txt"),
        os.path.join(script_dir, "n0012.txt"),
        os.path.join(script_dir, "..", "airfoils_txt", "n0012.txt"),
    ]

    data = None
    for p in candidates:
        if os.path.exists(p):
            data = np.loadtxt(p, skiprows=1)
            break

    if data is None:
        # Fallback: analytical NACA 0012 (only used if file not found)
        print("WARNING: n0012.txt not found, using analytical NACA 0012")
        x = np.linspace(0, 1, 40)
        yt = 0.12*(0.2969*np.sqrt(x+1e-9)-0.126*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
        return x, yt, x, -yt

    # n0012.txt: rows 0-39 = lower TE->LE, rows 40-79 = upper LE->TE
    lower_te2le = data[:40]   # x: 1->0, y: negative
    upper_le2te = data[40:]   # x: 0->1, y: positive
    lower_le2te = lower_te2le[::-1]  # flip to LE->TE for consistent plotting

    return (upper_le2te[:, 0], upper_le2te[:, 1],
            lower_le2te[:, 0], lower_le2te[:, 1])


def main(
    coords_path="outputs/best_coords_nom.csv",
    outputs_dir="outputs",
    n_points=40,
    show_naca=True,
    show_convergence=True,
    naca_dat_path=None,   # path to n0012.txt; if None, auto-searched
):
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Could not find: {coords_path}")

    coords = np.loadtxt(coords_path, delimiter=",", skiprows=1)

    # -------------------------------------------------------------------
    # Unpack coords  (talarai_pipeline.py convention):
    #   coords[:40]  = upper surface  TE->LE  (x: 1->0)
    #   coords[40:]  = lower surface  LE->TE  (x: 0->1)
    # Flip upper so both go LE->TE for plotting.
    # -------------------------------------------------------------------
    upper_le2te = coords[:n_points][::-1]      # x: 0->1
    lower_le2te = coords[n_points:2*n_points]  # x: 0->1

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
    print("=========================\n")

    history = _load_history(outputs_dir)
    summary = _load_summary(outputs_dir)

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

    # -- NACA 0012 overlay (loaded from actual n0012.txt) --
    if show_naca:
        xu_n, yu_n, xl_n, yl_n = load_naca0012(naca_dat_path)
        ax_foil.plot(xu_n, yu_n, color="#ff7043", lw=1.4, ls=":", alpha=0.85, label="NACA 0012")
        ax_foil.plot(xl_n, yl_n, color="#ff7043", lw=1.4, ls=":", alpha=0.85)

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

    ax_foil.set_title("TalarAI Optimized Hydrofoil vs NACA 0012",
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