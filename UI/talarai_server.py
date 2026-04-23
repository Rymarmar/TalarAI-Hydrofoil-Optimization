"""
talarai_server.py
Place in: TalarAI-Hydrofoil-Optimization/experiments/UI stuff/

Calls nom_optimize() directly. Each iteration, the NOMModel subclass pushes
the result into a queue. The SSE endpoint drains that queue in real time.
No file polling. No blocking sleeps in the response thread.

Usage:
  cd "TalarAI-Hydrofoil-Optimization/experiments/UI stuff"
  pip install flask flask-cors
  python talarai_server.py
"""

from __future__ import annotations
import sys, json, time, uuid, queue, threading, traceback, argparse
from datetime import datetime
from pathlib import Path

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
CSV_PATH     = str(PROJECT_ROOT / "data" / "airfoil_latent_params_6.csv")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500",
                   "http://127.0.0.1:5000", "http://localhost:5000"])

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")
    allowed = ["http://127.0.0.1:5500", "http://localhost:5500",
               "http://127.0.0.1:5000", "http://localhost:5000"]
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"]  = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_pipeline_loaded = False
_pipeline_lock   = threading.Lock()


def _preload_pipeline():
    global _pipeline_loaded
    with _pipeline_lock:
        if _pipeline_loaded:
            return
        try:
            from pipeline.talarai_pipeline import TalarAIPipeline
            TalarAIPipeline()
            _pipeline_loaded = True
            print("[server] Pipeline preloaded OK")
        except Exception as e:
            print(f"[server] Pipeline preload failed: {e}")


def _get_pipeline():
    from pipeline.talarai_pipeline import TalarAIPipeline
    return TalarAIPipeline()


# ── Serve the HTML at / ───────────────────────────────────────────────────

@app.route("/")
def index():
    for name in ["Talarai.html", "talariai.html", "index.html"]:
        html_path = _HERE / name
        if html_path.exists():
            return send_file(str(html_path))
    return "HTML file not found (expected Talarai.html or index.html)", 404


# ── Health ─────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"ok": True, "pipeline_loaded": _pipeline_loaded})


# ── Encoder helpers (live in server, not pipeline) ─────────────────────────
#
# The encoder takes 256x256 grayscale PNGs — same as the training pipeline
# (encode_airfoils.py: airfoil .txt → PNG → CNN encoder → 6 latent params).
# We reconstruct the architecture here, load the saved weights, and render
# drawn coords to an in-memory PNG before calling the CNN.

_ENC_IMG_SIZE = 256
_ENC_N_PARAMS = 6
_ENCODER_CANDIDATES = [
    PROJECT_ROOT / "pipeline" / "encoder_6params.weights.h5",
    PROJECT_ROOT / "encoder_6params.weights.h5",
    _HERE.parent / "pipeline" / "encoder_6params.weights.h5",
    _HERE.parent / "encoder_6params.weights.h5",
    _HERE / "encoder_6params.weights.h5",
]
_ENCODER_PATH = next((p for p in _ENCODER_CANDIDATES if p.exists()), None)

_encoder_model  = None
_encoder_loaded = False
_encoder_lock   = threading.Lock()


def _build_encoder_model():
    """
    Exact CNN architecture from encode_airfoils.py.
    Must match training exactly so saved weights load correctly.
    """
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    )
    from tensorflow.keras.models import Model

    inp = Input(shape=(_ENC_IMG_SIZE, _ENC_IMG_SIZE, 1), name="image")
    x = Conv2D(32,  (3, 3), activation='relu', padding='valid')(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64,  (3, 3), activation='relu', padding='valid')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(100,  activation='relu')(x)
    x = Dropout(0.05)(x)
    latent = Dense(_ENC_N_PARAMS, activation='linear',
                   name=f"latent_{_ENC_N_PARAMS}")(x)
    return Model(inp, latent, name="encoder")


def _load_encoder():
    """Load encoder weights once, thread-safe."""
    global _encoder_model, _encoder_loaded
    with _encoder_lock:
        if _encoder_loaded:
            return
        if _ENCODER_PATH is None:
            searched = "\n  ".join(str(p) for p in _ENCODER_CANDIDATES)
            raise FileNotFoundError(
                f"Encoder weights not found. Searched:\n  {searched}\n"
                f"Copy encoder_6params.weights.h5 to the pipeline/ folder."
            )
        import numpy as np
        model = _build_encoder_model()
        # Dummy forward pass builds the graph before loading weights
        model(np.zeros((1, _ENC_IMG_SIZE, _ENC_IMG_SIZE, 1), dtype="float32"),
              training=False)
        model.load_weights(str(_ENCODER_PATH))
        _encoder_model  = model
        _encoder_loaded = True
        print(f"[server] Encoder loaded from {_ENCODER_PATH}")


def _coords_to_png_array(coords: "np.ndarray") -> "np.ndarray":
    """
    Render (80,2) Selig coords → 256×256 grayscale float32 array in memory.

    Matches the training images exactly:
      - White filled foil silhouette on black background
      - x range [-0.05, 1.05],  y range [-0.25, 0.25]
      - matplotlib Agg backend (no display, no disk I/O)
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    coords = np.asarray(coords, dtype=float)
    upper_le2te = coords[:40][::-1]   # flip upper TE->LE to LE->TE
    lower_le2te = coords[40:]
    xc = np.concatenate([upper_le2te[:, 0], lower_le2te[::-1, 0]])
    yc = np.concatenate([upper_le2te[:, 1], lower_le2te[::-1, 1]])
    # Negate y: the training PNGs were airfoil images where the profile appears
    # with the upper surface visually on top (positive y upward in matplotlib).
    # The Selig coords we build have upper surface at positive y, which renders
    # at the top — same convention. However the encoder+decoder chain inverts the
    # camber, so we flip y here so the rendered PNG matches the training orientation
    # that produces z values which decode to a correctly-oriented foil.
    yc = -yc

    fig, ax = plt.subplots(figsize=(1, 1), dpi=_ENC_IMG_SIZE)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.fill(xc, yc, color='white')
    ax.plot(xc, yc, color='white', linewidth=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=_ENC_IMG_SIZE, bbox_inches='tight',
                pad_inches=0, facecolor='black')
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert('L').resize((_ENC_IMG_SIZE, _ENC_IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(_ENC_IMG_SIZE, _ENC_IMG_SIZE, 1)


def _encode_image_array(img_arr: "np.ndarray") -> "np.ndarray":
    """
    (256,256,1) float32 array in [0,1] → latent z (6,) via CNN encoder.
    Primary path: accepts a pre-rendered PNG from the browser.
    """
    import numpy as np
    _load_encoder()
    z = _encoder_model.predict(
        img_arr.reshape(1, _ENC_IMG_SIZE, _ENC_IMG_SIZE, 1), verbose=0
    )[0]
    return z.astype(np.float64)


def _encode_selig(selig: "np.ndarray") -> "np.ndarray":
    """
    (80,2) Selig coords → latent z (6,) via server-side PNG render + CNN encoder.
    Fallback path when no browser PNG is available.
    """
    import numpy as np
    _load_encoder()
    img = _coords_to_png_array(selig)
    return _encode_image_array(img)


# ── Encode drawn foil coords → latent z ───────────────────────────────────
#
# POST /api/encode
# Body: { "coords": [[x,y], ...] }   (sparse drawn points, any order)
#
# Pipeline:
#   1. Normalise chord to [0,1]
#   2. _resample_to_selig(): extract upper/lower envelopes, interpolate
#      onto 40-point linspace each → (80,2) Selig-ordered array
#   3. pipeline.encode_coords(selig):
#        renders to 256×256 grayscale PNG in memory (matplotlib Agg)
#        → CNN encoder (same arch as training) → z (6,)
#   4. pipeline.eval_latent_with_neuralfoil(z): decode z → NeuralFoil → CL/CD
#
# The returned z is sent back to /api/optimize/start as z_init, feeding
# directly into nom_optimize(z_init_array=z) → Priority 0 starting point.

@app.route("/api/encode", methods=["POST"])
def encode_foil():
    body   = request.get_json(force=True) or {}
    alpha  = float(body.get("alpha", 2.0))
    Re     = float(body.get("Re", 100_000))

    import numpy as np

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return jsonify({"error": f"Pipeline load failed: {e}"}), 500

    z = None

    # ── Primary path: browser-rendered PNG ────────────────────────────────
    # The browser renders the drawn foil to a 256x256 grayscale PNG matching
    # the training image format exactly (white fill on black, same axis limits).
    # This is sent as a base64 data URL and decoded here before encoding.
    image_b64 = body.get("image", "")
    if image_b64:
        try:
            import base64, io
            from PIL import Image

            # Strip the data URL header (data:image/png;base64,...)
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("L")
            img = img.resize((_ENC_IMG_SIZE, _ENC_IMG_SIZE), Image.LANCZOS)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_arr = img_arr.reshape(_ENC_IMG_SIZE, _ENC_IMG_SIZE, 1)

            z = _encode_image_array(img_arr)
            print(f"  [encode] used browser PNG  z={np.round(z, 4)}")
        except Exception as e:
            print(f"  [encode] browser PNG failed ({e}), falling back to coords")
            z = None

    # ── Fallback: server-side PNG render from coords ───────────────────────
    if z is None:
        coords = body.get("coords")
        if not coords or len(coords) < 6:
            return jsonify({"error": "image or coords required"}), 400
        try:
            pts = np.array(coords, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                return jsonify({"error": "coords must be [[x,y], ...]"}), 400
            x_range = pts[:, 0].max() - pts[:, 0].min()
            if x_range < 1e-6:
                return jsonify({"error": "All points have the same x"}), 400
            pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / x_range
            pts[:, 1] =  pts[:, 1] / x_range
            selig = _resample_to_selig(pts)
            z = _encode_selig(selig)
            print(f"  [encode] used server PNG fallback  z={np.round(z, 4)}")
        except Exception as e:
            return jsonify({"error": f"Encoding failed: {e}"}), 500

    z = np.array(z, dtype=float).reshape(6)

    # ── Build display coords from the drawn points ─────────────────────────
    # Always show the resampled drawn shape in the UI, not the decoder output.
    coords = body.get("coords", [])
    if coords:
        try:
            pts = np.array(coords, dtype=float)
            x_range = pts[:, 0].max() - pts[:, 0].min()
            if x_range > 1e-6:
                pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / x_range
                pts[:, 1] =  pts[:, 1] / x_range
                display_coords = _resample_to_selig(pts)
            else:
                display_coords = None
        except Exception:
            display_coords = None
    else:
        display_coords = None

    # ── Evaluate aerodynamics at the encoded z ─────────────────────────────
    CL = CD = LD = 0.0
    try:
        from optimization.ui_nom_driver import snap_condition
        a_s, r_s = snap_condition(alpha, Re)
        out = pipeline.eval_latent_with_neuralfoil(z, alpha=a_s, Re=r_s)
        CL  = float(out["CL"])
        CD  = float(out["CD"])
        LD  = CL / CD if CD > 0 else 0.0
        if display_coords is None:
            decoded = out.get("coords")
            display_coords = decoded if decoded is not None else None
    except Exception as e:
        print(f"  [encode] NeuralFoil eval failed: {e}")

    return jsonify({
        "z":      z.tolist(),
        "CL":     round(CL, 4),
        "CD":     round(CD, 6),
        "LD":     round(LD, 2),
        "coords": (display_coords.tolist()
                   if hasattr(display_coords, "tolist") else
                   (display_coords if display_coords is not None else [])),
    })


def _resample_to_selig(pts: "np.ndarray") -> "np.ndarray":
    """
    Resample an arbitrary cloud of foil points into 80-point Selig order:
      rows  0-39: upper surface TE→LE  (x decreasing)
      rows 40-79: lower surface LE→TE  (x increasing)

    Upper surface = highest y at each x station.
    Lower surface = lowest  y at each x station.

    The draw canvas allows x in [-0.05, 1.05] so users can draw a rounded LE.
    Points with x < 0 are used to determine the LE y-midpoint and the
    curvature approaching x=0, giving a smooth rounded nose rather than a
    flat vertical edge.
    """
    import numpy as np

    xg = np.linspace(0.0, 1.0, 40)
    pts_sorted = pts[np.argsort(pts[:, 0])]
    xs, ys = pts_sorted[:, 0], pts_sorted[:, 1]

    yu = np.zeros(40)
    yl = np.zeros(40)
    for i, xi in enumerate(xg):
        # Use a wider window near LE to capture points that extend to x<0
        win = 0.10 if xi < 0.15 else 0.08
        mask = np.abs(xs - xi) < win
        if mask.any():
            yu[i] = ys[mask].max()
            yl[i] = ys[mask].min()
        else:
            yu[i] = float(np.interp(xi, xs, ys))
            yl[i] = yu[i]

    # At x=0 (LE), force upper and lower to meet at their midpoint so the
    # foil has zero thickness at the LE — avoids the flat vertical edge
    # that appears when upper and lower strokes don't quite meet.
    le_mid = (yu[0] + yl[0]) / 2.0
    yu[0] = le_mid
    yl[0] = le_mid

    upper_le2te = np.stack([xg, yu], axis=1)
    lower_le2te = np.stack([xg, yl], axis=1)
    upper_te2le = upper_le2te[::-1].copy()   # flip to TE→LE

    return np.vstack([upper_te2le, lower_le2te])  # (80, 2)


# ── Start optimization ─────────────────────────────────────────────────────

@app.route("/api/optimize/start", methods=["POST"])
def optimize_start():
    body   = request.get_json(force=True) or {}
    job_id = str(uuid.uuid4())

    run_dir      = OUTPUTS_DIR / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    summary_path = run_dir / "nom_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    alpha   = float(body.get("alpha",            2.0))
    Re      = float(body.get("Re",               100_000))
    n_iters = int(body.get("n_iters",            150))
    lr      = float(body.get("tf_learning_rate", 0.0005))

    # foil_name_override: user typed a foil name in the baseline field (e.g. "e61" or "n0012")
    # This is passed directly as foil_name to nom_optimize, which looks it up in the
    # latent CSV by name. Accepts with or without .png extension, case-insensitive.
    foil_name_override = body.get("foil_name_override", "").strip()
    # Normalise: add .png if missing, lowercase
    if foil_name_override and not foil_name_override.lower().endswith(".png"):
        foil_name_override = foil_name_override.lower() + ".png"
    elif foil_name_override:
        foil_name_override = foil_name_override.lower()

    # z_init: optional 6-element list from /api/encode (drawn foil)
    z_init_raw = body.get("z_init", None)
    z_init     = None
    if z_init_raw is not None:
        try:
            import numpy as np
            z_init = list(np.array(z_init_raw, dtype=float).reshape(6))
        except Exception:
            z_init = None

    q = queue.Queue(maxsize=1000)

    with _jobs_lock:
        _jobs[job_id] = {
            "status":       "running",
            "result":       None,
            "queue":        q,
            "summary_path": summary_path,
            "n_iters":      n_iters,
        }

    rod_a_x    = float(body.get("rod_a_x",    0.50))
    rod_a_diam = float(body.get("rod_a_diam", 0.0))   # 0 = disabled
    rod_b_x    = float(body.get("rod_b_x",    0.25))
    rod_b_diam = float(body.get("rod_b_diam", 0.0))   # 0 = disabled

    # Priority: drawn foil z_init > named foil override > auto lookup
    kwargs = dict(
        alpha=alpha, Re=Re, n_iters=n_iters, tf_learning_rate=lr,
        csv_path=CSV_PATH,
        foil_name=foil_name_override if (foil_name_override and not z_init) else (
                  "" if z_init else "n0012.png"),
        lookup_baseline_path="",
        z_init_array=z_init,
        out_path=str(run_dir), live_display=False, save_frames=False,
        rod_a_x=rod_a_x, rod_a_diam=rod_a_diam,
        rod_b_x=rod_b_x, rod_b_diam=rod_b_diam,
    )

    t = threading.Thread(target=_run_nom, args=(job_id, q, kwargs), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


# ── NOM runner ─────────────────────────────────────────────────────────────

def _run_nom(job_id: str, q: queue.Queue, kwargs: dict):
    try:
        import optimization.ui_nom_driver as nd

        cancel_event = threading.Event()

        class _CancelledError(Exception):
            pass

        class _StreamingNOMModel(nd.NOMModel):
            def train_step(self, data):
                if cancel_event.is_set():
                    raise _CancelledError("cancelled by user")
                _orig_eval = self.nf_op._evaluate
                def _cancellable_eval(z):
                    if cancel_event.is_set():
                        raise _CancelledError("cancelled mid-iteration")
                    return _orig_eval(z)
                self.nf_op._evaluate = _cancellable_eval
                try:
                    result = super().train_step(data)
                finally:
                    self.nf_op._evaluate = _orig_eval
                if self.history_log:
                    entry = dict(self.history_log[-1])
                    if entry.get("coords") is not None:
                        import numpy as np
                        entry["coords"] = np.asarray(entry["coords"]).tolist()
                    try:
                        q.put_nowait(entry)
                        it = entry.get("iter", "?")
                        print(f"[stream] queued iter {it}, queue size={q.qsize()}")
                    except queue.Full:
                        print("[stream] queue full, dropping entry")
                return result

        _Orig = nd.NOMModel
        with _jobs_lock:
            _jobs[job_id]["cancel_event"] = cancel_event

        nd.NOMModel = _StreamingNOMModel
        try:
            nd.nom_optimize(**kwargs)
        except _CancelledError:
            print(f"[server] Job {job_id[:8]} cancelled by user")
        finally:
            nd.NOMModel = _Orig

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[server] NOM error: {err}\n{traceback.format_exc()}")
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
        try:
            q.put_nowait({"__error__": err})
        except Exception:
            pass

    finally:
        try:
            q.put_nowait({"__done__": True})
        except Exception:
            pass


# ── SSE stream ─────────────────────────────────────────────────────────────

@app.route("/api/optimize/stream")
def optimize_stream():
    job_id = request.args.get("job_id")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job_id or job is None:
        return jsonify({"error": "Unknown job_id"}), 404

    q            = job["queue"]
    n_iters      = job["n_iters"]
    summary_path = job["summary_path"]

    def generate():
        best_LD     = 0.0
        best_coords = None
        sent        = 0

        while True:
            try:
                entry = q.get(timeout=2.0)
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue

            if "__done__" in entry:
                break
            if "__error__" in entry:
                yield f"data: {json.dumps({'error': entry['__error__'], 'done': True})}\n\n"
                return

            CL    = float(entry.get("CL",    0) or 0)
            CD    = float(entry.get("CD",    0) or 0)
            cd_cl = float(entry.get("cd_cl", 0) or 0)
            loss  = float(entry.get("loss",  0) or 0)
            pen   = float(entry.get("pen",   0) or 0)
            LD    = CL / CD if CD > 0 else 0.0
            if LD > best_LD:
                best_LD     = LD
                best_coords = entry.get("coords")
            sent += 1

            event = {
                "iter":    entry.get("iter", sent),
                "n_iters": n_iters,
                "CL":      round(CL,      4),
                "CD":      round(CD,      6),
                "cd_cl":   round(cd_cl,   6),
                "loss":    round(loss,    6),
                "pen":     round(pen,     6),
                "LD":      round(LD,      2),
                "best_LD": round(best_LD, 2),
                "pct":     round(entry.get("iter", sent) / n_iters * 100, 1),
                "coords":  entry.get("coords"),
                "done":    False,
            }
            yield f"data: {json.dumps(event)}\n\n"

        summary = {}
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass
        with _jobs_lock:
            _jobs[job_id]["result"] = summary
        yield f"data: {json.dumps({**summary, 'done': True, 'final': True, 'best_coords': best_coords})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Baseline foil ──────────────────────────────────────────────────────────

@app.route("/api/baseline")
def baseline():
    alpha      = float(request.args.get("alpha",     2.0))
    Re         = float(request.args.get("Re",         100_000))
    foil_name  = request.args.get("foil_name", "").strip()

    from optimization.ui_nom_driver import snap_condition
    a_s, r_s = snap_condition(alpha, Re)

    import numpy as np
    import pandas as pd

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    z        = None
    filename = foil_name or "?"

    if foil_name:
        try:
            df = pd.read_csv(CSV_PATH)
            rows = df[df["filename"] == foil_name]
            if rows.empty:
                needle = foil_name.lower().replace(".png", "")
                rows = df[df["filename"].str.lower().str.replace(".png","",regex=False) == needle]
            if not rows.empty:
                z = rows.iloc[0][[f"p{i}" for i in range(1,7)]].values.astype(float)
                filename = str(rows.iloc[0]["filename"])
        except Exception as e:
            return jsonify({"error": f"CSV lookup failed: {e}"}), 500
        if z is None:
            return jsonify({"error": f"Foil '{foil_name}' not found in latent CSV"}), 404
    else:
        bl_path = OUTPUTS_DIR / f"best_baseline_foil_alpha{a_s:.1f}_Re{r_s:.1e}.json"
        if not bl_path.exists():
            return jsonify({"error": f"No baseline for alpha={a_s} Re={r_s}"}), 404
        with open(bl_path) as f:
            bl = json.load(f)
        z        = np.array(bl["latent"], dtype=float)
        filename = bl.get("filename", "?")

    try:
        out    = pipeline.eval_latent_with_neuralfoil(z, alpha=a_s, Re=r_s)
        coords = out.get("coords")
        CL     = float(out["CL"])
        CD     = float(out["CD"])
        LD     = CL / CD if CD > 0 else 0.0
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "filename": filename,
        "alpha":    a_s,
        "Re":       r_s,
        "CL":       round(CL, 4),
        "CD":       round(CD, 6),
        "LD":       round(LD, 2),
        "coords":   coords.tolist() if hasattr(coords, "tolist") else coords,
    })


# ── Cancel ─────────────────────────────────────────────────────────────────

@app.route("/api/optimize/cancel", methods=["POST"])
def optimize_cancel():
    job_id = request.get_json(force=True).get("job_id")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    cancel = job.get("cancel_event")
    if cancel:
        cancel.set()
        print(f"[server] Cancel requested for job {job_id[:8]}")
    return jsonify({"ok": True})


# ── Result ─────────────────────────────────────────────────────────────────

@app.route("/api/optimize/result")
def optimize_result():
    job_id = request.args.get("job_id")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job_id"}), 404
    if job["status"] == "running":
        return jsonify({"status": "running"}), 202
    return jsonify(job.get("result") or {"status": job["status"]})


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    if args.preload:
        _preload_pipeline()

    print(f"[server] Project root : {PROJECT_ROOT}")
    print(f"[server] Open browser : http://{args.host}:{args.port}/")

    try:
        from waitress import serve
        print("[server] Using waitress (streaming-capable)")
        serve(app, host=args.host, port=args.port, threads=8)
    except ImportError:
        print("[server] waitress not found — falling back to Flask dev server")
        print("[server] For proper SSE streaming: pip install waitress")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)