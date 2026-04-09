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


# ── Serve the HTML at / so Flask and the page are same-origin ─────────────

@app.route("/")
def index():
    for name in ["Talarai.html", "talariai.html", "index.html"]:
        html_path = _HERE / name
        if html_path.exists():
            return send_file(str(html_path))
    return "HTML file not found (expected talariai.html or index.html)", 404


# ── Health ─────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"ok": True, "pipeline_loaded": _pipeline_loaded})


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

    bl_override = body.get("lookup_baseline_path", "").strip()
    if not bl_override:
        bl_path = ""   # will use foil_name="n0012.png" below
    else:
        bl_path = bl_override

    # Queue that the NOM thread pushes into, SSE thread drains
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
    rod_a_diam = float(body.get("rod_a_diam", 0.08))
    rod_b_x    = float(body.get("rod_b_x",    0.25))
    rod_b_diam = float(body.get("rod_b_diam", 0.06))

    kwargs = dict(
        alpha=alpha, Re=Re, n_iters=n_iters, tf_learning_rate=lr,
        csv_path=CSV_PATH,
        foil_name="" if bl_path else "n0012.png",
        lookup_baseline_path=bl_path,
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
            """Pushes each entry to the queue and checks for cancellation between and within iterations."""
            def train_step(self, data):
                if cancel_event.is_set():
                    raise _CancelledError("cancelled by user")
                # Wrap nf_op._evaluate so cancel is checked between each FD call too
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

        # Swap in our subclass just for this run
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
        # Always signal done so the SSE stream closes cleanly
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
                # Send a comment heartbeat so the connection stays alive
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

        # Run finished — send summary
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


# ── Baseline foil for current (alpha, Re) ──────────────────────────────────

@app.route("/api/baseline")
def baseline():
    """Return coords + metadata for a foil at given alpha/Re.
    If foil_name is given (e.g. 'n0012.png'), look it up in the latent CSV.
    Otherwise fall back to the lookup-table best baseline file.
    """
    alpha      = float(request.args.get("alpha",     2.0))
    Re         = float(request.args.get("Re",         100_000))
    foil_name  = request.args.get("foil_name", "").strip()

    from optimization.ui_nom_driver import snap_condition
    a_s, r_s = snap_condition(alpha, Re)

    import numpy as np
    import pandas as pd

    try:
        from pipeline.talarai_pipeline import TalarAIPipeline
        pipeline = TalarAIPipeline()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # --- Resolve latent vector ---
    z        = None
    filename = foil_name or "?"

    if foil_name:
        # Look up the named foil in the latent CSV
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
        # Fall back to lookup-table JSON
        bl_path = OUTPUTS_DIR / f"best_baseline_foil_alpha{a_s:.1f}_Re{r_s:.1e}.json"
        if not bl_path.exists():
            return jsonify({"error": f"No baseline for alpha={a_s} Re={r_s}"}), 404
        with open(bl_path) as f:
            bl = json.load(f)
        z        = np.array(bl["latent"], dtype=float)
        filename = bl.get("filename", "?")

    # --- Decode and evaluate ---
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