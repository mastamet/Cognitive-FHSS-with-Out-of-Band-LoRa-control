"""Cognitive FHSS link monitor - Flask backend.   Run:  python3 app.py   then open http://localhost:8000"""
import os
from flask import Flask, jsonify, request, render_template
import db, replay

app = Flask(__name__)
LIVE = {"run": None}

@app.get("/")
def index(): return render_template("index.html", scenarios=replay.SCENARIOS)

@app.get("/health")
def health(): return jsonify(ok=True)

@app.get("/api/state")
def api_state():
    run = request.args.get("run", type=int) or (db.latest_run() or {}).get("id")
    s = db.state(run) if run else None
    return jsonify(s or {"run": None})

@app.get("/api/series")
def api_series():
    run = request.args.get("run", type=int); after = request.args.get("after", 0, type=float)
    return jsonify(db.series(run, after) if run else [])

@app.get("/api/hops")
def api_hops():
    run = request.args.get("run", type=int)
    return jsonify(db.hops(run) if run else [])

@app.post("/api/replay")
def api_replay():
    d = request.get_json(silent=True) or {}
    rid = replay.start(d.get("scenario", "S1"), float(d.get("speed", 1)))
    return jsonify(run=rid)

# ---- endpoints for the real hardware (ESP32-S3 receiver or a laptop reading its serial port) ----
@app.post("/api/ingest")
def api_ingest():
    d = request.get_json(force=True)
    if LIVE["run"] is None or d.get("new_run"):
        LIVE["run"] = db.new_run("Live hardware", "live", d.get("duration", 0))
    for b in (d["bins"] if "bins" in d else [d]): db.add_bin(LIVE["run"], b)
    return jsonify(run=LIVE["run"])

@app.post("/api/hop")
def api_hop():
    if LIVE["run"] is None: return jsonify(error="no live run"), 400
    db.add_hop(LIVE["run"], request.get_json(force=True)); return jsonify(ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False, threaded=True)
