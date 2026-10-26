"""Tiny SQLite storage layer. On Huawei Cloud you can later swap this file for RDS (MySQL/PostgreSQL)."""
import sqlite3, threading, time, json, os

DB_PATH = os.environ.get("FHSS_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fhss.db"))
_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.executescript("""
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL, scenario TEXT, mode TEXT,
    status TEXT, duration REAL, jam_t0 REAL, jam_type TEXT, jam_chans TEXT);
CREATE TABLE IF NOT EXISTS bins(run_id INTEGER, t REAL, cog_ch INTEGER, stat_ch INTEGER, state INTEGER, p_jam REAL,
    sent_cog INTEGER, recv_cog INTEGER, sent_stat INTEGER, recv_stat INTEGER);
CREATE TABLE IF NOT EXISTS hops(run_id INTEGER, t REAL, from_ch INTEGER, to_ch INTEGER, detect_ms REAL, delay_ms REAL);
CREATE INDEX IF NOT EXISTS ix_bins ON bins(run_id, t);
""")

def q(sql, args=(), one=False):
    with _lock:
        cur = _conn.execute(sql, args); rows = [dict(r) for r in cur.fetchall()]; _conn.commit()
    return (rows[0] if rows else None) if one else rows

def new_run(scenario, mode, duration=0.0, jam=None):
    with _lock:
        cur = _conn.execute("INSERT INTO runs(started,scenario,mode,status,duration,jam_t0,jam_type,jam_chans) VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), scenario, mode, "running", duration,
             jam["t0"] if jam else None, str(jam["type"]) if jam else None, json.dumps([int(c) for c in jam["chans"]]) if jam else "[]"))
        _conn.commit(); return cur.lastrowid

def set_status(run_id, status): q("UPDATE runs SET status=? WHERE id=?", (status, run_id))

def add_bin(run_id, d):
    q("INSERT INTO bins VALUES(?,?,?,?,?,?,?,?,?,?)", (run_id, d["t"], d["cog_ch"], d.get("stat_ch", -1), d["state"], d["p_jam"],
      d["sent_cog"], d["recv_cog"], d.get("sent_stat", 0), d.get("recv_stat", 0)))

def add_hop(run_id, d):
    q("INSERT INTO hops VALUES(?,?,?,?,?,?)", (run_id, d["t"], d["from_ch"], d["to_ch"], d.get("detect_ms"), d.get("delay_ms")))

def latest_run(): return q("SELECT * FROM runs ORDER BY id DESC LIMIT 1", one=True)
def get_run(run_id): return q("SELECT * FROM runs WHERE id=?", (run_id,), one=True)
def series(run_id, after): return q("SELECT * FROM bins WHERE run_id=? AND t>? ORDER BY t", (run_id, after))
def hops(run_id): return q("SELECT * FROM hops WHERE run_id=? ORDER BY t", (run_id,))

def state(run_id):
    run = get_run(run_id)
    if not run: return None
    last = q("SELECT * FROM bins WHERE run_id=? ORDER BY t DESC LIMIT 1", (run_id,), one=True)
    j0 = run["jam_t0"] if run["jam_t0"] is not None else 0.0
    agg = q("SELECT SUM(sent_cog) sc, SUM(recv_cog) rc, SUM(sent_stat) ss, SUM(recv_stat) rs FROM bins WHERE run_id=? AND t>=?",
            (run_id, j0), one=True)
    plr = lambda s, r: (1 - r / s) if s else None
    h = hops(run_id)
    return dict(run=run, last=last, hops=len(h), last_hop=h[-1] if h else None,
                plr_cog=plr(agg["sc"] or 0, agg["rc"] or 0), plr_stat=plr(agg["ss"] or 0, agg["rs"] or 0),
                jam_chans=json.loads(run["jam_chans"] or "[]"))
