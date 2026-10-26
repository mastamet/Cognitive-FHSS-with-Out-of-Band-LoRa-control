"""Replays the simulator through the dashboard database, as if a live link were reporting."""
import threading, time, numpy as np
import fhss_sim as fs, db

BIN = 0.1   # seconds per dashboard sample
SCENARIOS = {"S1": "Constant jammer", "S2": "Pulsed jammer (50%)", "S3": "Sweeping jammer", "S5": "Wi-Fi + constant jammer"}

class LoadedMLP:
    def __init__(self, path):
        z = np.load(path); self.mu, self.sd = z["mu"], z["sd"]
        self.W = [z["W0"], z["W1"], z["W2"]]; self.b = [z["b0"], z["b1"], z["b2"]]
    def proba(self, f):
        a = (np.asarray(f) - self.mu) / self.sd
        a = np.maximum(a @ self.W[0] + self.b[0], 0); a = np.maximum(a @ self.W[1] + self.b[1], 0)
        z = a @ self.W[2] + self.b[2]; e = np.exp(z - z.max()); return e / e.sum()

_clf = None
def clf():
    global _clf
    if _clf is None:
        import os; _clf = LoadedMLP(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier_weights.npz"))
    return _clf

def simulate(scn, seed, T):
    rng = np.random.default_rng(seed)
    probe = fs.make_env(scn, np.random.default_rng(seed), 0, T)
    wifi_only = fs.Env(np.random.default_rng(seed + 1), 1.0, None, probe.wifi) if probe.wifi else fs.Env(rng, 1.0)
    if probe.wifi: wifi_only.wser = probe.wser[:wifi_only.ns]
    c0 = fs.initial_channel(wifi_only, rng)
    env = fs.make_env(scn, np.random.default_rng(seed), c0, T)
    seq = np.random.default_rng(seed + 7).permutation(fs.N)
    st = fs.run_static(env, seq, np.random.default_rng(seed + 2), T, 0.0)
    log = []; fs.run_cognitive(env, c0, np.random.default_rng(seed + 3), clf(), T, 0.0, log=log)
    return env, st, log

def make_bins(env, st, log, T):
    nb = int(T / BIN); B = [dict(t=round((i + 1) * BIN, 3), cog_ch=0, stat_ch=0, sent_cog=0, recv_cog=0, sent_stat=0,
                                 recv_stat=0, p=np.zeros(3), n=0) for i in range(nb)]
    hops = []
    for e in log:
        if e[0] == "w":
            i = min(int(e[1] / BIN), nb - 1); b = B[i]
            b["cog_ch"] = e[2]; b["p"] += e[3]; b["n"] += 1
            b["sent_cog"] += len(e[4]); b["recv_cog"] += int(sum(e[4]))
        else:
            det = (e[1] - env.jam["t0"]) * 1000 if env.jam and not hops and e[1] >= env.jam["t0"] else None
            hops.append(dict(t=round(e[1], 3), from_ch=int(e[2]), to_ch=int(e[3]), detect_ms=det, delay_ms=e[4] * 1000))
    bi = np.minimum((st["t"] / BIN).astype(int), nb - 1)
    for i in range(nb):
        m = bi == i; B[i]["sent_stat"] = int(m.sum()); B[i]["recv_stat"] = int(st["ok"][m].sum())
        if m.any(): B[i]["stat_ch"] = int(st["ch"][m][-1])
    for b in B:
        p = b["p"] / max(b["n"], 1); b["state"] = int(np.argmax(p)); b["p_jam"] = float(p[2])
        b["cog_ch"] = int(b["cog_ch"])
    return B, hops

_stop = threading.Event(); _thread = None

def start(scn="S1", speed=1.0, T=20.0, seed=None):
    global _thread
    _stop.set()
    if _thread and _thread.is_alive(): _thread.join(timeout=2)
    _stop.clear()
    seed = int(seed if seed is not None else time.time()) % 100000
    env, st, log = simulate(scn, seed, T)
    B, hops = make_bins(env, st, log, T)
    run_id = db.new_run(SCENARIOS.get(scn, scn), "replay", T, env.jam)
    def work():
        hi = 0
        for b in B:
            if _stop.is_set(): db.set_status(run_id, "stopped"); return
            db.add_bin(run_id, b)
            while hi < len(hops) and hops[hi]["t"] <= b["t"]:
                db.add_hop(run_id, hops[hi]); hi += 1
            time.sleep(BIN / max(speed, 0.1))
        db.set_status(run_id, "finished")
    _thread = threading.Thread(target=work, daemon=True); _thread.start()
    return run_id
