"""
Cognitive FHSS with out-of-band LoRa control - Python simulator
Machakos University - EEE research proposal (Emmanuel Kipngeno Cheruiyot)

What it does
  1. Generates synthetic spectrum windows (carrier-detect samples + wideband power +
     packet statistics) for clear / interfered / jammed channels and trains a small MLP
     (stand-in for the Edge Impulse model). Exports the dataset as CSV for Edge Impulse.
  2. Simulates a static FHSS link and the cognitive link (detect -> sweep -> LoRa hop
     command -> ACK -> hop) under constant, pulsed and sweeping jammers and Wi-Fi.
  3. Reports PLR, detection latency, hop execution delay, LoRa control reliability.

All radio behaviour is MODELLED, not measured. Every assumption is in the CONFIG block.
Run:  python3 fhss_sim.py
"""
import numpy as np, time, json, sys
from collections import deque
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:      # plots are only needed when running this file directly
    plt = None

# ----------------------------------------------------------------- CONFIG
N = 32                  # candidate channels (2 MHz spacing)
DT = 0.25e-3            # sample period of carrier-detect / ADC sampling (s)
M = 16                  # samples per 4 ms sensing window
WIN = M * DT            # 4 ms
PKT_STEP = 8            # one payload packet every 8 samples = 2 ms (500 pkt/s)
HOP_DWELL = 10e-3       # static FHSS dwell time
SWEEP_PER_CH = 0.4e-3   # sensing radio: settle + a few carrier-detect reads per channel
SWEEP_SAMPLES = 4       # carrier-detect reads per channel during a sweep
T_LORA = 0.036          # LoRa SF7, BW125, 8-byte frame time on air (s)
LORA_TIMEOUT = 0.12     # wait before repeating a command (s)
LORA_RETRIES = 4        # total attempts
P_LORA_FRAME_LOSS = 0.02  # ASSUMED LoRa frame loss probability (to be measured on hardware)
FALLBACK_TIME = 5.0     # fallback if control link fails completely (s)
BLACKLIST_TIME = 30.0   # a jammed channel is avoided for this long (s)
SIG_DBM = -50.0         # wanted signal level at the receiver (own link)
RPD_THRESH = -64.0      # nRF24L01+ carrier detect threshold (dBm)
P_BASE_LOSS = 0.005     # background packet loss
P_WIFI_LOSS = 0.5       # packet loss while Wi-Fi burst active on the channel
TAU = 0.5               # jam probability threshold
N_NEEDED, N_WINDOWS = 2, 3   # declare jamming if 2 of last 3 windows flagged
JAM_START = 2.0         # jammer switches on at t = 2 s
CLASSES = ["clear", "interfered", "jammed"]
FEATURES = ["occupancy", "transitions", "mean_busy_run", "power_mean",
            "power_std", "power_max", "since_good_pkt", "crc_fail_last4"]

# ----------------------------------------------------------------- environment
def markov_series(rng, ns, duty, mean_on=4):
    mean_off = mean_on * (1 - duty) / duty
    out = np.zeros(ns, bool); i = 0; state = rng.random() < duty
    while i < ns:
        L = rng.geometric(1.0 / (mean_on if state else mean_off))
        out[i:i + L] = state; i += L; state = not state
    return out

class Env:
    def __init__(self, rng, T, jam=None, wifi=None):
        self.ns = int(T / DT); self.jam = jam; self.wifi = wifi
        if wifi: self.wser = markov_series(rng, self.ns, wifi["duty"])
        self.jp = jam["p"] if jam else -200.0
        self.wp = wifi["p"] if wifi else -200.0
        self.pj = 1.0 / (1.0 + np.exp(-((self.jp - SIG_DBM) - 3.0) / 2.0))  # loss prob vs JSR
    def _c(self, idx): return np.minimum(np.asarray(idx), self.ns - 1)
    def jam_active(self, ch, idx):
        idx = self._c(idx); j = self.jam
        if j is None: return np.zeros(idx.shape, bool)
        t = idx * DT; on = t >= j["t0"]; ch = np.asarray(ch)
        if j["type"] == "constant": return on & np.isin(ch, j["chans"])
        if j["type"] == "pulsed":
            ph = ((t - j["t0"]) % j["period"]) < j["duty"] * j["period"]
            return on & ph & np.isin(ch, j["chans"])
        if j["type"] == "sweep":
            k = (np.maximum(t - j["t0"], 0) // j["dwell"]).astype(int) % len(j["chans"])
            return on & (np.asarray(j["chans"])[k] == ch)
    def jam_any(self, idx):
        idx = self._c(idx); j = self.jam
        if j is None: return np.zeros(idx.shape, bool)
        t = idx * DT; on = t >= j["t0"]
        if j["type"] == "pulsed":
            return on & (((t - j["t0"]) % j["period"]) < j["duty"] * j["period"])
        return on
    def wifi_active(self, ch, idx):
        idx = self._c(idx)
        if self.wifi is None: return np.zeros(idx.shape, bool)
        return self.wser[idx] & np.isin(np.asarray(ch), self.wifi["chans"])
    def wifi_any(self, idx):
        idx = self._c(idx)
        return self.wser[idx] if self.wifi else np.zeros(idx.shape, bool)

def block(start, length): return np.arange(start, start + length) % N

# ----------------------------------------------------------------- sensing window
def window(env, ch, i0, rng, own, hist):
    """Measure one 4 ms window on channel ch. Returns features, label, packet outcomes."""
    idx = np.arange(i0, i0 + M)
    j = env.jam_active(ch, idx); w = env.wifi_active(ch, idx)
    ownm = ((idx % PKT_STEP) < 2) if own else np.zeros(M, bool)
    rpd = (j & (env.jp > RPD_THRESH)) | (w & (env.wp > RPD_THRESH)) | ownm | (rng.random(M) < 0.005)
    lin = (1e-6 + ownm * 10 ** (SIG_DBM / 10) + env.jam_any(idx) * 10 ** (env.jp / 10)
           + env.wifi_any(idx) * 10 ** (env.wp / 10))
    pw = np.clip(10 * np.log10(lin) + rng.normal(0, 1.5, M), -55, -5)
    pwn = (pw + 55) / 50
    r = rpd.astype(np.int8)
    trans = np.abs(np.diff(r)).sum() / (M - 1)
    d = np.diff(np.concatenate(([0], r, [0])))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    mbr = (en - st).mean() / M if len(st) else 0.0
    # packet outcomes for the two packets in this window
    res = []
    for off in (0, PKT_STEP):
        p = P_BASE_LOSS
        if j[off]: p = max(p, env.pj)
        if w[off]: p = max(p, P_WIFI_LOSS)
        ok = rng.random() >= p
        res.append(ok)
    h = list(hist)
    since = 0
    for v in reversed(h):
        if v: break
        since += 1
    crc = sum(1 for v in h[-4:] if not v)
    feats = [r.mean(), trans, mbr, pwn.mean(), pwn.std(), pwn.max(), since / 8.0, crc / 4.0]
    jam_hits = int((j & (env.jp > RPD_THRESH)).sum())
    label = 2 if jam_hits >= 4 else (1 if (w & (env.wp > RPD_THRESH)).sum() >= 2 else 0)
    return feats, label, res

# ----------------------------------------------------------------- training data + classifier
def random_cfg(rng, c, T, t0):
    jt = rng.choice(["none", "constant", "pulsed", "sweep"])
    jam = None
    if jt != "none":
        start = (c - rng.integers(0, 8)) if rng.random() < 0.6 else rng.integers(0, N)
        chans = block(start, 8)
        jam = dict(type=jt, chans=chans, t0=t0, p=rng.uniform(-58, -15), duty=rng.uniform(0.25, 0.9),
                   period=0.02, dwell=float(rng.choice([0.005, 0.02, 0.1])))
    wifi = None
    if rng.random() < 0.5:
        wifi = dict(chans=block(rng.integers(0, N), 10), duty=rng.uniform(0.1, 0.7), p=rng.uniform(-55, -40))
    return Env(rng, T, jam, wifi)

def gen_training(rng, nruns=160, T=5.0):
    X, y, g = [], [], []
    for run in range(nruns):
        c = int(rng.integers(N)); env = random_cfg(rng, c, T, 0.5)
        hist = deque([1] * 8, 8)
        for w in range(int(T / WIN)):
            f, lab, res = window(env, c, w * M, rng, True, hist)
            X.append(f); y.append(lab); g.append(run)
            for ok in res: hist.append(1 if ok else 0)
    return np.array(X), np.array(y), np.array(g)

class TinyMLP:
    """Forward pass of the trained MLP in plain numpy (fast inside the simulation)."""
    def __init__(self, sk, scaler):
        self.mu, self.sd = scaler.mean_, scaler.scale_
        self.W = [c.copy() for c in sk.coefs_]; self.b = [b.copy() for b in sk.intercepts_]
    def proba(self, f):
        a = (np.asarray(f) - self.mu) / self.sd
        a = np.maximum(a @ self.W[0] + self.b[0], 0)
        a = np.maximum(a @ self.W[1] + self.b[1], 0)
        z = a @ self.W[2] + self.b[2]; e = np.exp(z - z.max()); return e / e.sum()

# ----------------------------------------------------------------- links
def lora_exchange(t_start, rng, pl):
    t = t_start; tx_t = None
    for _ in range(LORA_RETRIES):
        cmd_ok = rng.random() >= pl
        if cmd_ok and tx_t is None: tx_t = t + T_LORA
        if cmd_ok and rng.random() >= pl: return tx_t, t + 2 * T_LORA, True
        t += LORA_TIMEOUT
    return tx_t, None, False

def run_static(env, seq, rng, T, t_meas):
    pk = np.arange(0, env.ns, PKT_STEP); t = pk * DT; keep = t >= t_meas
    pk, t = pk[keep], t[keep]
    ch = seq[(t // HOP_DWELL).astype(int) % N]
    j = env.jam_active(ch, pk); w = env.wifi_active(ch, pk)
    p = np.full(len(pk), P_BASE_LOSS)
    p = np.where(j, np.maximum(p, env.pj), p); p = np.where(w, np.maximum(p, P_WIFI_LOSS), p)
    ok = rng.random(len(p)) >= p
    return dict(sent=len(pk), recv=int(ok.sum()), plr=1 - ok.mean(), t=t, ch=ch, ok=ok)

def sweep_choose(env, t_dec, cur, black, rng):
    best, bestq = [], -1
    for k in range(N):
        if k == cur or black.get(k, 0) > t_dec: continue
        i0 = int((t_dec + k * SWEEP_PER_CH) / DT)
        idx = i0 + np.arange(SWEEP_SAMPLES)
        busy = ((env.jam_active(k, idx) & (env.jp > RPD_THRESH)) |
                (env.wifi_active(k, idx) & (env.wp > RPD_THRESH)) | (rng.random(SWEEP_SAMPLES) < 0.005))
        q = 1 - busy.mean()
        if q > bestq + 1e-9: best, bestq = [k], q
        elif abs(q - bestq) < 1e-9: best.append(k)
    return int(rng.choice(best)), N * SWEEP_PER_CH

def run_cognitive(env, c0, rng, clf, T, t_meas, trace=False, pl=P_LORA_FRAME_LOSS, use_clf=True, log=None):
    nw = int(T / WIN) - 2
    ch_tx = ch_rx = c0
    hist = deque([1] * 8, 8); flags = deque([0] * N_WINDOWS, N_WINDOWS)
    events = []; pending = False; black = {}
    sent = recv = fallbacks = 0
    det_lat = None; hop_delays = []; await_t = None; last_dec = 0.0; false_alarms = 0; hops = 0
    tr = [] if trace else None
    for w in range(nw):
        t0 = w * WIN
        for ev in [e for e in events if e[0] <= t0]:
            if ev[1] == "tx": ch_tx = ev[2]
            else:
                ch_rx = ev[2]; pending = False; hist = deque([1] * 8, 8); await_t = last_dec
                flags = deque([0] * N_WINDOWS, N_WINDOWS)
            events.remove(ev)
        f, lab, res = window(env, ch_rx, w * M, rng, ch_tx == ch_rx, hist)
        gots = []
        for n, ok in enumerate(res):
            tp = t0 + n * PKT_STEP * DT
            got = ok and ch_tx == ch_rx
            gots.append(bool(got))
            hist.append(1 if got else 0)
            if tp >= t_meas:
                sent += 1; recv += int(got)
            if got and await_t is not None:
                hop_delays.append(tp - await_t); await_t = None
        if trace: tr.append((t0, ch_rx))
        p = clf.proba(f) if use_clf else None
        if log is not None: log.append(("w", t0, ch_rx, p if use_clf else np.array([1.0, 0, 0]), gots))
        flagged = (p[2] >= TAU) if use_clf else (f[0] > 0.6)
        flags.append(int(flagged))
        if not pending and sum(flags) >= N_NEEDED:
            t_dec = t0 + WIN
            if env.jam is not None and t_dec >= env.jam["t0"] and det_lat is None:
                det_lat = t_dec - env.jam["t0"]
            if env.jam is None or t_dec < env.jam["t0"] or ch_rx not in env.jam["chans"]:
                false_alarms += 1
            black[ch_rx] = t_dec + BLACKLIST_TIME
            k, sw = sweep_choose(env, t_dec, ch_rx, black, rng)
            tx_t, rx_t, ok = lora_exchange(t_dec + sw, rng, pl)
            if not ok:
                fallbacks += 1; rx_t = t_dec + sw + FALLBACK_TIME
                if tx_t is None: tx_t = rx_t
            events += [(tx_t, "tx", k), (rx_t, "rx", k)]
            pending = True; last_dec = t_dec; hops += 1
            if log is not None: log.append(("h", t_dec, ch_rx, k, rx_t - t_dec))
    plr = 1 - recv / sent if sent else 0.0
    return dict(sent=sent, recv=recv, plr=plr, det_lat=det_lat, hop_delays=hop_delays,
                fallbacks=fallbacks, false_alarms=false_alarms, hops=hops, trace=tr)

# ----------------------------------------------------------------- scenarios
def make_env(scn, rng, c0, T, jsr=20.0):
    jp = SIG_DBM + jsr
    J = block(c0 - int(rng.integers(0, 8)), 8)
    wifi = dict(chans=block(int(rng.integers(0, N)), 10), duty=0.4, p=-45.0)
    if scn == "S0": return Env(rng, T)
    if scn == "S1": return Env(rng, T, dict(type="constant", chans=J, t0=JAM_START, p=jp))
    if scn == "S2": return Env(rng, T, dict(type="pulsed", chans=J, t0=JAM_START, p=jp, duty=0.5, period=0.02))
    if scn == "S3": return Env(rng, T, dict(type="sweep", chans=J, t0=JAM_START, p=jp, dwell=0.02))
    if scn == "S4": return Env(rng, T, None, wifi)
    if scn == "S5": return Env(rng, T, dict(type="constant", chans=J, t0=JAM_START, p=jp), wifi)

SCN_NAMES = {"S0": "No jamming", "S1": "Constant jammer", "S2": "Pulsed jammer (50%)",
             "S3": "Sweeping jammer", "S4": "Wi-Fi only", "S5": "Wi-Fi + constant"}

def initial_channel(env_wifi_only, rng):
    c, _ = sweep_choose(env_wifi_only, 0.0, -1, {}, rng); return c

def one_pair(scn, seed, clf, T, jsr=20.0, trace=False):
    rng = np.random.default_rng(seed)
    # initial sweep sees only background (Wi-Fi) because the jammer is not yet on
    probe = make_env(scn, np.random.default_rng(seed), 0, T, jsr)
    wifi_only = Env(np.random.default_rng(seed + 1), 1.0, None, probe.wifi) if probe.wifi else Env(rng, 1.0)
    wifi_only.wser = probe.wser[:wifi_only.ns] if probe.wifi else None
    c0 = initial_channel(wifi_only, rng)
    env = make_env(scn, np.random.default_rng(seed), c0, T, jsr)
    seq = np.random.default_rng(seed + 7).permutation(N)
    st = run_static(env, seq, np.random.default_rng(seed + 2), T, JAM_START)
    cg = run_cognitive(env, c0, np.random.default_rng(seed + 3), clf, T, JAM_START, trace=trace)
    return st, cg, env

def ci95(x):
    x = np.asarray(x, float); n = len(x)
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(n)) if n > 1 else (x.mean(), 0.0)

# ----------------------------------------------------------------- main
def main():
    t00 = time.time(); out = "/mnt/user-data/outputs/"
    rng = np.random.default_rng(2026)
    print("Generating training data ..."); X, y, g = gen_training(rng)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report
    runs = np.unique(g); rng.shuffle(runs)
    tr_r, va_r, te_r = runs[:int(.7 * len(runs))], runs[int(.7 * len(runs)):int(.85 * len(runs))], runs[int(.85 * len(runs)):]
    sel = lambda rr: np.isin(g, rr)
    # balance classes in the training set
    def balance(Xs, ys, cap=6000):
        keep = []
        for c in range(3):
            ii = np.where(ys == c)[0]; keep += list(rng.choice(ii, min(cap, len(ii)), replace=False))
        keep = np.array(keep); return Xs[keep], ys[keep], keep
    Xtr, ytr, _ = balance(X[sel(tr_r)], y[sel(tr_r)])
    scaler = StandardScaler().fit(Xtr)
    sk = MLPClassifier((16, 8), max_iter=400, random_state=1, early_stopping=True).fit(scaler.transform(Xtr), ytr)
    clf = TinyMLP(sk, scaler)
    Xte, yte = X[sel(te_r)], y[sel(te_r)]
    pred = sk.predict(scaler.transform(Xte))
    acc = (pred == yte).mean(); cm = confusion_matrix(yte, pred)
    far = ((pred == 2) & (yte != 2)).sum() / max((yte != 2).sum(), 1)
    miss = ((pred != 2) & (yte == 2)).sum() / max((yte == 2).sum(), 1)
    print(f"Classifier test accuracy {acc:.3f}, false alarm {far:.3f}, miss {miss:.3f}")
    print(classification_report(yte, pred, target_names=CLASSES, digits=3))
    nparams = sum(c.size for c in sk.coefs_) + sum(b.size for b in sk.intercepts_)
    # export dataset for Edge Impulse (balanced, session-wise split column)
    import csv
    Xb, yb, kb = balance(X, y, cap=8000); gb = g[kb]
    split = np.where(np.isin(gb, te_r), "testing", "training")
    with open(out + "spectrum_dataset_for_edge_impulse.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["timestamp"] + FEATURES + ["label", "split"])
        for n, (f, l, s) in enumerate(zip(Xb, yb, split)):
            wr.writerow([n * 4] + [f"{v:.5f}" for v in f] + [CLASSES[l], s])

    # ---- scenario comparison
    NR, T = 10, 60.0
    results = {}
    for scn in SCN_NAMES:
        s_plr, c_plr, lat, hd, fa, fb = [], [], [], [], 0, 0
        for r in range(NR):
            st, cg, _ = one_pair(scn, 1000 * int(scn[1]) + r, clf, T)
            s_plr.append(st["plr"]); c_plr.append(cg["plr"])
            if cg["det_lat"] is not None and scn in ("S1", "S2", "S3", "S5"): lat.append(cg["det_lat"])
            hd += cg["hop_delays"]; fa += cg["false_alarms"]; fb += cg["fallbacks"]
        results[scn] = dict(static=ci95(s_plr), cog=ci95(c_plr), lat=lat, hd=hd, fa=fa, fb=fb,
                            s_raw=s_plr, c_raw=c_plr)
        print(scn, SCN_NAMES[scn], "static %.2f%% | cognitive %.2f%%" %
              (100 * results[scn]["static"][0], 100 * results[scn]["cog"][0]), f"({time.time()-t00:.0f}s)")
        sys.stdout.flush()

    # ---- jammer power sweep (constant jammer)
    jsrs = [-10, -5, 0, 5, 10, 15, 20, 30]; sweep = []
    for jsr in jsrs:
        a, b = [], []
        for r in range(6):
            st, cg, _ = one_pair("S1", 5000 + r, clf, 30.0, jsr=jsr); a.append(st["plr"]); b.append(cg["plr"])
        sweep.append((ci95(a), ci95(b)))
    print("JSR sweep done", f"({time.time()-t00:.0f}s)")

    # ---- example trace for the channel-time map
    st, cg, env = one_pair("S1", 1001, clf, 10.0, trace=True)
    rngx = np.random.default_rng(1001 + 7); seq = rngx.permutation(N)

    # ---- LoRa control reliability Monte Carlo (does not depend on 2.4 GHz jamming)
    lrng = np.random.default_rng(5); lora_rows = []
    for pl in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
        okc = sum(lora_exchange(0, lrng, pl)[2] for _ in range(20000)); lora_rows.append((pl, okc / 20000))

    # ---------------- figures
    plt.rcParams.update({"font.size": 10})
    # Fig 1: PLR by scenario
    fig, ax = plt.subplots(figsize=(8, 4.2)); xs = np.arange(len(SCN_NAMES)); wd = 0.38
    sm = [100 * results[s]["static"][0] for s in SCN_NAMES]; se = [100 * results[s]["static"][1] for s in SCN_NAMES]
    cm_ = [100 * results[s]["cog"][0] for s in SCN_NAMES]; ce = [100 * results[s]["cog"][1] for s in SCN_NAMES]
    ax.bar(xs - wd / 2, sm, wd, yerr=se, capsize=3, label="Static FHSS", color="#9aa5b1")
    ax.bar(xs + wd / 2, cm_, wd, yerr=ce, capsize=3, label="Cognitive FHSS", color="#1f6feb")
    ax.axhline(5, ls="--", c="k", lw=0.8); ax.text(len(xs) - 0.5, 5.6, "5% target", ha="right", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}\n{SCN_NAMES[s]}" for s in SCN_NAMES], fontsize=8)
    ax.set_ylabel("Packet loss rate (%)"); ax.legend(); ax.set_title("PLR by scenario (mean, 95% CI, 10 runs of 60 s)")
    fig.tight_layout(); fig.savefig(out + "fig1_plr_by_scenario.png", dpi=160); plt.close(fig)
    # Fig 2: PLR vs JSR
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(jsrs, [100 * s[0][0] for s in sweep], [100 * s[0][1] for s in sweep], marker="o", capsize=3, label="Static FHSS", color="#9aa5b1")
    ax.errorbar(jsrs, [100 * s[1][0] for s in sweep], [100 * s[1][1] for s in sweep], marker="o", capsize=3, label="Cognitive FHSS", color="#1f6feb")
    ax.set_xlabel("Jammer-to-signal ratio (dB)"); ax.set_ylabel("Packet loss rate (%)"); ax.legend()
    ax.set_title("PLR against jammer power (constant jammer, 8 channels)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out + "fig2_plr_vs_jsr.png", dpi=160); plt.close(fig)
    # Fig 3: channel-time map
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    tt = np.arange(0, 6, 0.01)
    for a in axs:
        jc = np.sort(env.jam["chans"])
        for c in jc: a.fill_between([JAM_START, 6], c - .5, c + .5, color="#d1242f", alpha=.15, lw=0)
        a.set_xlim(0, 6); a.set_ylim(-.5, N - .5); a.set_xlabel("Time (s)")
    axs[0].step(tt, seq[(tt // HOP_DWELL).astype(int) % N], where="post", lw=.4, c="#57606a")
    axs[0].set_title("Static FHSS: hops through jammed channels"); axs[0].set_ylabel("Channel index")
    ct = np.array([x[0] for x in cg["trace"]]); cc = np.array([x[1] for x in cg["trace"]]); m = ct < 6
    axs[1].step(ct[m], cc[m], where="post", c="#1f6feb", lw=1.6)
    axs[1].set_title("Cognitive FHSS: leaves the jammed block"); fig.tight_layout()
    fig.savefig(out + "fig3_channel_time_map.png", dpi=160); plt.close(fig)
    # Fig 4: latency histograms
    lat = np.array(sum([results[s]["lat"] for s in ("S1", "S2", "S3", "S5")], [])) * 1000
    hd = np.array(sum([results[s]["hd"] for s in ("S1", "S2", "S3", "S5")], [])) * 1000
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].hist(lat, bins=20, color="#1f6feb"); axs[0].axvline(10, c="k", ls="--", lw=.8)
    axs[0].set_xlabel("Detection latency (ms)"); axs[0].set_ylabel("Runs"); axs[0].set_title("Jamming detection latency")
    axs[1].hist(hd[hd < 400], bins=25, color="#1f6feb"); axs[1].axvline(150, c="k", ls="--", lw=.8)
    axs[1].set_xlabel("Hop execution delay (ms)"); axs[1].set_title("Decision to first valid packet")
    fig.tight_layout(); fig.savefig(out + "fig4_latency_histograms.png", dpi=160); plt.close(fig)
    # Fig 5: confusion matrix
    fig, ax = plt.subplots(figsize=(4.6, 4)); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j2 in range(3): ax.text(j2, i, f"{cmn[i, j2]*100:.1f}%\n({cm[i, j2]})", ha="center", va="center", fontsize=8,
                                   color="white" if cmn[i, j2] > .5 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Classifier test set (accuracy {acc*100:.1f}%)")
    fig.tight_layout(); fig.savefig(out + "fig5_confusion_matrix.png", dpi=160); plt.close(fig)

    # ---------------- summary
    L = ["# Simulation results - cognitive FHSS with out-of-band LoRa control", "",
         "All values come from a MODEL of the radios (see assumptions at the end), not from measurements.", "",
         "## Packet loss rate (mean +/- 95% CI, 10 runs of 60 s, measured after the jammer starts at 2 s)", "",
         "| Scenario | Static FHSS (%) | Cognitive FHSS (%) | Hop-ups |", "|---|---|---|---|"]
    for s in SCN_NAMES:
        r = results[s]
        L.append(f"| {s} {SCN_NAMES[s]} | {100*r['static'][0]:.2f} +/- {100*r['static'][1]:.2f} | "
                 f"{100*r['cog'][0]:.2f} +/- {100*r['cog'][1]:.2f} | false alarms {r['fa']}, fallbacks {r['fb']} |")
    L += ["", "## Jammer power sweep (constant jammer on 8 channels, 6 runs of 30 s)", "", "| JSR (dB) | Static PLR (%) | Cognitive PLR (%) |", "|---|---|---|"]
    for jsr, s in zip(jsrs, sweep): L.append(f"| {jsr} | {100*s[0][0]:.2f} | {100*s[1][0]:.2f} |")
    L += ["", "## Detection latency and hop execution delay (S1, S2, S3, S5 pooled)", "",
          f"- Detection latency: mean {lat.mean():.1f} ms, median {np.median(lat):.1f} ms, max {lat.max():.1f} ms (n={len(lat)}); target < 10 ms",
          f"- Hop execution delay: mean {hd.mean():.0f} ms, median {np.median(hd):.0f} ms, 95th percentile {np.percentile(hd,95):.0f} ms (n={len(hd)}); target < 150 ms",
          "", "## Classifier (MLP 8-16-8-3, stand-in for the Edge Impulse model)", "",
          f"- Parameters: {nparams}; test accuracy {acc*100:.1f}% (sessions not seen in training)",
          f"- False alarm rate (non-jammed windows flagged jammed): {far*100:.2f}%; miss rate (jammed windows missed): {miss*100:.2f}%",
          "", "## LoRa control reliability (hop command + ACK, 4 attempts, SF7)", "", "| Frame loss probability | Command delivered (%) |", "|---|---|"]
    for pl, ok in lora_rows: L.append(f"| {pl*100:.0f}% | {ok*100:.2f} |")
    L += ["", "## Key assumptions (change in the CONFIG block of fhss_sim.py)", "",
          "- 32 channels, 2 MHz spacing; 500 packets/s; static FHSS dwell 10 ms; wanted signal -50 dBm at the receiver.",
          "- Jammer: 8-channel block containing the cognitive link's first channel; starts at t = 2 s.",
          "- nRF24 carrier detect modelled as busy when power > -64 dBm; AD8317 modelled as wideband power, -55..-5 dBm, 1.5 dB noise.",
          "- Packet loss: 0.5% background, 50% during a Wi-Fi burst, a logistic function of JSR for jammer hits.",
          f"- LoRa: SF7, 8-byte frame, 36 ms time on air, assumed frame loss {P_LORA_FRAME_LOSS*100:.0f}% (independent of 2.4 GHz jamming) - measure on hardware.",
          "- Sweep: 0.4 ms per channel with 4 carrier-detect reads; classifier decision needs 2 of the last 3 windows of 4 ms."]
    open(out + "simulation_results.md", "w").write("\n".join(L))
    print("\n".join(L)); print(f"Done in {time.time()-t00:.0f}s")

if __name__ == "__main__":
    main()
