"""Four-way-compatible metrics for KDFlow runs: 2 warm-up steps excluded, then measure until
>= 49,152 tokens (prompt+response) and >= 3 steps. Wall window = log timestamp of step-2 end -> last step end."""
import re, json, sys, glob, os, csv
from datetime import datetime
E = "/home/xxf/Distill/ai_gen/kdflow-mopd-a6000-20261118"
BUDGET, WARM, MINSTEPS = 49152, 2, 3
FW = {"orig": "Orig", "ours": "Ours", "trl": "TRL", "unsloth": "Unsloth"}
def parse(run):
    rows = []
    for l in open(f"{E}/runs/{run}/container.log", errors="ignore"):
        m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*step \[(\d+)/(\d+)\]", l)
        if not m or "timing/step_time" not in l: continue
        g = lambda k: float(re.search(re.escape(k) + r": ([0-9.\-e]+)", l).group(1))
        bs = int(re.search(r"bs(\d+)", run).group(1))
        rows.append(dict(step=int(m.group(2)), ts=datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(),
            prompt=g("rollout/prompt_length/mean") * bs, resp=g("rollout/response_length/mean") * bs,
            t=g("timing/step_time"), rollout=g("timing/rollout"), teacher=g("timing/teacher_forward"),
            train=g("timing/student_train"), sync=g("timing/rollout_weight_sync")))
    return rows
def gpu(run):
    out = []
    for r in csv.reader(open(f"{E}/runs/{run}/gpu-samples.csv")):
        try: out.append((datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(r[1]), float(r[2])))
        except Exception: pass
    return out
res = []
for d in sorted(glob.glob(f"{E}/runs/kdflow-bs*")):
    run = os.path.basename(d); bs = int(run.split("bs")[1]); rows = parse(run)
    if len(rows) <= WARM: res.append(dict(run=run, bs=bs, status="incomplete", steps=len(rows))); continue
    win, tok = [], 0
    for r in rows[WARM:]:
        win.append(r); tok += r["prompt"] + r["resp"]
        if tok >= BUDGET and len(win) >= MINSTEPS: break
    ok = tok >= BUDGET and len(win) >= MINSTEPS
    T = sum(r["t"] for r in win); wall = win[-1]["ts"] - rows[WARM - 1]["ts"]
    s = gpu(run); t0, t1 = rows[WARM - 1]["ts"], win[-1]["ts"]
    util = [u for t, m, u in s if t0 <= t <= t1]
    n = len(win)
    res.append(dict(run=run, bs=bs, status="ok" if ok else "BUDGET_NOT_REACHED", measured_steps=n, tokens=int(tok),
        resp_tokens=int(sum(r["resp"] for r in win)), tok_s=tok / T, tok_s_wall=tok / wall if wall > 0 else None,
        resp_tok_s=sum(r["resp"] for r in win) / T, s_per_step=T / n,
        rollout=sum(r["rollout"] for r in win) / n, teacher=sum(r["teacher"] for r in win) / n,
        train=sum(r["train"] for r in win) / n, sync=sum(r["sync"] for r in win) / n,
        peak_gib=max(m for _, m, _ in s) / 1024, avg_util=sum(util) / len(util) if util else None,
        exit=open(f"{d}/exit_code").read().strip() if os.path.exists(f"{d}/exit_code") else "running"))
fw = {}
try:
    for r in csv.DictReader(open("/home/xxf/Distill/ai_gen/fourway-speed-a6000-20261118/reports/summary.csv")):
        fw.setdefault(r.get("method") or r.get("run", "").split("-")[0], {})[str(r.get("bs") or r.get("batch_size"))] = r
except Exception as e: print("four-way summary not loaded:", e)
json.dump(res, open(f"{E}/summary.json", "w"), indent=1)
for r in sorted(res, key=lambda r: r["bs"]):
    if r["status"] == "incomplete": print(r); continue
    print("bs%-3d %-18s steps=%2d tok=%6d  tok/s=%6.1f (wall %6.1f) resp=%5.1f  s/step=%5.1f [roll %4.1f teach %4.1f train %4.1f sync %3.1f]  peak=%4.1f GiB util=%s %s" % (
        r["bs"], r["status"], r["measured_steps"], r["tokens"], r["tok_s"], r["tok_s_wall"] or 0, r["resp_tok_s"], r["s_per_step"],
        r["rollout"], r["teacher"], r["train"], r["sync"], r["peak_gib"], "%.0f%%" % r["avg_util"] if r["avg_util"] else "-", r["exit"]))
