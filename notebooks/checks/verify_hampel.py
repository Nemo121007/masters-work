"""Проверка фильтра Хэмпела блокнота: ячейка на test.csv и синтетика с известным ответом."""
import contextlib, io, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".claude" / "skills" / "run-notebook"))
from run_notebook import _make_shell, PROJECT_ROOT
ids = ["b79e4cc6", "a1imports", "72430d98981e0032", "cdb3e6d2", "777bfdcb5c54ef", "deacacdefd3a0f35"]
cells = {c.get("id"): c for c in json.loads(Path("example_model.ipynb").read_text(encoding="utf-8"))["cells"]}
shell = _make_shell(PROJECT_ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    for cid in ids:
        assert shell.run_cell("".join(cells[cid]["source"])).success, cid
started = time.perf_counter()
result = shell.run_cell("".join(cells["hampel-code-cell"]["source"]))
print(f"######## hampel-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
def frame_of(xy, t, label):
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "x": xy[:, 0], "n": xy[:, 1], "y": label, "lat": 44.0, "lon": 132.0})
def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)
def simulate(count, sigma_acc, sigma_meas, seed, gaps=True):
    r = np.random.default_rng(seed)
    dt = np.where(r.random(count) < (0.03 if gaps else 0.0), r.uniform(20.0, 170.0, count), 10.0)
    pos = np.zeros(2); vel = np.array([5.0, 2.0]); out = []
    for d in dt:
        acc = r.normal(0.0, sigma_acc, 2)
        pos = pos + vel * d + 0.5 * acc * d * d; vel = vel + acc * d; out.append(pos.copy())
    return np.array(out) + r.normal(0.0, sigma_meas, (count, 2)), np.cumsum(dt)
def naive(xy, t, w, floor):
    # Эталон первого прохода для одного отрезка: окно — 2w+1 ближайших скоростей, у краёв сдвинуто.
    v = np.hypot(*np.diff(xy, axis=0).T) / np.diff(t)
    k = len(v); size = min(2 * w + 1, k); out = np.full(len(t), np.nan)
    for j in range(k):
        s = min(max(j - w, 0), k - size); win = v[s:s + size]; m = np.median(win)
        out[j + 1] = abs(v[j] - m) / max(1.4826 * np.median(np.abs(win - m)), floor)
    return out
W = HAMPEL_HALF_WINDOW

xy, t = simulate(800, 0.02, 5.0, 1, gaps=False)
f = frame_of(xy, t, "move"); lay = hampel_layout(f, W)
floor = fit_hampel_floor(f, lay)
_, _, first = hampel_detect(f, lay, floor, np.array([np.inf]))
ref = naive(xy, t, W, floor)
check("первый проход = эталон (окно со сдвигом у краёв)", np.allclose(first, ref, equal_nan=True), f"max разница {np.nanmax(np.abs(first - ref)):.1e}; σ_min {floor:.3f} м/с")
s_inf, r_inf, _ = hampel_detect(f, lay, floor, np.array([np.inf]))
check("без отвергнутых точек второй проход = первому", np.allclose(s_inf[0], first, equal_nan=True) and not r_inf.any())
check("первая точка без оценки", np.isnan(first[0]) and not np.isnan(first[1]))

t3 = t.copy(); t3[300] = t3[299]
_, _, f3 = hampel_detect(frame_of(xy, t3, "move"), hampel_layout(frame_of(xy, t3, "move"), W), floor, np.array([np.inf]))
check("совпадающее время", np.isnan(f3[300]) and not np.isnan(f3[301]))

t4 = t.copy(); t4[400:] += 600.0
xy4 = xy.copy(); xy4[380:400] += np.random.default_rng(9).normal(0, 300, (20, 2))
fr4 = frame_of(xy4, t4, "move"); _, _, f4 = hampel_detect(fr4, hampel_layout(fr4, W), floor, np.array([np.inf]))
fr4b = frame_of(xy, t4, "move"); _, _, f4b = hampel_detect(fr4b, hampel_layout(fr4b, W), floor, np.array([np.inf]))
check("разрыв по времени: первая точка без оценки, окно не пересекает разрыв",
      np.isnan(f4[400]) and not np.isnan(f4[401]) and np.allclose(f4[400:], f4b[400:], equal_nan=True))

t5 = np.r_[t[:5], t[5:11] + 1000.0, t[11:200] + 2000.0]
fr5 = frame_of(xy[:200], t5, "move"); _, _, f5 = hampel_detect(fr5, hampel_layout(fr5, W), floor, np.array([np.inf]))
check("короткие отрезки: 4 скорости — без оценки, 5 — с оценкой", np.all(np.isnan(f5[:5])) and np.all(~np.isnan(f5[6:11])) and np.isnan(f5[5]),
      f"с оценкой в отрезках: {int(np.sum(~np.isnan(f5[:5])))}, {int(np.sum(~np.isnan(f5[5:11])))}")

line = np.column_stack([np.arange(100) * 50.0, np.zeros(100)]); tl = np.arange(100) * 10.0
line[50, 0] += 3.0
frl = frame_of(line, tl, "move"); _, _, fl = hampel_detect(frl, hampel_layout(frl, W), floor, np.array([np.inf]))
check("MAD = 0 → оценка конечна (деление на σ_min)", np.all(np.isfinite(fl[1:])) and abs(fl[50] - 0.3 / floor) < 1e-6, f"оценка {fl[50]:.2f}, 0.3/σ_min {0.3 / floor:.2f}")

tau = 6.0
xy7 = xy.copy(); xy7[500] += 800.0
fr7 = frame_of(xy7, t, "move"); s7, r7, f7 = hampel_detect(fr7, hampel_layout(fr7, W), floor, np.array([tau]))
check("выброс: первый проход — след на следующей точке", f7[500] > tau and f7[501] > tau, f"оценки {f7[500]:.0f}, {f7[501]:.0f}")
check("выброс: после второго прохода отвергнута только точка выброса", r7[0, 500] and not r7[0, 501] and int(r7[0, 490:520].sum()) == 1,
      f"отвергнуто рядом {int(r7[0, 490:520].sum())}, оценка следующей {s7[0, 501]:.2f}")

xy8 = xy.copy(); xy8[600:640] += 2000.0
fr8 = frame_of(xy8, t, "move"); _, r8, _ = hampel_detect(fr8, hampel_layout(fr8, W), floor, np.array([tau]))
print(f"эпизод 40 точек со смещением 2 км: отвергнуто внутри {int(r8[0, 600:640].sum())}, на краях {int(r8[0, 600])}/{int(r8[0, 640])} (свойство: только вход и выход)")

xy9, t9 = simulate(5000, 0.05, 8.0, 3)
label = np.full(len(t9), "move", dtype=object)
r = np.random.default_rng(4)
for start in r.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(r.integers(1, 8))
    xy9[start:start + length] += r.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
l9 = hampel_layout(f9, W); fl9 = fit_hampel_floor(f9, l9)
sel = select_hampel_threshold(f9, l9, fl9, MCC_BAND)
s9, r9, _ = hampel_detect(f9, l9, fl9, np.array([sel["tau"]])); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): σ_min {fl9:.3f}, порог {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
