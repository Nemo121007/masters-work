"""Проверка MPCT блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
# Облегчённый режим: проверка правильности, а не качества — меньше данных test.csv и кандидатов порога.
assert shell.run_cell(r"""
_threshold_candidates_full = threshold_candidates
def threshold_candidates(scores, n_quantiles=20, n_log=10):
    return _threshold_candidates_full(scores, n_quantiles, n_log)
for _bundle in (df1, df2):
    _bundle["train"] = _bundle["train"].iloc[-2000:].reset_index(drop=True)
    _bundle["test"] = _bundle["test"].iloc[:1000].reset_index(drop=True)
""").success
started = time.perf_counter()
result = shell.run_cell("".join(cells["mpct-code-cell"]["source"]))
print(f"######## mpct-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import itertools, warnings; warnings.filterwarnings("ignore")
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
def valid(xy, t, idx, v):
    return all(t[b] - t[a] > TIME_GAP_SEC or np.hypot(*(xy[b] - xy[a])) / (t[b] - t[a]) <= v for a, b in zip(idx[:-1], idx[1:]))

r = np.random.default_rng(0); worst = 0; bad_valid = 0
for trial in range(150):
    n = int(r.integers(4, 13)); t = np.cumsum(r.choice([10.0, 10.0, 60.0, 200.0], n)); xy = np.cumsum(r.normal(0, 60, (n, 2)), axis=0)
    v = float(r.uniform(2, 10))
    chosen = mpct_select_points(xy, t, np.array([v]))[0]; idx = np.flatnonzero(chosen)
    bad_valid += not valid(xy, t, idx, v)
    opt = max(k for k in range(1, n + 1) if any(valid(xy, t, list(c), v) for c in itertools.combinations(range(n), k)))
    worst = max(worst, opt - len(idx))
check("ДП = полный перебор (150 случайных отрезков до 12 точек)", worst == 0 and bad_valid == 0, f"недобор {worst}, недопустимых решений {bad_valid}")

xy, t = simulate(2000, 0.02, 5.0, 1, gaps=False)
f = frame_of(xy, t, "move")
grid = mpct_speed_grid(f)
check("сетка v_+: 30 значений от 95-го процентиля", len(grid) == 30 and np.all(np.diff(grid) > 0), f"{grid[0]:.2f}–{grid[-1]:.2f} м/с")
v = float(grid[5])
rej, sc = mpct_detect(f, np.array([v]))
print(f"штатный ход: отвергнуто {int(rej[0].sum())} из {len(f)} при v_+ {v:.2f}")
xy1 = xy.copy(); xy1[0] += 3000.0
rej1, _ = mpct_detect(frame_of(xy1, t, "move"), np.array([v]))
check("ловушка якоря: аномальная первая точка не уводит решение", rej1[0, 0] and int(rej1[0, 1:].sum()) <= int(rej[0, 1:].sum()) + 1,
      f"отвергнуто после первой {int(rej1[0, 1:].sum())} (без выброса {int(rej[0, 1:].sum())})")
xy2 = xy.copy(); xy2[1000] += 800.0
rej2, _ = mpct_detect(frame_of(xy2, t, "move"), np.array([v]))
check("выброс отвергнут без следа на соседях", rej2[0, 1000] and not rej2[0, 1001:1005].any(), f"отвергнуто в 995–1009: {np.flatnonzero(rej2[0, 995:1010]) + 995}")
xy3 = xy.copy(); xy3[1200:1210] += 2000.0
rej3, _ = mpct_detect(frame_of(xy3, t, "move"), np.array([v]))
check("смещённый эпизод 100 с отвергнут целиком, соседи приняты", rej3[0, 1200:1210].all() and not rej3[0, 1195:1200].any() and not rej3[0, 1210:1215].any())
xy4 = xy.copy(); xy4[1300:1330] += 2000.0
rej4, _ = mpct_detect(frame_of(xy4, t, "move"), np.array([v]))
print(f"смещённый эпизод 300 с (30 точек): отвергнуто внутри {int(rej4[0, 1300:1330].sum())}, снаружи рядом {int(rej4[0, 1280:1300].sum() + rej4[0, 1330:1350].sum())} (свойство модели)")
t5 = t.copy(); t5[1500] = t5[1499]
_, sc5 = mpct_detect(frame_of(xy, t5, "move"), np.array([v]))
check("совпадающее время — без решения", not sc5[1500] and sc5[1499] and sc5[1501])
t6 = t.copy(); t6[1700:] += 600.0; xy6 = xy.copy(); xy6[1700:] += 50_000.0
rej6, _ = mpct_detect(frame_of(xy6, t6, "move"), np.array([v]))
check("разрыв по времени: после паузы цепочка заново", not rej6[0, 1700:1710].any(), f"отвергнуто после разрыва {int(rej6[0, 1700:1710].sum())}")
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
rej_rot, _ = mpct_detect(frame_of(xy2 @ rot.T, t, "move"), np.array([v]))
check("не зависит от курса (поворот траектории)", np.array_equal(rej_rot, rej2))

xy9, t9 = simulate(2500, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
sel = select_mpct_speed(f9, MCC_BAND)
r9, sc9 = mpct_detect(f9, np.array([sel["tau"]])); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): v_+ {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc9))} FP {int(np.sum(p & ~truth & sc9))} FN {int(np.sum(~p & truth & sc9))}")
''')
