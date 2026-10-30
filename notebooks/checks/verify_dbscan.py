"""Проверка DBSCAN блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
result = shell.run_cell("".join(cells["dbscan-code-cell"]["source"]))
print(f"######## dbscan-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN
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

xy, t = simulate(3000, 0.02, 5.0, 1)
f = frame_of(xy, t, "move")
idx, first = dbscan_segments(f, np.ones(len(f), dtype=bool))
feat = dbscan_features(xy[idx], t[idx], np.where(first, -1, np.arange(len(idx)) - 1))
ok, worst = True, 0.0
for i in (5, 100, 1234, 2999):
    s1, s0 = xy[i] - xy[i - 1], xy[i - 1] - xy[i - 2]
    v1, v0 = np.hypot(*s1) / (t[i] - t[i - 1]), np.hypot(*s0) / (t[i - 1] - t[i - 2])
    d = np.arctan2(s1[0], s1[1]) - np.arctan2(s0[0], s0[1]); d = (d + np.pi) % (2 * np.pi) - np.pi
    ref = np.array([v1, (v1 - v0) / (t[i] - t[i - 1]), v1 * d / (t[i] - t[i - 1])])
    worst = max(worst, float(np.max(np.abs(feat[i] - ref))))
check("признаки = эталон (v, a_long, a_lat)", worst < 1e-9, f"max разница {worst:.1e}")

model = fit_dbscan_on_train(f)
eps = float(np.quantile(model["k_distance"], 0.9))
db = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit(model["points"])
core_ref = np.zeros(len(model["points"]), dtype=bool); core_ref[db.core_sample_indices_] = True
check("ядровые точки = sklearn DBSCAN", np.array_equal(core_ref, model["k_distance"] <= eps), f"ядровых {int(core_ref.sum())} из {len(core_ref)}")
core = model["points"][core_ref]
dist = NearestNeighbors(n_neighbors=1).fit(core).kneighbors(model["points"])[0][:, 0]
check("шум sklearn DBSCAN = дальше ε от ядровой точки", np.array_equal(db.labels_ == -1, dist > eps), f"шум {int(np.sum(db.labels_ == -1))}")

s, r = dbscan_detect(f, model, np.array([eps]))
check("первые 2 точки без оценки", np.all(np.isnan(s[0, :2])) and not np.isnan(s[0, 2]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _ = dbscan_detect(frame_of(xy, t3, "move"), model, np.array([eps]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _ = dbscan_detect(frame_of(xy, t4, "move"), model, np.array([eps]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))

angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_rot, _ = dbscan_detect(frame_of(xy @ rot.T, t, "move"), model, np.array([eps]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s, equal_nan=True, atol=1e-9))

xy5 = xy.copy(); xy5[1500] += 800.0
f5 = frame_of(xy5, t, "move")
s5, r5 = dbscan_detect(f5, model, np.array([eps]))
i5, fi5 = dbscan_segments(f5, np.ones(len(f5), dtype=bool))
u5 = apply_robust(dbscan_features(xy5[i5], t[i5], np.where(fi5, -1, np.arange(len(i5)) - 1)), model["center"], model["scale"])
d5 = NearestNeighbors(n_neighbors=1).fit(core).kneighbors(u5[1500:1503])[0][:, 0]
check("выброс: первый проход — след на следующих точках", d5[0] > eps and d5[1] > eps, f"расстояния {np.round(d5, 1)} при ε {eps:.2f}")
check("выброс: после второго прохода следующие точки не отвергнуты", r5[0, 1500] and not r5[0, 1501:1504].any(),
      f"отвергнуто в 1495–1519: {np.flatnonzero(r5[0, 1495:1520]) + 1495}, без выброса: {np.flatnonzero(r[0, 1495:1520]) + 1495}")

xy9, t9 = simulate(5000, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
m9 = fit_dbscan_on_train(f9); sel = select_dbscan_eps(f9, m9, MCC_BAND)
s9, r9 = dbscan_detect(f9, m9, np.array([sel["tau"]])); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): ε {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
