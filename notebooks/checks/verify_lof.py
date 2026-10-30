"""Проверка LOF блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
    # Функции признаков DBSCAN без эксперимента DBSCAN.
    src = "".join(cells["dbscan-code-cell"]["source"])
    assert shell.run_cell(src[: src.index("dbscan_scenarios = (")]).success
started = time.perf_counter()
result = shell.run_cell("".join(cells["lof-code-cell"]["source"]))
print(f"######## lof-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
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

xy, t = simulate(3000, 0.02, 5.0, 1)
f = frame_of(xy, t, "move")
model = fit_lof_on_train(f)
T = model["points"]
for k in (10, 35):
    dens = lof_train_density(model, k)
    own = lof_scores(model, k, dens, T, np.arange(len(T)))
    ref = -LocalOutlierFactor(n_neighbors=k).fit(T).negative_outlier_factor_
    check(f"k={k}: LOF точек обучения = sklearn", np.allclose(own, ref, rtol=1e-6), f"max отн. разница {np.max(np.abs(own / ref - 1)):.1e}")
    probe = T[:200] + np.random.default_rng(5).normal(0, 0.5, (200, 3))
    new = lof_scores(model, k, dens, probe, np.full(200, -1))
    ref_new = -LocalOutlierFactor(n_neighbors=k, novelty=True).fit(T).score_samples(probe)
    check(f"k={k}: LOF новых точек = sklearn novelty", np.allclose(new, ref_new, rtol=1e-6), f"max отн. разница {np.max(np.abs(new / ref_new - 1)):.1e}")

k = 20
s, r, first = lof_detect(f, model, k, np.array([np.inf]), model["self_rows"])
check("первые 2 точки без оценки", np.all(np.isnan(s[0, :2])) and not np.isnan(s[0, 2]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _, _ = lof_detect(frame_of(xy, t3, "move"), model, k, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _, _ = lof_detect(frame_of(xy, t4, "move"), model, k, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_new, _, _ = lof_detect(f, model, k, np.array([np.inf]))
s_rot, _, _ = lof_detect(frame_of(xy @ rot.T, t, "move"), model, k, np.array([np.inf]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s_new, equal_nan=True, rtol=1e-6))

tau = float(np.nanquantile(first, 0.995))
xy5 = xy.copy(); xy5[1500] += 800.0
f5 = frame_of(xy5, t, "move")
s5, r5, first5 = lof_detect(f5, model, k, np.array([tau]))
check("выброс: первый проход — след на следующих точках", first5[1500] > tau and first5[1501] > tau and first5[1502] > tau,
      f"LOF {np.round(first5[1500:1503], 1)} при пороге {tau:.2f}")
s5b, r5b, _ = lof_detect(f, model, k, np.array([tau]))
check("выброс: после второго прохода следующие точки не отвергнуты", r5[0, 1500] and not r5[0, 1501:1504].any(),
      f"отвергнуто в 1495–1519: {np.flatnonzero(r5[0, 1495:1520]) + 1495}, без выброса: {np.flatnonzero(r5b[0, 1495:1520]) + 1495}")
idx, fst = dbscan_segments(f5, np.ones(len(f5), dtype=bool)); pos = np.arange(len(idx))
acc = ~(first5[idx] > tau); last = np.maximum.accumulate(np.where(acc, pos, -1))
feat2 = apply_robust(dbscan_features(xy5[idx], t[idx], np.where(fst, -1, np.r_[-1, last[:-1]])), model["center"], model["scale"])
ok2 = ~np.isnan(feat2[:, 0]); full = np.full(len(f5), np.nan)
full[idx[ok2]] = lof_scores(model, k, lof_train_density(model, k), feat2[ok2], np.full(int(ok2.sum()), -1))
check("второй проход: пересчёт только изменённых точек = полный пересчёт", np.allclose(full, s5[0], equal_nan=True, rtol=1e-9))

xy9, t9 = simulate(5000, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
m9 = fit_lof_on_train(f9); sel = select_lof_parameters(f9, m9, MCC_BAND)
s9, r9, _ = lof_detect(f9, m9, int(sel["k"]), np.array([sel["tau"]]), m9["self_rows"]); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): k {int(sel['k'])}, порог {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
