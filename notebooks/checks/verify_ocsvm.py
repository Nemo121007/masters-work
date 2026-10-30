"""Проверка One-Class SVM блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
result = shell.run_cell("".join(cells["ocsvm-code-cell"]["source"]))
print(f"######## ocsvm-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
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
model = fit_ocsvm_on_train(f)
T = model["points"]; svm = model["svm"]
gamma = 1.0 / (T.shape[1] * T.var())
check("γ = 1/(d·Var(T))", np.isclose(svm._gamma, gamma), f"γ {svm._gamma:.4g}")
probe = np.vstack([T[:100], T[:100] * 8.0])
kern = np.exp(-gamma * ((probe[:, None, :] - svm.support_vectors_[None, :, :]) ** 2).sum(axis=2))
ref = -(kern @ svm.dual_coef_[0] + svm.intercept_[0])
own = -svm.decision_function(probe)
check("оценка = -(Σ α_j K(u_j, u) - ρ)", np.allclose(own, ref, atol=1e-9), f"max разница {np.max(np.abs(own - ref)):.1e}")
s0 = -svm.decision_function(T)
check("доля T вне области (s > 0) ≤ ν, опорных векторов ≥ ν", np.mean(s0 > 0) <= OCSVM_NU + 0.01 and len(svm.support_) >= OCSVM_NU * len(T) - 1,
      f"вне области {np.mean(s0 > 0):.3f}, опорных {len(svm.support_) / len(T):.3f}")
check("далёкие точки вне области", np.median(own[100:]) > 0, f"медиана оценки далёких {np.median(own[100:]):.3f}, штатных {np.median(own[:100]):.3f}")

s, r, first = ocsvm_detect(f, model, np.array([np.inf]))
check("первые 2 точки без оценки", np.all(np.isnan(s[0, :2])) and not np.isnan(s[0, 2]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _, _ = ocsvm_detect(frame_of(xy, t3, "move"), model, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _, _ = ocsvm_detect(frame_of(xy, t4, "move"), model, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_rot, _, _ = ocsvm_detect(frame_of(xy @ rot.T, t, "move"), model, np.array([np.inf]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s, equal_nan=True, atol=1e-9))

tau = float(np.nanquantile(first, 0.995))
xy5 = xy.copy(); xy5[1500] += 800.0
f5 = frame_of(xy5, t, "move")
s5, r5, first5 = ocsvm_detect(f5, model, np.array([tau]))
check("выброс: первый проход — след на следующих точках", first5[1500] > tau and first5[1501] > tau and first5[1502] > tau,
      f"оценки {np.round(first5[1500:1503], 3)} при пороге {tau:.3f}")
s5b, r5b, _ = ocsvm_detect(f, model, np.array([tau]))
check("выброс: после второго прохода следующие точки не отвергнуты", r5[0, 1500] and not r5[0, 1501:1504].any(),
      f"отвергнуто в 1495–1519: {np.flatnonzero(r5[0, 1495:1520]) + 1495}, без выброса: {np.flatnonzero(r5b[0, 1495:1520]) + 1495}")
idx, fst = dbscan_segments(f5, np.ones(len(f5), dtype=bool)); pos = np.arange(len(idx))
acc = ~(first5[idx] > tau); last = np.maximum.accumulate(np.where(acc, pos, -1))
feat2 = apply_robust(dbscan_features(xy5[idx], t[idx], np.where(fst, -1, np.r_[-1, last[:-1]])), model["center"], model["scale"])
ok2 = ~np.isnan(feat2[:, 0]); full = np.full(len(f5), np.nan); full[idx[ok2]] = -svm.decision_function(feat2[ok2])
check("второй проход: пересчёт только изменённых точек = полный пересчёт", np.allclose(full, s5[0], equal_nan=True, atol=1e-12))

xy9, t9 = simulate(5000, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
m9 = fit_ocsvm_on_train(f9); sel = select_ocsvm_threshold(f9, m9, MCC_BAND)
s9, r9, _ = ocsvm_detect(f9, m9, np.array([sel["tau"]])); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): порог {sel['tau']:.3f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
