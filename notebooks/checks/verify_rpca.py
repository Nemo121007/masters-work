"""Проверка Robust PCA блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
# Облегчённый режим: проверка правильности, а не качества — меньше данных test.csv и кандидатов порога.
assert shell.run_cell(r"""
_threshold_candidates_full = threshold_candidates
def threshold_candidates(scores, n_quantiles=20, n_log=10):
    return _threshold_candidates_full(scores, n_quantiles, n_log)
for _bundle in (dataset,):
    _bundle["train"] = _bundle["train"].iloc[-2000:].reset_index(drop=True)
    _bundle["test"] = _bundle["test"].iloc[:1000].reset_index(drop=True)
""").success
started = time.perf_counter()
result = shell.run_cell("".join(cells["rpca-code-cell"]["source"]))
print(f"######## rpca-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
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
from scipy.optimize import linprog as _linprog_reference
r = np.random.default_rng(7)
true_l = r.normal(size=(600, 3)) @ r.normal(size=(3, 48))
true_s = np.zeros_like(true_l); mask = r.random(true_l.shape) < 0.03; true_s[mask] = r.normal(0, 10, mask.sum())
low, sparse, it, conv = principal_component_pursuit(true_l + true_s)
sv = np.linalg.svd(low, compute_uv=False)
check("PCP: ранг и разреженная часть восстановлены", conv and int(np.sum(sv > RPCA_RANK_TOL * sv[0])) == 3 and np.linalg.norm(low - true_l) / np.linalg.norm(true_l) < 1e-3,
      f"итераций {it}, ранг {int(np.sum(sv > RPCA_RANK_TOL * sv[0]))}, отн. ошибка L {np.linalg.norm(low - true_l) / np.linalg.norm(true_l):.1e}")

xy, t = simulate(2000, 0.02, 5.0, 1)
f = frame_of(xy, t, "move")
model = fit_rpca_on_train(f)
U = model["basis"]
print(f"синтетика: окон {model['windows']}, ранг {U.shape[1]} из {U.shape[0]}, итераций {model['iterations']}, сошлось {model['converged']}")
check("разложение на окнах штатного хода сошлось", model["converged"])
check("базис ортонормирован", np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-9))
probe = r.normal(size=(5, U.shape[0])); probe[:, 7] += 20.0
own = rpca_window_residual(U, probe)
ref = []
for x in probe:
    m, k = U.shape
    res = _linprog_reference(np.r_[np.zeros(k), np.ones(m)], A_ub=np.block([[U, -np.eye(m)], [-U, -np.eye(m)]]), b_ub=np.r_[x, -x],
                  bounds=[(None, None)] * k + [(0, None)] * m, method="highs")
    ref.append(x - U @ res.x[:k])
ref = np.array(ref)
check("остаток = точное ℓ1-решение (независимый вызов линейного программирования)", np.allclose(np.abs(own).sum(axis=1), np.abs(ref).sum(axis=1), rtol=1e-6),
      f"max отн. разница ℓ1 {np.max(np.abs(np.abs(own).sum(1) / np.abs(ref).sum(1) - 1)):.1e}")

s, rj, first = rpca_detect(f, model, np.array([np.inf]))
check("первые 2 точки без оценки", np.all(np.isnan(s[0, :2])) and not np.isnan(s[0, 2]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _, _ = rpca_detect(frame_of(xy, t3, "move"), model, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _, _ = rpca_detect(frame_of(xy, t4, "move"), model, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))
t5 = np.r_[t[:17], t[17:35] + 1000.0, t[35:200] + 2000.0]
s5s, _, _ = rpca_detect(frame_of(xy[:200], t5, "move"), model, np.array([np.inf]))
check("отрезок 17 точек — без оценки, 18 — с оценкой", np.all(np.isnan(s5s[0, :17])) and np.all(~np.isnan(s5s[0, 19:35])),
      f"с оценкой: {int(np.sum(~np.isnan(s5s[0, :17])))} из 17, {int(np.sum(~np.isnan(s5s[0, 17:35])))} из 18")
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_rot, _, _ = rpca_detect(frame_of(xy @ rot.T, t, "move"), model, np.array([np.inf]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s, equal_nan=True, atol=1e-6))

tau = float(np.nanquantile(first, 0.995))
xy6 = xy.copy(); xy6[1500] += 800.0
f6 = frame_of(xy6, t, "move")
s6, r6, first6 = rpca_detect(f6, model, np.array([tau]))
check("выброс: первый проход — след на следующих точках", first6[1500] > tau and first6[1501] > tau and first6[1502] > tau,
      f"оценки {np.round(first6[1499:1504], 1)} при пороге {tau:.2f}")
s6b, r6b, _ = rpca_detect(f, model, np.array([tau]))
check("выброс: после второго прохода следующие точки не отвергнуты", r6[0, 1500] and not r6[0, 1501:1504].any(),
      f"отвергнуто в 1490–1519: {np.flatnonzero(r6[0, 1490:1520]) + 1490}, без выброса: {np.flatnonzero(r6b[0, 1490:1520]) + 1490}")
idx, fst = dbscan_segments(f6, np.ones(len(f6), dtype=bool)); pos = np.arange(len(idx))
acc = ~(first6[idx] > tau); last = np.maximum.accumulate(np.where(acc, pos, -1))
feat2 = apply_robust(dbscan_features(xy6[idx], t[idx], np.where(fst, -1, np.r_[-1, last[:-1]])), model["center"], model["scale"])
st = rpca_window_starts(fst, ~np.isnan(feat2[:, 0]), RPCA_WINDOW); cov = st[:, None] + np.arange(RPCA_WINDOW)
res = rpca_window_residual(U, feat2[cov].reshape(len(st), -1))
nrm = np.linalg.norm(res.reshape(len(st), RPCA_WINDOW, -1), axis=2); full_pts = np.full(len(idx), -np.inf)
np.maximum.at(full_pts, cov.ravel(), nrm.ravel()); full = np.full(len(f6), np.nan); ok = np.isfinite(full_pts); full[idx[ok]] = full_pts[ok]
check("второй проход: пересчёт только изменённых окон = полный пересчёт", np.allclose(full, s6[0], equal_nan=True, atol=1e-8))

xy9, t9 = simulate(2500, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
m9 = fit_rpca_on_train(f9); fp9 = rpca_first_pass(f9, m9); sel = select_rpca_threshold(f9, m9, MCC_BAND, fp9)
s9, r9, _ = rpca_detect(f9, m9, np.array([sel["tau"]]), fp9); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): ранг {m9['basis'].shape[1]}, порог {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
