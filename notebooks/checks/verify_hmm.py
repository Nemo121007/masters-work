"""Проверка HMM блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
result = shell.run_cell("".join(cells["hmm-code-cell"]["source"]))
print(f"######## hmm-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
def frame_of(xy, t, label):
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "x": xy[:, 0], "y": xy[:, 1], "l": label, "lat": 44.0, "lon": 132.0})
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
def with_episodes(xy, t, seed, n_ep, lo, hi):
    xy = xy.copy(); label = np.full(len(t), "move", dtype=object); r = np.random.default_rng(seed)
    for start in r.choice(np.arange(50, len(t) - hi - 50), n_ep, replace=False):
        length = int(r.integers(lo, hi)); xy[start:start + length] += r.normal(0, 300, (length, 2)); label[start:start + length] = "anomaly"
    return xy, label

xy, t = simulate(2500, 0.02, 5.0, 1)
xy_l, lab_l = with_episodes(xy, t, 2, 15, 30, 80)
f = frame_of(xy_l, t, lab_l)
model = fit_hmm_on_train(f)
stay = np.diag(model["transmat"])
print(f"EM: итераций {model['iterations']}, сошёлся {model['converged']}, вероятность остаться: {np.round(stay, 3)}")
check("EM сошёлся", model["converged"])
check("переходы: длинные эпизоды → высокая вероятность остаться в режимах аномалии",
      model["transmat"][3:, 3:].sum(axis=1).min() > 0.9, f"P(аномалия → аномалия) по режимам {np.round(model['transmat'][3:, 3:].sum(axis=1), 3)}")
idx, fst = dbscan_segments(f, np.ones(len(f), dtype=bool))
feat = apply_robust(dbscan_features(xy_l[idx], t[idx], np.where(fst, -1, np.arange(len(idx)) - 1)), model["center"], model["scale"])
own = hmm_filter_scores(feat, fst, model)
ref = np.full(len(idx), np.nan); A_ = np.arange(6) >= 3
for run in _hmm_runs(fst, ~np.isnan(feat[:, 0])):
    logb = _hmm_log_emission(feat[run], model); p = np.log(model["startprob"]) + logb[0]
    out = [p - logsumexp(p)]
    for k in range(1, len(run)):
        p = logsumexp(out[-1][:, None] + np.log(model["transmat"]), axis=0) + logb[k]; out.append(p - logsumexp(p))
    ref[run] = np.exp(np.array(out))[:, A_].sum(axis=1)
check("фильтр = явная рекурсия вперёд", np.allclose(own, ref, equal_nan=True, atol=1e-9), f"max разница {np.nanmax(np.abs(own - ref)):.1e}")
s, r, first = hmm_detect(f, model, np.array([np.inf]))
truth = lab_l == "anomaly"; sc = ~np.isnan(s[0])
print(f"медиана оценки: штатный ход {np.nanmedian(s[0][~truth]):.3f}, аномалия {np.nanmedian(s[0][truth]):.3f}")
check("внутри длинных эпизодов оценка высокая", np.nanmedian(s[0][truth]) > 0.5 and np.nanmedian(s[0][~truth]) < 0.5)
check("первые 2 точки без оценки", np.all(np.isnan(s[0, :2])) and not np.isnan(s[0, 2]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _, _ = hmm_detect(frame_of(xy_l, t3, lab_l), model, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _, _ = hmm_detect(frame_of(xy_l, t4, lab_l), model, np.array([np.inf]))
check("разрыв по времени: фильтр заново", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_rot, _, _ = hmm_detect(frame_of(xy_l @ rot.T, t, lab_l), model, np.array([np.inf]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s, equal_nan=True, atol=1e-6))
clean = np.flatnonzero(~truth[1400:1700]) + 1400
j = int(clean[np.argmax(np.diff(clean) == 1)]) if len(clean) else 1500
xy6 = xy_l.copy(); xy6[j] += 800.0
s6, r6, first6 = hmm_detect(frame_of(xy6, t, lab_l), model, np.array([0.5]))
print(f"выброс в точке {j}: оценки первого прохода {np.round(first6[j:j + 4], 3)}, отвергнуто {np.flatnonzero(r6[0, j - 3:j + 8]) + j - 3}")
check("выброс отвергнут", r6[0, j])

xy9, t9 = simulate(2500, 0.02, 5.0, 3)
xy9, lab9 = with_episodes(xy9, t9, 4, 25, 5, 60)
f9 = frame_of(xy9, t9, lab9); truth9 = lab9 == "anomaly"
tic = time.perf_counter()
m9 = fit_hmm_on_train(f9); sel = select_hmm_threshold(f9, m9, MCC_BAND)
s9, r9, _ = hmm_detect(f9, m9, np.array([sel["tau"]])); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"эпизоды 5–60 точек ({time.perf_counter() - tic:.1f} с): порог {sel['tau']:.3f}, "
      f"TP {int(np.sum(p & truth9 & sc))} FP {int(np.sum(p & ~truth9 & sc))} FN {int(np.sum(~p & truth9 & sc))}, без оценки {int(np.sum(~sc))}")
''')
