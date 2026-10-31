"""Проверка LSTM-автоэнкодера блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
    for cid, cut in (("rpca-code-cell", "rpca_scenarios = ("),):
        src = "".join(cells[cid]["source"])
        assert shell.run_cell(src[: src.index(cut)]).success, cid
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
result = shell.run_cell("".join(cells["lstm-code-cell"]["source"]))
print(f"######## lstm-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
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

xy, t = simulate(2000, 0.02, 5.0, 1)
f = frame_of(xy, t, "move")
model = fit_lstm_on_train(f)
print(f"обучение: окон {model['n_train']}, калибровочных {model['n_calib']}, эпох {model['epochs']}, ошибка калибровки {model['calib_loss']:.4g}")
check("ранняя остановка: эпох ≤ 50, ошибка калибровки конечна", model["epochs"] <= NN_MAX_EPOCHS and np.isfinite(model["calib_loss"]))
m2 = fit_lstm_on_train(f)
s, r, first = lstm_detect(f, model, np.array([np.inf]))
s2, _, _ = lstm_detect(f, m2, np.array([np.inf]))
check("воспроизводимость (фиксированный seed)", np.allclose(s, s2, equal_nan=True, rtol=1e-5), f"max отн. разница {np.nanmax(np.abs(s - s2) / np.maximum(np.abs(s), 1e-9)):.1e}")
idx, fst = dbscan_segments(f, np.ones(len(f), dtype=bool))
fe = dbscan_features(xy[idx], t[idx], np.where(fst, -1, np.arange(len(idx)) - 1))
st = rpca_window_starts(fst, ~np.isnan(fe[:, 0]), LSTM_WINDOW)
calib = st[int(len(st) * (1 - NN_CALIB_FRACTION)):]
fit_all = st[st + LSTM_WINDOW <= calib[0]]
check("калибровочные окна не пересекаются с обучающими", fit_all.max() + LSTM_WINDOW <= calib.min(), f"конец обучающих {fit_all.max() + LSTM_WINDOW}, начало калибровочных {calib.min()}")
med = np.nanmedian(s[0])
check("оценка штатного хода ≈ d = 3 (χ²₃ по калибровке)", 1.0 < med < 6.0, f"медиана {med:.2f}")

check("первые 2 точки без оценки", np.all(np.isnan(s[0, :2])) and not np.isnan(s[0, 2]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _, _ = lstm_detect(frame_of(xy, t3, "move"), model, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _, _ = lstm_detect(frame_of(xy, t4, "move"), model, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_rot, _, _ = lstm_detect(frame_of(xy @ rot.T, t, "move"), model, np.array([np.inf]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s, equal_nan=True, rtol=1e-4, atol=1e-4))

tau = float(np.nanquantile(first, 0.995))
xy6 = xy.copy(); xy6[1500] += 800.0
f6 = frame_of(xy6, t, "move")
s6, r6, first6 = lstm_detect(f6, model, np.array([tau]))
far = np.r_[1485:1498, 1504:1516]
check("минимум по окнам: выброс не переносится на дальних соседей", first6[1500] > tau and np.all(first6[far] <= tau),
      f"выброс {first6[1500]:.0f}, макс. у соседей на 3–15 точек {np.max(first6[far]):.2f} при пороге {tau:.2f}")
check("выброс: первый проход — след на следующих точках", first6[1501] > tau and first6[1502] > tau, f"оценки {np.round(first6[1500:1503], 1)}")
s6b, r6b, _ = lstm_detect(f, model, np.array([tau]))
check("выброс: после второго прохода следующие точки не отвергнуты", r6[0, 1500] and not r6[0, 1501:1504].any(),
      f"отвергнуто в 1485–1519: {np.flatnonzero(r6[0, 1485:1520]) + 1485}, без выброса: {np.flatnonzero(r6b[0, 1485:1520]) + 1485}")
i6, f6s = dbscan_segments(f6, np.ones(len(f6), dtype=bool)); pos = np.arange(len(i6))
acc = ~(first6[i6] > tau); last = np.maximum.accumulate(np.where(acc, pos, -1))
feat2 = apply_robust(dbscan_features(xy6[i6], t[i6], np.where(f6s, -1, np.r_[-1, last[:-1]])), model["center"], model["scale"])
st2 = rpca_window_starts(f6s, ~np.isnan(feat2[:, 0]), LSTM_WINDOW); cov = st2[:, None] + np.arange(LSTM_WINDOW)
win = feat2[cov].astype(np.float32); err = batched_model_forward(model["model"], win, lambda m, b: m(b), TORCH_INFER_BATCH) - win
d = np.sum((err - model["error_mean"]) ** 2 / model["error_var"], axis=2); pts = np.full(len(i6), np.inf)
np.minimum.at(pts, cov.ravel(), d.ravel()); full = np.full(len(f6), np.nan); ok = np.isfinite(pts); full[i6[ok]] = pts[ok]
check("второй проход: пересчёт только изменённых окон = полный пересчёт", np.allclose(full, s6[0], equal_nan=True, rtol=1e-4, atol=1e-6))

xy9, t9 = simulate(2500, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
m9 = fit_lstm_on_train(f9); sel = select_lstm_threshold(f9, m9, MCC_BAND)
s9, r9, _ = lstm_detect(f9, m9, np.array([sel["tau"]])); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): эпох {m9['epochs']}, порог {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
