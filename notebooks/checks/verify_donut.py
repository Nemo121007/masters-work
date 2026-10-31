"""Проверка DONUT блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
    for cid, cut in (("rpca-code-cell", "rpca_scenarios = ("), ("lstm-code-cell", "lstm_scenarios = (")):
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
result = shell.run_cell("".join(cells["donut-code-cell"]["source"]))
print(f"######## donut-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
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
model = fit_donut_on_train(f)
print(f"обучение: окон {model['n_train']}, калибровочных {model['n_calib']}, эпох {model['epochs']}, ошибка калибровки {model['calib_loss']:.4g}")
check("ранняя остановка: эпох ≤ 50, ошибка калибровки конечна", model["epochs"] <= NN_MAX_EPOCHS and np.isfinite(model["calib_loss"]))
m2 = fit_donut_on_train(f)
s, r, first = donut_detect(f, model, np.array([np.inf]))
s2, _, _ = donut_detect(f, m2, np.array([np.inf]))
check("воспроизводимость (фиксированный seed)", np.allclose(s, s2, equal_nan=True, rtol=1e-5))
fl = torch.from_numpy(np.zeros((4, 48), dtype=np.float32)); qm, qs = model["model"].encode(fl.to(DEVICE)); kl = float((-torch.log(qs) + 0.5 * (qs**2 + qm**2) - 0.5).sum(1).mean())
check("VAE: KL(q‖N(0,I)) > 0 и оценка детерминирована", kl > 0 and np.array_equal(s, donut_detect(f, model, np.array([np.inf]))[0], equal_nan=True), f"KL {kl:.3f}, медиана оценки штатного хода {np.nanmedian(s[0]):.2f}")
xyf = xy.copy(); xyf[1601:] += np.random.default_rng(3).normal(0, 300, (len(xy) - 1601, 2))
sf, _, _ = donut_detect(frame_of(xyf, t, "move"), model, np.array([np.inf]))
check("причинность: будущие точки не меняют оценку", np.allclose(sf[0, :1601], s[0, :1601], equal_nan=True, rtol=1e-6))
check("первые 17 точек без оценки", np.all(np.isnan(s[0, :17])) and not np.isnan(s[0, 17]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _, _ = donut_detect(frame_of(xy, t3, "move"), model, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _, _ = donut_detect(frame_of(xy, t4, "move"), model, np.array([np.inf]))
check("разрыв по времени: 17 точек без оценки", np.all(np.isnan(s4[0, 1200:1217])) and not np.isnan(s4[0, 1217]))
angle = 1.1; rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
s_rot, _, _ = donut_detect(frame_of(xy @ rot.T, t, "move"), model, np.array([np.inf]))
check("не зависит от курса (поворот траектории)", np.allclose(s_rot, s, equal_nan=True, rtol=1e-4, atol=1e-4))

tau = float(np.nanquantile(first, 0.995))
xy6 = xy.copy(); xy6[1500] += 800.0
f6 = frame_of(xy6, t, "move")
s6, r6, first6 = donut_detect(f6, model, np.array([tau]))
check("выброс: первый проход — след на следующих точках", first6[1500] > tau and first6[1501] > tau,
      f"оценки {np.round(first6[1500:1505], 1)}, порог {tau:.2f}")
s6b, r6b, _ = donut_detect(f, model, np.array([tau]))
check("выброс: после второго прохода следующие точки не отвергнуты", r6[0, 1500] and not r6[0, 1501:1517].any(),
      f"отвергнуто в 1495–1519: {np.flatnonzero(r6[0, 1495:1520]) + 1495}, без выброса: {np.flatnonzero(r6b[0, 1495:1520]) + 1495}")
i6, f6s = dbscan_segments(f6, np.ones(len(f6), dtype=bool)); pos = np.arange(len(i6))
acc = ~(first6[i6] > tau); last = np.maximum.accumulate(np.where(acc, pos, -1))
feat2 = apply_robust(dbscan_features(xy6[i6], t[i6], np.where(f6s, -1, np.r_[-1, last[:-1]])), model["center"], model["scale"])
full = np.full(len(f6), np.nan)
for tgt in np.flatnonzero(~np.isnan(first6[i6])):
    hist = [p for p in range(tgt) if acc[p] and not np.isnan(feat2[p, 0])][-DONUT_HISTORY:]
    if len(hist) < DONUT_HISTORY:
        full[i6[tgt]] = first6[i6[tgt]]; continue
    full[i6[tgt]] = _donut_scores(model, feat2[np.array(hist)][None], feat2[tgt][None])[0]
check("второй проход: эталон с явной историей из принятых точек", np.allclose(full, s6[0], equal_nan=True, rtol=1e-4, atol=1e-5),
      f"max разница {np.nanmax(np.abs(full - s6[0])):.1e}")

xy9, t9 = simulate(2500, 0.02, 5.0, 3)
label = np.full(len(t9), "move", dtype=object)
rr = np.random.default_rng(4)
for start in rr.choice(np.arange(50, len(t9) - 50), 40, replace=False):
    length = int(rr.integers(1, 8))
    xy9[start:start + length] += rr.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f9 = frame_of(xy9, t9, label); truth = label == "anomaly"
tic = time.perf_counter()
m9 = fit_donut_on_train(f9); sel = select_donut_threshold(f9, m9, MCC_BAND)
s9, r9, _ = donut_detect(f9, m9, np.array([sel["tau"]])); sc = ~np.isnan(s9[0]); p = r9[0]
print(f"дрейф и аномалии ({time.perf_counter() - tic:.1f} с): эпох {m9['epochs']}, порог {sel['tau']:.2f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}, без оценки {int(np.sum(~sc))}")
''')
