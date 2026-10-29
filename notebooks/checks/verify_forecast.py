"""Проверка моделей прогноза AR и ARIMA блокнота на синтетике с известным ответом."""
import contextlib, io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".claude" / "skills" / "run-notebook"))
from run_notebook import _make_shell, PROJECT_ROOT
ids = ["b79e4cc6", "a1imports", "72430d98981e0032", "cdb3e6d2", "777bfdcb5c54ef", "deacacdefd3a0f35"]
cells = {c.get("id"): c for c in json.loads(Path("example_model.ipynb").read_text(encoding="utf-8"))["cells"]}
shell = _make_shell(PROJECT_ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    for cid in ids:
        assert shell.run_cell("".join(cells[cid]["source"])).success, cid
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_process import arma_generate_sample
def frame_of(o, n, t, label):
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "x": o, "n": n, "y": label, "lat": 44.0, "lon": 132.0})
def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
count = 3000; t = np.arange(count) * 10.0
for order, params in (((2, 0, 0), ([1.5, -0.6], [], 5.0)), ((2, 1, 1), ([0.6, 0.3], [-0.7], 3.0))):
    r = np.random.default_rng(1)
    o, n = [(np.cumsum(e) if order[1] else e + 3000.0) for e in
            (arma_generate_sample(np.r_[1, -np.array(params[0])], np.r_[1, params[1]], count + 200, scale=params[2], distrvs=r.standard_normal)[200:] for _ in range(2))]
    frame = frame_of(o, n, t, np.full(count, "move"))
    det = fit_forecast_detector(frame, order, "n"); k = det["n_unscored"]
    true = np.r_[params[0], params[1], params[2] ** 2]
    check(f"{order} параметры", np.allclose(det["params"][0], true, atol=0.08 * np.maximum(1, np.abs(true))), f"{np.round(det['params'][0], 3)} истина {true}")
    scores, _ = forecast_detect(frame, det, np.array([np.inf]))
    pred_o = ARIMA(o - o[0], order=order, trend="n").filter(det["params"][0]).get_prediction(start=0).predicted_mean + o[0]
    pred_n = ARIMA(n - n[0], order=order, trend="n").filter(det["params"][1]).get_prediction(start=0).predicted_mean + n[0]
    ref = np.hypot(o - pred_o, n - pred_n)
    check(f"{order} фильтр = statsmodels", np.abs(scores[0, 300:] - ref[300:]).max() < 1e-3, f"{np.abs(scores[0, 300:] - ref[300:]).max():.1e} м")
    check(f"{order} первые {k} точек без оценки", np.all(np.isnan(scores[0, :k])) and not np.isnan(scores[0, k]))
    keep = np.ones(count, bool); keep[[500, 501, 502, 900]] = False; keep[1500:1510] = False
    s2, _ = forecast_detect(frame_of(o[keep], n[keep], t[keep], np.full(keep.sum(), "move")), det, np.array([np.inf]))
    og = np.where(keep, o - o[0], np.nan); ng = np.where(keep, n - n[0], np.nan)
    ref2 = np.hypot(og - ARIMA(og, order=order, trend="n").filter(det["params"][0]).get_prediction(start=0).predicted_mean,
                    ng - ARIMA(ng, order=order, trend="n").filter(det["params"][1]).get_prediction(start=0).predicted_mean)[keep]
    check(f"{order} пропуски = statsmodels", np.abs(s2[0][300:] - ref2[300:]).max() < 1e-3)
    t3 = t.copy(); t3[1000] = t3[999]
    s3, _ = forecast_detect(frame_of(o, n, t3, np.full(count, "move")), det, np.array([np.inf]))
    check(f"{order} совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
    t4 = t.copy(); t4[1200:] += 600.0
    s4, _ = forecast_detect(frame_of(o, n, t4, np.full(count, "move")), det, np.array([np.inf]))
    check(f"{order} разрыв по времени", np.all(np.isnan(s4[0, 1200:1200 + k])) and not np.isnan(s4[0, 1200 + k]))
    o5 = o.copy(); o5[1500] += 5000.0
    tau = np.nanquantile(scores[0], 0.999) * 3
    s5, r5 = forecast_detect(frame_of(o5, n, t, np.full(count, "move")), det, np.array([tau, np.inf]))
    check(f"{order} выброс с заменой", r5[0, 1500] and r5[0].sum() == 1)
    check(f"{order} без замены — эхо", s5[1, 1501] > tau)
    o6 = o.copy(); o6[2000:2050] += 5000.0
    s6, r6 = forecast_detect(frame_of(o6, n, t, np.full(count, "move")), det, np.array([tau]))
    check(f"{order} перезапуск после 180 с отвергнутых", r6[0, 2000:2019].all() and np.all(np.isnan(s6[0, 2019:2019 + k])))

r = np.random.default_rng(3); count = 4000
def axis():
    e = arma_generate_sample(np.r_[1, -0.6, -0.3], np.r_[1, -0.7], count + 200, scale=3.0, distrvs=r.standard_normal)[200:]
    return np.cumsum(e + 2.0)
o, n = axis(), axis()
label = np.full(count, "move", dtype=object)
for start in r.choice(np.arange(50, count - 50), 40, replace=False):
    length = int(r.integers(1, 8))
    o[start:start + length] += r.normal(0, 3000, length); n[start:start + length] += r.normal(0, 3000, length)
    label[start:start + length] = "anomaly"
frame = frame_of(o, n, np.arange(count) * 10.0, label)
truth = label == "anomaly"
for order in ((2, 0, 0), (2, 1, 1)):
    det = fit_forecast_detector(frame, order, "n")
    sel = select_forecast_threshold(frame, det, MCC_BAND)
    scores, pred, scored = forecast_predict(frame, det, sel["tau"])
    tp = int(np.sum(pred & truth & scored)); fp = int(np.sum(pred & ~truth & scored)); fn = int(np.sum(~pred & truth & scored))
    print(order, "дрейф и аномалии: параметры", np.round(det["params"][0], 3), "порог", round(sel["tau"], 1), "м,",
          f"TP {tp} FP {fp} FN {fn}, отвергнутые ⊂ {{s>τ}}: {bool(np.all(scores[pred] > sel['tau']))}")
''')
