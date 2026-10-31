"""Проверка фильтра Калмана блокнота: ячейка на test.csv и синтетика с известным ответом."""
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
result = shell.run_cell("".join(cells["kalman-code-cell"]["source"]))
print(f"######## kalman-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
def frame_of(xy, t, label):
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "x": xy[:, 0], "n": xy[:, 1], "y": label, "lat": 44.0, "lon": 132.0})
def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
def simulate(count, sigma_acc, sigma_meas, seed, drift=(5.0, 2.0)):
    r = np.random.default_rng(seed)
    dt = np.where(r.random(count) < 0.03, r.uniform(20.0, 170.0, count), 10.0)
    pos = np.zeros(2); vel = np.array(drift); out = []
    for d in dt:
        acc = r.normal(0.0, sigma_acc, 2)
        pos = pos + vel * d + 0.5 * acc * d * d; vel = vel + acc * d; out.append(pos.copy())
    return np.array(out) + r.normal(0.0, sigma_meas, (count, 2)), np.cumsum(dt)

xy, t = simulate(6000, 0.05, 8.0, 1)
frame = frame_of(xy, t, np.full(len(t), "move"))
sa, sm = fit_kalman_on_train(frame)
vl = restart_speed_limit(frame)
check("параметры", abs(sa / 0.05 - 1) < 0.15 and abs(sm / 8.0 - 1) < 0.1, f"σ_acc {sa:.4f} (0.05), σ_meas {sm:.2f} (8.0)")

scores, _ = kalman_detect(frame, sa, sm, vl, np.array([np.inf]))
step = np.r_[np.diff(t), 0.0]
tr = np.zeros((2, 2, len(t))); tr[0, 0] = tr[1, 1] = 1.0; tr[0, 1] = step
ref = np.zeros(len(t))
for column in range(2):
    kf = KalmanFilter(k_endog=1, k_states=2, k_posdef=2)
    kf.bind(xy[:, column] - xy[0, column])
    kf["design"] = np.array([[1.0, 0.0]]); kf["obs_cov"] = np.array([[sm**2]])
    kf["transition"] = tr; kf["selection"] = np.eye(2); kf["state_cov"] = kalman_process_noise(step, sa)
    kf.initialize_approximate_diffuse(_DIFFUSE_VAR_M2)
    res = kf.filter()
    ref += res.forecasts_error[0] ** 2 / res.forecasts_error_cov[0, 0]
diff = np.abs(scores[0, 2:] - ref[2:])
check("NIS = statsmodels (с 6-й точки)", np.max(diff[4:] / np.maximum(ref[6:], 1.0)) < 1e-3, f"max отн. разница {np.max(diff[4:] / np.maximum(ref[6:], 1.0)):.1e}")
check("первые 2 точки без оценки", np.all(np.isnan(scores[0, :2])) and not np.isnan(scores[0, 2]))
check("медиана NIS на штатном ходу ≈ χ²₂ (1.39)", 1.1 < np.nanmedian(scores[0]) < 1.7, f"{np.nanmedian(scores[0]):.2f}")

t3 = t.copy(); t3[1000] = t3[999]
s3, _ = kalman_detect(frame_of(xy, t3, np.full(len(t), "move")), sa, sm, vl, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _ = kalman_detect(frame_of(xy, t4, np.full(len(t), "move")), sa, sm, vl, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))

tau = KALMAN_CHI999 * 3
xy5 = xy.copy(); xy5[1500] += 5000.0
s5, r5 = kalman_detect(frame_of(xy5, t, np.full(len(t), "move")), sa, sm, vl, np.array([tau, np.inf]))
check("выброс с заменой", r5[0, 1500] and not r5[0, 1501:1520].any(), f"отвергнуто рядом {int(r5[0, 1495:1520].sum())}")
check("без замены — след выброса", s5[1, 1501] > tau, f"NIS соседа {s5[1, 1501]:.0f}")
xy6 = xy.copy(); xy6[2000:2060] += 50_000.0
s6, r6 = kalman_detect(frame_of(xy6, t, np.full(len(t), "move")), sa, sm, vl, np.array([tau]))
check("длинный эпизод 50 км: отвергнут целиком, возврат на трек",
      r6[0, 2000:2060].all() and not r6[0, 2060:2100].any(),
      f"отвергнуто в эпизоде {int(r6[0, 2000:2060].sum())} из 60, после {int(r6[0, 2060:2100].sum())}")

xy7, t7 = simulate(5000, 0.05, 8.0, 3)
label = np.full(len(t7), "move", dtype=object)
r = np.random.default_rng(4)
for start in r.choice(np.arange(50, len(t7) - 50), 40, replace=False):
    length = int(r.integers(1, 8))
    xy7[start:start + length] += r.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f7 = frame_of(xy7, t7, label); truth = label == "anomaly"
sa7, sm7 = fit_kalman_on_train(f7)
vl7 = restart_speed_limit(f7)
sel = select_kalman_threshold(f7, sa7, sm7, vl7, MCC_BAND)
s7, r7 = kalman_detect(f7, sa7, sm7, vl7, np.array([sel["tau"]])); sc = ~np.isnan(s7[0]); p = r7[0]
print(f"дрейф и аномалии: σ_acc {sa7:.4f}, σ_meas {sm7:.2f}, порог {sel['tau']:.1f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}")
''')
