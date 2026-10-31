"""Проверка скользящего гауссовского процесса блокнота: ячейка на test.csv и синтетика."""
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
result = shell.run_cell("".join(cells["gp-code-cell"]["source"]))
print(f"######## gp-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF, WhiteKernel
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

# 1. Прогноз по окну против sklearn: универсальный кригинг — предел очень широкого априорного распределения тренда.
r = np.random.default_rng(0)
hyper = np.array([15.0, 50.0, 6.0])
prior = 1e8
worst_mean, worst_var = 0.0, 0.0
for trial in range(20):
    m = int(r.integers(3, GP_WINDOW + 1))
    tw = np.sort(r.uniform(0, 300, m)); yw = np.column_stack([5 * tw, 2 * tw]) + 1000.0 + r.normal(0, 6, (m, 2)); tq = tw[-1] + 10.0
    wt = np.zeros((1, GP_WINDOW)); wx = np.zeros((1, GP_WINDOW, 2)); valid = np.zeros((1, GP_WINDOW), bool)
    wt[0, -m:] = tw; wx[0, -m:] = yw; valid[0, -m:] = True
    mean, var = gp_window_prediction(wt, wx, valid, np.array([tq]), hyper)
    kernel = ConstantKernel(prior, "fixed") * DotProduct(sigma_0=0.0, sigma_0_bounds="fixed") + ConstantKernel(prior, "fixed") \
        + ConstantKernel(hyper[0] ** 2, "fixed") * RBF(hyper[1], "fixed") + WhiteKernel(hyper[2] ** 2, "fixed")
    for axis in range(2):
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit((tw - tw.mean())[:, None], yw[:, axis])
        mu, sd = gpr.predict(np.array([[tq - tw.mean()]]), return_std=True)
        worst_mean = max(worst_mean, abs(mu[0] - mean[0, axis]))
        worst_var = max(worst_var, abs(sd[0] ** 2 - (var[0] + hyper[2] ** 2)) / (var[0] + hyper[2] ** 2))
check("прогноз окна = sklearn (широкое априорное тренда)", worst_mean < 1e-3 and worst_var < 1e-3,
      f"max|разница| среднего {worst_mean:.1e} м, отн. дисперсии {worst_var:.1e}")

# 2. Гиперпараметры и калибровка на синтетике.
xy, t = simulate(6000, 0.05, 8.0, 1)
frame = frame_of(xy, t, np.full(len(t), "move"))
hyper = fit_gp_on_train(frame)
vl = restart_speed_limit(frame)
check("σ_noise ≈ шуму GPS 8 м", abs(hyper[2] / 8.0 - 1) < 0.15, f"гиперпараметры {np.round(hyper, 3)}")
scores, _ = gp_detect(frame, hyper, vl, np.array([np.inf]))
check("медиана NIS на штатном ходу ≈ χ²₂ (1.39)", 1.0 < np.nanmedian(scores[0]) < 1.9, f"{np.nanmedian(scores[0]):.2f}")
check("первые 3 точки без оценки", np.all(np.isnan(scores[0, :3])) and not np.isnan(scores[0, 3]))
t3 = t.copy(); t3[1000] = t3[999]
s3, _ = gp_detect(frame_of(xy, t3, np.full(len(t), "move")), hyper, vl, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t.copy(); t4[1200:] += 600.0
s4, _ = gp_detect(frame_of(xy, t4, np.full(len(t), "move")), hyper, vl, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1203])) and not np.isnan(s4[0, 1203]))
tau = GP_CHI999 * 3
xy5 = xy.copy(); xy5[1500] += 5000.0
s5, r5 = gp_detect(frame_of(xy5, t, np.full(len(t), "move")), hyper, vl, np.array([tau, np.inf]))
check("выброс с заменой", r5[0, 1500] and not r5[0, 1501:1520].any(), f"отвергнуто рядом {int(r5[0, 1495:1520].sum())}")
check("без замены — след выброса", s5[1, 1501] > tau, f"оценка соседа {s5[1, 1501]:.0f}")
xy6 = xy.copy(); xy6[2000:2060] += 50_000.0
s6, r6 = gp_detect(frame_of(xy6, t, np.full(len(t), "move")), hyper, vl, np.array([tau]))
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
h7 = fit_gp_on_train(f7)
vl7 = restart_speed_limit(f7)
sel = select_gp_threshold(f7, h7, vl7, MCC_BAND)
s7, r7 = gp_detect(f7, h7, vl7, np.array([sel["tau"]])); sc = ~np.isnan(s7[0]); p = r7[0]
print(f"дрейф и аномалии: гиперпараметры {np.round(h7, 3)}, порог {sel['tau']:.1f}, "
      f"TP {int(np.sum(p & truth & sc))} FP {int(np.sum(p & ~truth & sc))} FN {int(np.sum(~p & truth & sc))}")
''')
