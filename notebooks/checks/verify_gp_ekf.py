"""Проверка GP-EKF блокнота: ячейка на test.csv и синтетика с известным полем ускорений."""
import contextlib, io, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".claude" / "skills" / "run-notebook"))
from run_notebook import _make_shell, PROJECT_ROOT
ids = ["b79e4cc6", "a1imports", "72430d98981e0032", "cdb3e6d2", "777bfdcb5c54ef", "deacacdefd3a0f35"]
cells = {c.get("id"): c for c in json.loads(Path("example_model.ipynb").read_text(encoding="utf-8"))["cells"]}
shell = _make_shell(PROJECT_ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    for cid in ids + ["kalman-code-cell"]:
        assert shell.run_cell("".join(cells[cid]["source"])).success, cid
started = time.perf_counter()
result = shell.run_cell("".join(cells["gp-ekf-code-cell"]["source"]))
print(f"######## gp-ekf-code-cell: {time.perf_counter() - started:.1f} s, {'ok' if result.success else 'FAIL'}", flush=True)
shell.run_cell(r'''
import warnings; warnings.filterwarnings("ignore")
def frame_of(xy, t, label):
    return pd.DataFrame({"time": pd.to_datetime(t, unit="s"), "x": xy[:, 0], "n": xy[:, 1], "y": label, "lat": 44.0, "lon": 132.0})
def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)
OMEGA = 0.004
def field(v):
    # Поворот с угловой скоростью, растущей со скоростью на север: поле ускорений зависит от вектора скорости.
    rate = OMEGA * (1.0 + 0.5 * np.tanh(v[..., 1] / 3.0))
    return np.stack([-rate * v[..., 1], rate * v[..., 0]], axis=-1)
def simulate(count, sigma_acc, sigma_meas, seed, with_field=True):
    r = np.random.default_rng(seed)
    dt = np.where(r.random(count) < 0.03, r.uniform(20.0, 170.0, count), 10.0)
    pos = np.zeros(2); vel = np.array([5.0, 2.0]); out = []
    for d in dt:
        sub = int(d)
        acc_noise = r.normal(0.0, sigma_acc, 2)
        for _ in range(sub):
            acc = (field(vel) if with_field else 0.0) + acc_noise
            pos = pos + vel + 0.5 * acc; vel = vel + acc
        out.append(pos.copy())
    return np.array(out) + r.normal(0.0, sigma_meas, (count, 2)), np.cumsum(dt)

# 1. Совпадение с фильтром Калмана при нулевой поправке.
xy0, t0 = simulate(3000, 0.05, 8.0, 1, with_field=False)
f0 = frame_of(xy0, t0, np.full(len(t0), "move"))
sa0, sm0 = fit_kalman_on_train(f0)
nodes = [np.linspace(-50, 50, 3), np.linspace(-50, 50, 3)]
zero = {"nodes": nodes, "mean": np.zeros((2, 3, 3)), "variance": np.zeros((2, 3, 3)), "gradient": np.zeros((2, 2, 3, 3))}
s_gp, _ = gp_ekf_detect(f0, {"correction": zero, "sigma_acc": np.array([sa0, sa0]), "sigma_meas": sm0, "speed_limit": np.inf}, np.array([np.inf]))
s_k, _ = kalman_detect(f0, sa0, sm0, np.inf, np.array([np.inf]))
both = ~np.isnan(s_k[0])
check("g ≡ 0 → NIS фильтра Калмана", np.array_equal(both, ~np.isnan(s_gp[0])) and np.max(np.abs(s_gp[0][both] - s_k[0][both]) / np.maximum(s_k[0][both], 1.0)) < 1e-4,
      f"max отн. разница {np.max(np.abs(s_gp[0][both] - s_k[0][both]) / np.maximum(s_k[0][both], 1.0)):.1e}")

# 2. Синтетика с полем ускорений.
SA, SM = 0.01, 6.0
xy, t = simulate(12000, SA, SM, 2)
frame = frame_of(xy, t, np.full(len(t), "move"))
tic = time.perf_counter(); model = fit_gp_ekf_on_train(frame); fit_time = time.perf_counter() - tic
corr = model["correction"]
print(f"обучение {fit_time:.1f} с; Калман σ_acc {model['kalman_sigma'][0]:.4f}, σ_meas {model['kalman_sigma'][1]:.2f}; "
      f"GP-EKF σ_acc {model['sigma_acc'].round(4)}, σ_meas {model['sigma_meas']:.2f}; "
      + "; ".join(f"σ_f {a['amplitude']:.3g} ℓ {a['lengths'].round(2)} шум цели {a['target_noise']:.2g}" for a in corr["axes"]))
inputs = corr["inputs"]
r = np.random.default_rng(5)
probe = inputs[r.choice(len(inputs), 300)] + r.normal(0, 0.05, (300, 2))
look_mean, look_var, look_jac = gp_ekf_lookup(corr, probe)
exact = [gp_ekf_posterior(a, inputs, probe) for a in corr["axes"]]
amp = max(a["amplitude"] for a in corr["axes"])
err_mean = max(np.max(np.abs(look_mean[:, k] - exact[k][0])) for k in range(2)) / amp
err_var = max(np.max(np.abs(look_var[:, k] - exact[k][1])) for k in range(2)) / amp**2
check("сетка: среднее и дисперсия = точный GP", err_mean < 0.02 and err_var < 0.02, f"ошибка {err_mean:.1e} σ_f, {err_var:.1e} σ_f²")
h = 1e-4
num_jac = np.stack([(gp_ekf_posterior(a, inputs, probe + h * np.eye(2)[m])[0] - gp_ekf_posterior(a, inputs, probe - h * np.eye(2)[m])[0]) / (2 * h)
                    for a in corr["axes"] for m in range(2)], axis=1).reshape(-1, 2, 2)
exact_jac = np.stack([e[2] for e in exact], axis=1)
scale_jac = np.max(np.abs(num_jac))
check("якобиан: аналитический = численный", np.max(np.abs(exact_jac - num_jac)) / scale_jac < 1e-4, f"{np.max(np.abs(exact_jac - num_jac)) / scale_jac:.1e}")
check("якобиан: сетка ≈ точный", np.max(np.abs(look_jac - exact_jac)) / scale_jac < 0.05, f"{np.max(np.abs(look_jac - exact_jac)) / scale_jac:.1e}")

truth = field(inputs)
mean_at = np.column_stack([gp_ekf_posterior(a, inputs, inputs)[0] for a in corr["axes"]])
slope = float(np.sum(mean_at * truth) / np.sum(truth * truth))
check("поправка ≈ истинное поле (занижение сглаживателем)", slope > 0.8, f"наклон μ к истинному полю {slope:.3f}")
xy_t, t_t = simulate(4000, SA, SM, 3)
ftest = frame_of(xy_t, t_t, np.full(len(t_t), "move"))
share_train, share_test = gp_ekf_explained_share(frame, model), gp_ekf_explained_share(ftest, model)
i_t, g_t, _ = gp_ekf_smoothed_transitions(ftest, *model["kalman_sigma"])
mu_t, field_t = gp_ekf_lookup(corr, i_t)[0], field(i_t)
r2_field = 1 - np.sum((field_t - mu_t) ** 2) / np.sum((field_t - field_t.mean(0)) ** 2)
ceiling = 1 - np.sum((g_t - field_t) ** 2) / np.sum((g_t - g_t.mean(0)) ** 2)
check("перенос поправки: μ ≈ истинное поле на тестовой траектории", r2_field > 0.9 and share_test > 0.9 * ceiling,
      f"R² μ→поле {r2_field:.3f}; доля test {share_test:.3f} при потолке {ceiling:.3f}; train {share_train:.3f}")
check("параметры шума", abs(model["sigma_meas"] / SM - 1) < 0.1 and np.all(model["sigma_acc"] > 0.9 * SA) and np.all(model["sigma_acc"] < 2 * SA)
      and np.all(model["sigma_acc"] < 0.5 * model["kalman_sigma"][0]),
      f"σ_acc {model['sigma_acc'].round(4)} ({SA}; ошибка модели — в шум процесса), Калман {model['kalman_sigma'][0]:.4f}, σ_meas {model['sigma_meas']:.2f} ({SM})")
i0, g0, s0 = gp_ekf_smoothed_transitions(f0, sa0, sm0)
c0 = fit_gp_ekf_correction(i0, g0, s0)
sup = gp_ekf_support_index(s0, np.random.default_rng(GP_EKF_SEED))
r2_0 = 1 - np.sum((g0 - gp_ekf_lookup(c0, i0)[0]) ** 2) / np.sum((g0 - g0.mean(0)) ** 2)
check("без поля: поправка ≈ 0, опорные точки через ≥ 120 с", abs(r2_0) < 0.05 and np.min(np.diff(s0[sup])) >= GP_EKF_SUPPORT_GAP_SEC - 1e-6,
      f"доля {r2_0:.3f}; опорных {len(sup)}, мин. интервал {np.min(np.diff(s0[sup])):.0f} с; "
      + "; ".join(f"σ_f {a['amplitude']:.2g} ℓ {a['lengths'].round(2)}" for a in c0["axes"]))

tic = time.perf_counter(); scores, _ = gp_ekf_detect(ftest, model, np.array([np.inf])); pass_time = time.perf_counter() - tic
sk, _ = kalman_detect(ftest, *model["kalman_sigma"], np.inf, np.array([np.inf]))
check("медиана NIS на штатном ходу ≈ χ²₂ (1.39)", 1.1 < np.nanmedian(scores[0]) < 1.7,
      f"GP-EKF {np.nanmedian(scores[0]):.2f}, Калман {np.nanmedian(sk[0]):.2f}; проход {pass_time:.1f} с на {len(t_t)} точек")
print(f"логарифм правдоподобия на test (с 3-й точки): GP-EKF {np.nansum(-0.5 * scores[0]):.0f}, Калман {np.nansum(-0.5 * sk[0]):.0f} (только квадратичная часть)")
check("первые 2 точки без оценки", np.all(np.isnan(scores[0, :2])) and not np.isnan(scores[0, 2]))

t3 = t_t.copy(); t3[1000] = t3[999]
s3, _ = gp_ekf_detect(frame_of(xy_t, t3, "move"), model, np.array([np.inf]))
check("совпадающее время", np.isnan(s3[0, 1000]) and not np.isnan(s3[0, 1001]))
t4 = t_t.copy(); t4[1200:] += 600.0
s4, _ = gp_ekf_detect(frame_of(xy_t, t4, "move"), model, np.array([np.inf]))
check("разрыв по времени", np.all(np.isnan(s4[0, 1200:1202])) and not np.isnan(s4[0, 1202]))
tau = KALMAN_CHI999 * 3
xy5 = xy_t.copy(); xy5[1500] += 5000.0
s5, r5 = gp_ekf_detect(frame_of(xy5, t_t, "move"), model, np.array([tau, np.inf]))
check("выброс с заменой", r5[0, 1500] and not r5[0, 1501:1520].any(), f"отвергнуто рядом {int(r5[0, 1495:1520].sum())}")
check("без замены — след выброса", s5[1, 1501] > tau, f"NIS соседа {s5[1, 1501]:.0f}")
xy6 = xy_t.copy(); xy6[2000:2060] += 50_000.0
s6, r6 = gp_ekf_detect(frame_of(xy6, t_t, "move"), model, np.array([tau]))
check("длинный эпизод 50 км: отвергнут целиком, возврат на трек",
      r6[0, 2000:2060].all() and not r6[0, 2060:2100].any(),
      f"отвергнуто в эпизоде {int(r6[0, 2000:2060].sum())} из 60, после {int(r6[0, 2060:2100].sum())}")

xy7, t7 = simulate(5000, SA, SM, 4)
label = np.full(len(t7), "move", dtype=object)
r = np.random.default_rng(4)
for start in r.choice(np.arange(50, len(t7) - 50), 40, replace=False):
    length = int(r.integers(1, 8))
    xy7[start:start + length] += r.normal(0, 3000, (length, 2)); label[start:start + length] = "anomaly"
f7 = frame_of(xy7, t7, label); truth7 = label == "anomaly"
tic = time.perf_counter()
m7 = fit_gp_ekf_on_train(f7)
sel = select_gp_ekf_threshold(f7, m7, MCC_BAND)
s7, r7 = gp_ekf_detect(f7, m7, np.array([sel["tau"]])); sc = ~np.isnan(s7[0]); p = r7[0]
sa7, sm7 = fit_kalman_on_train(f7)
vlk = restart_speed_limit(f7)
selk = select_kalman_threshold(f7, sa7, sm7, vlk, MCC_BAND)
sk7, rk7 = kalman_detect(f7, sa7, sm7, vlk, np.array([selk["tau"]])); sck = ~np.isnan(sk7[0]); pk = rk7[0]
print(f"поле и аномалии ({time.perf_counter() - tic:.0f} с): GP-EKF порог {sel['tau']:.1f}, TP {int(np.sum(p & truth7 & sc))} FP {int(np.sum(p & ~truth7 & sc))} FN {int(np.sum(~p & truth7 & sc))}; "
      f"Калман порог {selk['tau']:.1f}, TP {int(np.sum(pk & truth7 & sck))} FP {int(np.sum(pk & ~truth7 & sck))} FN {int(np.sum(~pk & truth7 & sck))}")
''')
