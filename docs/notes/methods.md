# Общие тезисы

Справочник по методам: постановка по оригинальным статьям и адаптация в блокноте `notebooks/example_model.ipynb`. Полные описания адаптаций — в markdown-ячейках блокнота и в `report/Отчёт.tex`; обозначения и термины — по `report/glossary.md`.

Ряд после предобработки — точки $i$ со временем $t_i$, плоскими координатами ENU $x_i$ (восток), $y_i$ (север), м, и эталонной меткой $l_i\in\{\mathrm{move},\mathrm{anomaly}\}$; $\hat l_i$ — предсказание. Положительный класс — аномалия:

$$
a_i=\mathbf1[l_i=\mathrm{anomaly}], \qquad \hat a_i=\mathbf1[\hat l_i=\mathrm{anomaly}].
$$

Модель $M$ сопоставляет точке оценку аномальности $s_i^{(M)}$: чем больше, тем аномальнее; решение $\hat a_i=\mathbf1[s_i>\tau]$. Без непрерывной оценки — MPCT (порог — предел скорости $v_+$); у DBSCAN порог — радиус $\varepsilon$.

Выборки. Тестовая выборка — ряд от первой точки штатного хода не левее $\lfloor0.6N\rfloor$; пул обучения — всё до неё; обучающая выборка — обучающий блок пула (начало $0.375$, длина $0.4$ пула), его первая и последняя точки — штатный ход. Блок найден отдельным экспериментом как самый короткий блок, в котором достаточно окон для нейросетей и эпизодов аномалий, распределение признаков штатного хода совпадает с пулом и покрыты режимы движения пула.

Обучающий отрезок — подряд идущие точки штатного хода обучающей выборки; рвётся на каждой аномальной точке и на разрыве по времени больше 180 с. По обучающим отрезкам оцениваются параметры моделей, которые учатся на штатном ходу.

Выбор порога (общий для всех моделей): сетка кандидатов — 180 квантилей оценки аномальности на обучающей выборке и 60 значений, равномерных в логарифмической шкале; канонический кандидат, если у оценки есть теоретическая граница ($\chi^2_{0.999}(2)$ у NIS, 3 у фильтра Хэмпела, 0.5 у Isolation Forest и HMM, 0 у One-Class SVM). Выбирается наибольший MCC, среди кандидатов в полосе MCC (не дальше 0.02 от лучшего) — наименьшая операционная ошибка длины. На тестовой выборке — тот же порог без подбора.

Отвергнутая точка не используется в оценках следующих точек. У моделей прогноза (AR, ARIMA, фильтр Калмана, гауссовский процесс, GP-EKF) она не входит в прогноз; прогноз начинается заново после разрыва по времени больше 180 с, а также после 180 с подряд отвергнутых точек — если точка достижима от последней принятой со скоростью не выше наибольшей скорости штатного хода обучающей выборки. У моделей по признакам решение принимается в два прохода: признаки пересчитываются от последних принятых точек.

Признаки моделей по признакам (HMM, DBSCAN, LOF, Isolation Forest, One-Class SVM, Robust PCA, LSTM-автоэнкодер, TCN, DONUT):

$$
\mathbf f_i=\bigl(v_i,\ a^{\mathrm{long}}_i,\ a^{\mathrm{lat}}_i\bigr),\qquad a^{\mathrm{long}}_i=\frac{v_i-v_{i-1}}{\Delta t_i},\qquad a^{\mathrm{lat}}_i=\frac{v_i\,\Delta\psi_i}{\Delta t_i},\qquad \mathbf u_i=\frac{\mathbf f_i-m}{\mathrm{IQR}},
$$

где $v_i$ — скорость на шаге от предыдущей точки в плоскости ENU, $\Delta\psi_i$ — изменение курса между соседними шагами, $m$ и $\mathrm{IQR}$ — медиана и межквартильный размах на штатном ходу обучающей выборки.

Точка без оценки — точка, в которой модель не может дать оценку (начало отрезка, история окна); в метрики классификации не входит, в заполненной траектории восстанавливается как отвергнутая.

# Maximum Physically Consistent Trajectory

Custers et al. ищут самую длинную подпоследовательность точек, в которой скорость между соседними выбранными точками не выше $v_+$; точки вне неё — аномалии.

## Адаптация в блокноте

Подпоследовательность $i_1<\ldots<i_m$ точек отрезка допустима, если

$$
\frac{\lVert\mathbf p_{i_{k+1}}-\mathbf p_{i_k}\rVert}{t_{i_{k+1}}-t_{i_k}}\le v_+,\qquad \mathbf p_i=(x_i,y_i).
$$

Выбирается допустимая подпоследовательность наибольшей длины, $\hat a_i=1$ для точек вне неё. Динамическое программирование: предшественник ищется среди точек за последние 180 с, из более ранних — только конец самой длинной цепочки и только если он достижим со скоростью не выше $v_+$ (вместо полного перебора за $O(n^2)$). Непрерывной оценки нет; $v_+$ — из 30 значений от 95-го процентиля скоростей обучающих отрезков до трёх их максимумов, выбор — по общему правилу. Свойство: кинематика (ускорение, курс) не учитывается, длинные смещённые эпизоды, согласованные внутри, принимаются.

**Источник:** Custers B., van de Kerkhof M., Meulemans W., Speckmann B., Staals F. Maximum Physically Consistent Trajectories // ACM Transactions on Spatial Algorithms and Systems. 2021. Vol. 7, No. 4. Article 17. 33 p. DOI: [10.1145/3452378](https://doi.org/10.1145/3452378). [PDF](docs/methods/Maximum%20Physically%20Consistent%20Trajectories.pdf).

# AR(p) на уровнях ENU

Авторегрессия на уровнях координат; детектор — причинный прогноз и нормированная ошибка на шаге.

## Адаптация в блокноте

AR(2) по каждой оси — стационарная модель уровня с притяжением к первой точке отрезка $\mu_x$:

$$
x_j=\mu_x+\eta_j,\qquad \eta_j=\phi_1\eta_{j-1}+\phi_2\eta_{j-2}+\varepsilon_j,\qquad \varepsilon_j\sim\mathcal N(0,\sigma_x^2).
$$

Ряд переносится на сетку с шагом 10 с; в пустых узлах прогноз продолжается без обновления. Оценка аномальности — квадрат нормированной инновации (NIS):

$$
s_j=\frac{(x_j-\hat x_j)^2}{D_{x,j}}+\frac{(y_j-\hat y_j)^2}{D_{y,j}},\qquad \hat a_j=\mathbf1[s_j>\tau],
$$

где $D_{x,j}$, $D_{y,j}$ — дисперсии прогноза осей, растут с горизонтом. Параметры — по максимуму правдоподобия на обучающих отрезках, разрезанных на каждом шаге времени, отличном от 10 с: по участкам «точка — пропуск — точка» правдоподобие выбирало бы $\phi\approx(0,1)$. На данных `2.csv` оценка $\phi\approx(2,-1)$ — модель постоянной скорости.


# ARIMA для прогнозирования и anomaly detection

Модель прогнозирует следующее значение по предыдущим и помечает точку как аномальную при большом расхождении с прогнозом.

Qin et al. не фиксируют порядок ARIMA(2,1,1). В статье: скользящее окно, порядок по AIC/BIC, переобучение после сдвига окна, short-step прогноз, экспоненциально взвешенное усреднение предсказаний, детекция по относительной ошибке (порог порядка 5–15% в зависимости от ряда).

Адаптация: порядок (2,1,1) фиксирован, модель не переобучается по скользящему окну; оценка — NIS прогноза, а не относительная ошибка статьи.

Для одномерного ряда модель ARIMA(p,d,q):

$$
\phi(B)(1-B)^d x_t = c+\theta(B)\varepsilon_t, \qquad \varepsilon_t\sim\mathcal N(0,\sigma^2),
$$

$$
\phi(B)=1-\phi_1B-\ldots-\phi_pB^p, \qquad \theta(B)=1+\theta_1B+\ldots+\theta_qB^q.
$$

Фиксированный вариант:

$$
\Delta x_t = c+\phi_1\Delta x_{t-1}+\phi_2\Delta x_{t-2}+\varepsilon_t+\theta_1\varepsilon_{t-1}.
$$

$$
r_t=x_t-\hat x_t.
$$

## Адаптация в блокноте

Две независимые ARIMA(2,1,1) без дрейфа на координатах ENU:

$$
\Delta x_j=\phi_1\Delta x_{j-1}+\phi_2\Delta x_{j-2}+\varepsilon_j+\theta_1\varepsilon_{j-1},\qquad \Delta x_j=x_j-x_{j-1}.
$$

Член $\theta_1\varepsilon_{j-1}$ описывает шум GPS. Сетка, перезапуск, оценка аномальности (NIS) и оценка параметров — как у AR(2); первые три точки отрезка — без оценки.

**Источник:** Qin Yu, Lyu Jibin, Lirui Jiang. An Improved ARIMA-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks // International Journal of Distributed Sensor Networks. 2016. Vol. 2016. Article ID 9653230. 9 p. DOI: [10.1155/2016/9653230](https://doi.org/10.1155/2016/9653230). [PDF](../papers/methods/An%20Improved%20ARIMA-Based%20Traffic%20Anomaly%20Detection%20Algorithm%20for%20Wireless%20Sensor%20Networks.pdf).

# Калмановский фильтр с моделью постоянной скорости

Фильтр ведёт скрытое состояние «положение и скорость» и считает измерение аномальным, если оно плохо согласуется с предсказанным продолжением движения. Калман задаёт фильтр, но не детектор аномалий: normalized innovation squared — статистическая адаптация.

## Общая математическая постановка

Состояние и модель постоянной скорости:

$$
\mathbf x_i= \begin{pmatrix} p_x\\ p_y\\ v_x\\ v_y \end{pmatrix}_i, \qquad \mathbf x_i=A_i\mathbf x_{i-1}+\mathbf w_i, \qquad \mathbf w_i\sim\mathcal N(0,Q_i),
$$

$$
A_i= \begin{pmatrix} 1&0&\Delta t_i&0\\ 0&1&0&\Delta t_i\\ 0&0&1&0\\ 0&0&0&1 \end{pmatrix}.
$$

Измерение:

$$
\mathbf z_i=H\mathbf x_i+\mathbf v_i, \qquad \mathbf v_i\sim\mathcal N(0,R), \qquad H= \begin{pmatrix} 1&0&0&0\\ 0&1&0&0 \end{pmatrix}.
$$

Инновация и её ковариация:

$$
\mathbf r_i=\mathbf z_i-H\hat{\mathbf x}_{i|i-1}, \qquad S_i=HP_{i|i-1}H^T+R.
$$

$$
s_i=\mathbf r_i^T S_i^{-1}\mathbf r_i.
$$

При корректно заданной гауссовской модели $s_i\sim\chi^2_2$.

## Адаптация в блокноте

Состояние $\mathbf x_i=(x_i,y_i,v^x_i,v^y_i)^\top$ в плоскости ENU; шум процесса — дискретное белое ускорение без ограничения шага:

$$
Q_i=\sigma_{\mathrm{acc}}^2\begin{pmatrix}\tfrac{\Delta t_i^4}{4}I&\tfrac{\Delta t_i^3}{2}I\\\tfrac{\Delta t_i^3}{2}I&\Delta t_i^2 I\end{pmatrix},\qquad R=\sigma_{\mathrm{meas}}^2 I.
$$

Оценка аномальности — NIS, $\hat a_i=\mathbf1[s_i>\tau]$; $\chi^2_{0.999}(2)$ — канонический кандидат порога. $\sigma_{\mathrm{acc}}$, $\sigma_{\mathrm{meas}}$ — по максимуму правдоподобия на обучающих отрезках. Начальное состояние — по первым двум точкам отрезка, они без оценки; отвергнутая точка состояние не обновляет; перезапуск — по общему правилу.

# Gaussian Process Regression

Нормальная траектория описывается функцией времени с оценкой неопределённости. Точка аномальна, если лежит далеко от этой функции относительно предсказанного разброса.

## Общая математическая постановка

$$
f(t)\sim GP(m(t),k(t,t')), \qquad k(t,t') = \sigma_f^2 \exp \left( -\frac{(t-t')^2}{2\ell^2} \right).
$$

$$
z_i=f(t_i)+\varepsilon_i, \qquad \varepsilon_i\sim\mathcal N(0,\sigma_n^2), \qquad f(t_*)\mid\mathcal D \sim \mathcal N(\mu_*,\sigma_*^2).
$$

## Адаптация в блокноте

Гауссовский процесс по окну из последних 18 принятых точек отрезка прогнозирует положение следующей точки; по каждой оси — линейный тренд (универсальный кригинг) и RBF с шумом:

$$
x(t)=\beta_0+\beta_1(t-\bar t)+f(t),\qquad k(t,t')=\sigma_{\mathrm{rbf}}^2\exp\Bigl(-\frac{(t-t')^2}{2\ell^2}\Bigr)+\sigma_{\mathrm n}^2\,\delta_{tt'}.
$$

$$
s_i=\sum_{\text{оси}}\frac{(z_i-\mu_i)^2}{\sigma_i^2+\sigma_{\mathrm n}^2},\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

Гиперпараметры общие для осей — по максимуму суммы логарифмов прогнозной плотности на окнах обучающих отрезков. Отвергнутая точка в окно не входит; первые три точки отрезка — без оценки.

**Источники:**

- Smith M., Reece S., Roberts S., Psorakis I., Rezek I. Maritime Abnormality Detection Using Gaussian Processes // Knowledge and Information Systems. 2014. Vol. 38, No. 3. P. 717–741. DOI: [10.1007/s10115-013-0685-z](https://doi.org/10.1007/s10115-013-0685-z).
- Penacho Riveiros A., Bastianello N., Barreau M. Model-free Anomaly Detection for Dynamical Systems with Gaussian Processes. 2026. 6 p. [PDF](../papers/methods/Model-free%20Anomaly%20Detection%20for.pdf).


# GP-EKF, GP-BayesFilters

Ko и Fox не детектируют аномалии: они подставляют гауссовские процессы вместо параметрических моделей прогноза и наблюдения внутрь фильтра Байеса. В статье три фильтра: GP-PF, GP-EKF и GP-UKF. В опыт входит GP-EKF. Его отличие от обычного EKF — якобиан среднего GP, а не якобиан заранее заданной функции.

В статье нет входа управления, отдельного от состояния. У трека управления нет, поэтому аргумент процесса — скорость. Наблюдение ГНСС линейно по положению, второй GP для него не учится: в статье GP-наблюдение нужно, когда отображение состояния в измерение неизвестно.

## Общая математическая постановка

Фильтр Байеса:

$$
p(x_k\mid z_{1:k},u_{1:k-1}) \propto p(z_k\mid x_k)\int p(x_k\mid x_{k-1},u_{k-1})\,p(x_{k-1}\mid z_{1:k-1})\,dx_{k-1}.
$$

Прогноз GP — гауссовское распределение перехода. Наблюдение в статье тоже GP. Для скалярного выхода и квадратично-экспоненциального ядра

$$
k(x,x')=\sigma_f^2\exp\Bigl(-\frac12(x-x')W(x-x')^T\Bigr)+\sigma_n^2\delta,
$$

$$
\mathrm{GP}_\mu(x_*,D)=k_*^T K^{-1}y, \qquad \mathrm{GP}_\sigma(x_*,D)=k(x_*,x_*)-k_*^T K^{-1}k_*.
$$

Якобиан среднего по входу, нужный EKF:

$$
\frac{\partial \mathrm{GP}_\mu(x_*,D)}{\partial x_*} = \frac{\partial k_*}{\partial x_*}^{\!T} K^{-1}y, \qquad \frac{\partial k(x_*,x)}{\partial x_*[i]}=-W_{ii}(x_*[i]-x[i])\,\sigma_f^2\exp\Bigl(-\frac12(x_*-x)W(x_*-x)^T\Bigr).
$$

Шаг GP-EKF. Процесс учит приращение состояния, поэтому линеаризация содержит единичную матрицу:

$$
\bar\mu_k=\mu_{k-1}+\mathrm{GP}_\mu([\mu_{k-1},u_{k-1}],D_p), \qquad Q_k=\mathrm{GP}_\sigma([\mu_{k-1},u_{k-1}],D_p),
$$

$$
G_k=I+\frac{\partial\mathrm{GP}_\mu}{\partial x_{k-1}}, \qquad \bar\Sigma_k=G_k\Sigma_{k-1}G_k^T+Q_k.
$$

Дальше обычное обновление EKF: прогноз измерения, $R_k$, якобиан наблюдения $H_k$, коэффициент Калмана, новое среднее и ковариация. Выходы по координатам — независимые GP, поэтому $Q_k$ диагональна.

Усиленный GP (enhanced GP) учит не само приращение, а остаток после параметрической модели $f$:

$$
\Delta\tilde x_k=x_{k+1}-x_k-f(x_k,u_k).
$$

## Адаптация в блокноте

Усиленный GP над моделью постоянной скорости фильтра Калмана: ускорение-поправка $\mathbf g=(g^x,g^y)^\top$ по каждой оси — независимый GP от скорости $\mathbf v=(v^x,v^y)$:

$$
\mathbf x_i=F_i\mathbf x_{i-1}+B_i\,\mathbf g(\mathbf v_{i-1})+\mathbf w_i,\qquad B_i=\begin{pmatrix}\tfrac{\Delta t_i^2}{2}I\\\Delta t_i I\end{pmatrix}.
$$

Опорные точки — переходы сглаженного состояния (сглаживатель RTS с оценками фильтра Калмана) на обучающих отрезках, разнесённые не меньше чем на 120 с, до 1000. Гиперпараметры GP — по маргинальному правдоподобию; шум ускорения по осям и $\sigma_{\mathrm{meas}}$ — по правдоподобию фильтра. Оценка аномальности — NIS, как у фильтра Калмана.

Результат на данных `2.csv`: длины корреляции поправки малы (0.07–0.13 м/с), доля дисперсии ускорения, объяснённая поправкой, отрицательна — ускорение не зависит от скорости, GP-EKF сводится к фильтру Калмана.

**Источник:** Ko J., Fox D. GP-BayesFilters: Bayesian Filtering Using Gaussian Process Prediction and Observation Models // Autonomous Robots. 2009. Vol. 27, No. 1. P. 75–90. DOI: [10.1007/s10514-009-9119-x](https://doi.org/10.1007/s10514-009-9119-x). [PDF](docs/GP-BayesFilters_Bayesian_Filtering_Using_Gaussian_.pdf).

# Hidden Markov Model

Toloue и Jahan строят пять HMM с дискретными наблюдениями $(S,\Delta S,\mathrm{Drift},\Delta C)$: unexpected stop, risky speed, drift, spiral, normal. Модели обучаются Baum–Welch и сравниваются алгоритмом Forward на всей траектории.

Адаптация идеи морской HMM, а не формула статьи: одна гауссовская HMM с режимами штатного хода и аномалии, оценка — причинная апостериорная вероятность режима аномалии в каждой точке.

## Общая математическая постановка

$$
S_i\in\{1,\ldots,K\}, \qquad \pi_k=P(S_1=k), \qquad A_{jk}=P(S_i=k\mid S_{i-1}=j).
$$

$$
o_i\mid S_i=k \sim \mathcal N(\mu_k,\mathrm{diag}(\sigma_k^2)).
$$

$$
p(o_{1:N}\mid\theta) = \sum_{s_{1:N}} \pi_{s_1} \prod_{i=2}^{N}A_{s_{i-1}s_i} \prod_{i=1}^{N} p(o_i\mid S_i=s_i).
$$

## Адаптация в блокноте

Наблюдение — нормированные признаки $\mathbf u_i$; по три режима штатного хода и аномалии в одной цепи:

$$
S_i\in\mathcal S_{\mathrm{move}}\cup\mathcal S_{\mathrm{anom}},\qquad \mathbf u_i\mid S_i=k\sim\mathcal N(\mathbf m_k,D_k),
$$

$$
f_i(k)\propto\mathcal N(\mathbf u_i;\mathbf m_k,D_k)\sum_j f_{i-1}(j)\,T_{jk},\qquad s_i=\sum_{k\in\mathcal S_{\mathrm{anom}}}f_i(k),\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

EM (Baum–Welch) по всем отрезкам обучающей выборки; в размеченной точке режимы чужого класса запрещены; 5 запусков, лучший по правдоподобию. Признаки при обучении — по предыдущей точке ряда, как при обнаружении; точка штатного хода после серии аномалий — без наблюдения, цепь не рвётся. Ограничение: внутри длинного эпизода признаки одной точки неотличимы от штатного хода, оценка почти нулевая на обоих классах.

**Источник:** Toloue K. F., Jahan M. V. Anomalous Behavior Detection of Marine Vessels Based on Hidden Markov Model // 2018 6th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2018. P. 10–12. [PDF](docs/methods/AnomalousBehaviorDetectionofMarineVessels.pdf).

# Hampel Filter

В скользящем окне точка сравнивается с медианой соседей. Выброс — сильное отклонение относительно типичного разброса окна.

Roos-Hoefgeest Toribio et al. ускоряют фильтр, заменяя MAD оценкой mMAD. Ниже — стандартный Hampel, не эта замена. Центрированное окно $x_{i-h},\ldots,x_{i+h}$ использует будущие точки. Для online-потока нужно хвостовое окно либо задержка решения на $h$ отсчётов.

## Общая математическая постановка

$$
W_i=\{x_{i-h},\ldots,x_i,\ldots,x_{i+h}\}, \qquad m_i=\operatorname{median}(W_i),
$$

$$
MAD_i=\operatorname{median}_{x\in W_i}|x-m_i|, \qquad s_i=\frac{|x_i-m_i|}{1.4826\,MAD_i+\varepsilon}.
$$

Точка аномальна при $s_i>\tau$.

## Адаптация в блокноте

Фильтр применяется к модулю скорости в плоскости ENU, полуокно $w=10$, у краёв отрезка окно сдвигается; вместо $\varepsilon$ — нижняя граница разброса $\sigma_{\min}$ (5-й перцентиль $1.4826\,\mathrm{MAD}$ на штатном ходу):

$$
s_i=\frac{|v_i-m_i|}{\max\bigl(1.4826\,\mathrm{MAD}_i,\ \sigma_{\min}\bigr)},\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

Решение — в два прохода. На данных `2.csv` фильтр неприменим: аномалии — длинные эпизоды, окно лежит внутри эпизода, и аномалии сравниваются с аномалиями.

**Источник:** Roos-Hoefgeest Toribio M., Garnung Menéndez A., Roos-Hoefgeest Toribio S., Álvarez García I. A Novel Approach to Speed Up Hampel Filter for Outlier Detection // Sensors. 2025. Vol. 25. Article 3319. DOI: [10.3390/s25113319](https://doi.org/10.3390/s25113319). [PDF](docs/methods/A%20Novel%20Approach%20to%20Speed%20Up%20Hampel%20Filter%20for%20Outlier%20Detection.pdf).

# DBSCAN

Точки группируются по плотности соседей. Аномалии — точки вне плотных кластеров. В оригинале это batch-метод: окрестность считается по всему набору.

## Общая математическая постановка

$$
N_\varepsilon(x)=\{x'\in X:d(x,x')\le\varepsilon\}.
$$

Точка — core point при $|N_\varepsilon(x)|\ge\mathrm{MinPts}$. Шум: $c_i=-1$.

## Адаптация в блокноте

Обнаружение новизны: основные точки (core points, $\mathrm{MinPts}=6$) — по признакам $\mathbf u$ штатного хода обучающих отрезков; Han et al. кластеризуют AIS по $(\mathrm{lat},\mathrm{lon},\mathrm{SOG},\mathrm{COG},\mathrm{Heading})$, здесь абсолютные координаты не используются. Оценка — расстояние до ближайшей основной точки, порог совпадает с радиусом:

$$
s_i=\min_{c\in C_\varepsilon}\lVert\mathbf u_i-\mathbf u_c\rVert,\qquad \hat a_i=\mathbf1[s_i>\varepsilon].
$$

$\varepsilon$ — из 30 квантилей расстояния до 6-го соседа на штатном ходу, от медианы до 99.9-го перцентиля; решение — в два прохода.

**Источники:**

- Ester M., Kriegel H.-P., Sander J., Xu X. A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise // Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96). AAAI Press, 1996. P. 226–231. [PDF](docs/methods/A%20Density-Based%20Algorithm%20for%20Discovering%20Clusters.pdf).
- Han X., Armenakis C., Jadidi M. DBSCAN Optimization for Improving Marine Trajectory Clustering and Anomaly Detection // The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. 2020. Vol. XLIII-B4-2020. P. 455–461. DOI: [10.5194/isprs-archives-XLIII-B4-2020-455-2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-455-2020). [PDF](docs/methods/DBSCAN%20OPTIMIZATION%20FOR%20IMPROVING%20MARINE%20TRAJECTORY%20CLUSTERING.pdf).


# Local Outlier Factor

Степень аномальности точки — насколько её локальная плотность ниже плотности ближайших соседей. LOF не является временной моделью: время входит только через признаки.

## Общая математическая постановка

$N_k(x)$ — $k$ ближайших соседей.

$$
\operatorname{reachdist}_k(x,y)=\max\{k\text{-distance}(y),d(x,y)\}.
$$

$$
\mathrm{lrd}_k(x)=\left( \frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \operatorname{reachdist}_k(x,y) \right)^{-1}.
$$

$$
\mathrm{LOF}_k(x)=\frac{1}{|N_k(x)|} \sum_{y\in N_k(x)} \frac{\mathrm{lrd}_k(y)}{\mathrm{lrd}_k(x)}.
$$

При $\mathrm{LOF}_k(x)\approx 1$ плотность точки близка к соседям.

## Адаптация в блокноте

Обнаружение новизны: соседи ищутся в наборе $T$ признаков штатного хода обучающих отрезков; для точки из $T$ сама точка в соседи не входит.

$$
s_i=\mathrm{LOF}_k(\mathbf u_i),\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

$k\in\{10,20,35,50\}$ и порог подбираются совместно; решение — в два прохода.

**Источник:** Breunig M. M., Kriegel H.-P., Ng R. T., Sander J. LOF: Identifying Density-Based Local Outliers // Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. Dallas, USA. ACM, 2000. P. 93–104. DOI: [10.1145/342009.335388](https://doi.org/10.1145/342009.335388). [PDF](docs/methods/LOF_ Identifying Density-Based Local Outliers.pdf).

# Isolation Forest

Случайные деревья отделяют точки разрезами по признакам. Аномалия отделяется за меньшее число разрезов. В оригинале метод не требует отдельной выборки только нормальных точек.

## Общая математическая постановка

Средняя длина пути $E[h(x)]$, нормировка через гармоническое число $H$:

$$
c(n)=2H(n-1)-\frac{2(n-1)}{n}, \qquad s(x)=2^{-\frac{E[h(x)]}{c(n)}}.
$$

## Адаптация в блокноте

Обнаружение новизны: 200 деревьев на подвыборках $n_{\mathrm{sub}}=256$ точек набора $T$ (как у LOF); оценка — по статье, без сдвига на $1/2$:

$$
s_i=2^{-\bar h(\mathbf u_i)/c(n_{\mathrm{sub}})},\qquad \hat a_i=\mathbf1[s_i>\tau];
$$

0.5 — канонический кандидат порога; решение — в два прохода.

**Источник:** Liu F. T., Ting K. M., Zhou Z.-H. Isolation-Based Anomaly Detection // ACM Transactions on Knowledge Discovery from Data. 2012. Vol. 6, No. 1. Article 3. 39 p. DOI: [10.1145/2133360.2133363](https://doi.org/10.1145/2133360.2133363). [PDF](docs/methods/Isolation-Based%20Anomaly%20Detection.pdf).

# One-Class SVM

На нормальных точках оценивается область высокой концентрации обучающего распределения. Объект вне этой области считается аномалией.

## Общая математическая постановка

$$
\min_{\mathbf w,\rho,\xi} \frac12\|\mathbf w\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n}\xi_i -\rho, \qquad \mathbf w^T\phi(x_i)\ge\rho-\xi_i, \quad \xi_i\ge0.
$$

Параметр $\nu\in(0,1]$ — верхняя граница доли ошибок на обучении и нижняя граница доли support vectors.

$$
f(x)=\sum_i\alpha_i K(x_i,x)-\rho.
$$

Аномалия при $f(x)<0$.

## Адаптация в блокноте

Модель обучается на всех точках набора $T$ (как у LOF), ядро RBF, $\gamma=1/(d\,\mathrm{Var}(T))$, $\nu=0.1$ фиксирован:

$$
s_i=-f(\mathbf u_i),\qquad \hat a_i=\mathbf1[s_i>\tau];
$$

0 — канонический кандидат порога; порог выбирается по общему правилу; решение — в два прохода.

**Источник:** Schölkopf B., Platt J. C., Shawe-Taylor J., Smola A. J., Williamson R. C. Estimating the Support of a High-Dimensional Distribution // Neural Computation. 2001. Vol. 13, No. 7. P. 1443–1471. DOI: [10.1162/089976601750264965](https://doi.org/10.1162/089976601750264965). В `../papers/methods` лежит технический отчёт Microsoft Research MSR-TR-99-87 (27 November 1999; revised 18 September 2000): [PDF](docs/methods/Estimating%20the%20Support%20of%20a%20High-Dimensional%20Distribution.pdf).

# Robust PCA / Principal Component Pursuit

Матрица раскладывается на низкоранговую часть и разреженные отклонения. Крупные элементы разреженной части — кандидаты в аномалии.

## Общая математическая постановка

$$
X=L+S, \qquad \min_{L,S} \|L\|_* + \lambda\|S\|_1.
$$

В классической постановке Candès et al. $\lambda=1/\sqrt{\max(n_1,n_2)}$.

## Адаптация в блокноте

Матрица $X$ — окна штатного хода обучающей выборки, по $n_W=16$ подряд идущих точек ($\mathbf x_W=(\mathbf u_j,\ldots,\mathbf u_{j+15})$); PCP задаёт подпространство $U$. Окно любой выборки — остаток $\ell_1$-проекции на $U$ (линейное программирование):

$$
\mathbf c^*=\arg\min_{\mathbf c}\lVert\mathbf x_W-U\mathbf c\rVert_1,\qquad \mathbf s_W=\mathbf x_W-U\mathbf c^*,\qquad s_i=\max_{W\ni i}\lVert\mathbf s_W(i)\rVert,\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

Окна — с шагом 1 внутри отрезка; решение — в два прохода.

**Источник:** Candès E. J., Li X., Ma Y., Wright J. Robust Principal Component Analysis? // Journal of the ACM. 2011. Vol. 58, No. 3. Article 11. P. 1–37. DOI: [10.1145/1970392.1970395](https://doi.org/10.1145/1970392.1970395). В `../papers/methods` лежит препринт arXiv:0912.3599v1 (18 December 2009): [PDF](docs/methods/Robust%20Principal%20Component%20Analysis.pdf).

# LSTM Autoencoder / EncDec-AD

Сеть учится восстанавливать окна нормальной траектории. Аномалия — окно с нетипичной ошибкой восстановления.

## Общая математическая постановка

$$
X_i=(x_{i-L+1},\ldots,x_i), \qquad h_i=E_\theta(X_i), \qquad \hat X_i=D_\phi(h_i).
$$

Обучение: $\min_{\theta,\phi}\sum_i\lVert X_i-\hat X_i\rVert_2^2$.

В EncDec-AD Malhotra et al. вектор ошибки $e=X-\hat X$ на нормальной проверочной выборке описывается средним $\mu$ и ковариацией $\Sigma$, score — расстояние Махаланобиса. Сырой $\lVert e\rVert_2^2$ — упрощение без этой нормировки.

## Адаптация в блокноте

Окно — признаки $\mathbf u$ 16 подряд идущих точек (как у Robust PCA), скрытое состояние размера 32. Ошибка нормируется по позиции в окне (диагональная ковариация вместо полной) по калибровочным окнам — последним 15% окон штатного хода:

$$
d_W(i)=\sum_{k=1}^{3}\frac{\bigl(e_{W,k}(i)-\mu_{p,k}\bigr)^2}{\sigma_{p,k}^2},\qquad s_i=\min_{W\ni i}d_W(i),\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

Минимум по окнам относит отклонение к точке выброса. Обучение — до 8000 окон, до 300 эпох с ранней остановкой по ошибке на калибровочных окнах.

**Источник:** Malhotra P., Ramakrishnan A., Anand G., Vig L., Agarwal P., Shroff G. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection // ICML 2016 Anomaly Detection Workshop. New York, 2016. arXiv: [1607.00148](https://arxiv.org/abs/1607.00148). [PDF](docs/methods/LSTM-based%20Encoder-Decoder%20for%20Multi-sensor%20Anomaly%20Detection.pdf).

# Temporal Convolutional Network

Каузальная свёрточная сеть предсказывает продолжение нормального ряда. Аномалия — остаток, редкий для распределения ошибок на нормальных данных.

He и Zhao прогнозируют несколько шагов вперёд и смешивают признаки нескольких масштабов. Один шаг — частный случай.

## Общая математическая постановка

$$
\hat x_{t+1:t+H}=f_\theta(x_{t-L+1},\ldots,x_t), \qquad r_{t+h}=x_{t+h}-\hat x_{t+h}.
$$

Остатки на нормальных данных описываются диагональным гауссовским законом. Нулевое среднее не предполагается: $\mu$ и $\sigma^2$ оцениваются.

## Адаптация в блокноте

Один шаг: по признакам 15 предыдущих принятых точек прогнозируются признаки следующей; три причинных свёрточных уровня (ядро 3, расширения 1, 2, 4, 32 канала):

$$
\hat{\mathbf u}_i=g\bigl(\mathbf u_{i-15},\ldots,\mathbf u_{i-1}\bigr),\qquad s_i=\sum_{k=1}^{3}\frac{\bigl(u_{i,k}-\hat u_{i,k}-\mu_k\bigr)^2}{\sigma_k^2},\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

$\mu_k$, $\sigma_k^2$ — по калибровочным окнам; обучение — как у LSTM-автоэнкодера. Первые 17 точек отрезка — без оценки.

**Источник:** He Y., Zhao J. Temporal Convolutional Networks for Anomaly Detection in Time Series // Journal of Physics: Conference Series. 2019. Vol. 1213. Article 042050. DOI: [10.1088/1742-6596/1213/4/042050](https://doi.org/10.1088/1742-6596/1213/4/042050). [PDF](docs/methods/Temporal_Convolutional_Networks_for_Anomaly_Detect.pdf).

# DONUT — VAE для anomaly detection

Вариационный автоэнкодер задаёт распределение окон. Окно с низкой вероятностью считается аномальным. DONUT сделан для сезонных KPI веб-приложений: modified ELBO, учёт пропусков и KDE-интерпретация реконструкции. Перенос на окна признаков траектории — адаптация, не алгоритм статьи; реализация — PyTorch, не пакет NetManAIOps.

## Общая математическая постановка

$$
q_\phi(z|x)=\mathcal N\left( \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)) \right), \qquad p_\theta(x|z).
$$

$$
\mathcal L(x)=\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}\left( q_\phi(z|x)\|p(z) \right).
$$

## Адаптация в блокноте

Окно — признаки 15 предыдущих принятых точек и точки $i$; скрытый вектор размерности 8, кодировщик и декодер — по два слоя из 100 нейронов. Обучение — M-ELBO, как в статье: аномальные точки исключаются из правдоподобия ($\alpha_j=0$), ещё 1% точек случайно получает $\alpha_j=0$:

$$
\mathcal L(\mathbf x_W)=\mathbb E_{q(z\mid\mathbf x_W)}\Bigl[\sum_{j\in W}\alpha_j\log p(\mathbf u_j\mid z)\Bigr]-\beta\,D_{\mathrm{KL}}\bigl(q(z\mid\mathbf x_W)\,\Vert\,\mathcal N(0,I)\bigr),\qquad \beta=\frac{1}{|W|}\sum_{j\in W}\alpha_j.
$$

Оценка — минус вероятность восстановления последней точки окна по 100 выборкам $z$:

$$
s_i=-\frac{1}{L}\sum_{l=1}^{L}\log p(\mathbf u_i\mid z_l),\qquad z_l\sim q(z\mid\mathbf x_W),\qquad \hat a_i=\mathbf1[s_i>\tau].
$$

**Источник:** Xu H., Chen W., Zhao N., Li Z., Bu J., Li Z., Liu Y., Zhao Y., Pei D., Feng Y., Chen J., Wang Z., Qiao H. Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications // Proceedings of The Web Conference 2018 (WWW '18). Lyon, France, 23–27 April 2018. ACM, 2018. 12 p. DOI: [10.1145/3178876.3185996](https://doi.org/10.1145/3178876.3185996). arXiv: [1802.03903](https://arxiv.org/abs/1802.03903). [PDF](docs/methods/Unsupervised%20Anomaly%20Detection%20via%20Variational%20Auto-Encoder.pdf).
