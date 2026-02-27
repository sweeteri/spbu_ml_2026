#%% md
# # Ужасы линейной регрессии
#%%
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(42)
#%% md
# # Разминка
# 1) Какие признаки может обработать линейная регрессия?
# 2) Почему нет понятия полиномиальных моделей?
# 3) Вам дан набор данных из 8 элементов. Вы используете стохастический градиентный спуск с батчем 8. Сколько шагов нужно сделать, чтобы пройти весь набор данных?
# 4) Вам нужно предсказать, к какой категории относится товар. Всего категорий 100. Как лучше решать такую задачу - через регрессию или классификацию?
# 5) Вам нужно предсказать, в каком часу была сделана покупка некоторого товара. Вы решили использовать для этого регрессию. Расскажите, как вы будете решать эту задачу. Как вы сформируете таргет? Как будете выдавать предсказания на инференсе?
# 6) Как выглядит точное решение задачи линейной регрессии (нормальная форма)?
#%% md
# # База
# Напомню, что линейная регрессия - это модель следующего вида:
#     $$a(x) = \langle w, x \rangle + w_0$$,
#     где $w \in \mathbb{R}^d$, $w_0 \in \mathbb{R}$. 
# Обучить линейную регрессию - значит найти $w$ и $w_0$., минимизирующие разницу между предсказанными значениями  $y'$ и  $y$.
#%% md
# # Ловушки в данных
#%% md
# Рассмотрим набор данных, состоящий из 2 разных предикторов и четырех таргетов. Этот набор специально создан, чтобы показать важность анализа данных даже в, казалось бы, простой ситуации. Построим 4 разные модели по парам (x123, y1), (x123, y2),(x123, y3), (x4, y4).
#%%
data_path = r"C:\Users\2005k\PycharmProjects\spbu_ml_2026\data"
#%%
quartet_df = pd.read_csv(f"{data_path}/Anscombe_quartet_data.csv")
#%% md
# Для начала рассмотрим значения корреляций между признаками и таргетами
#%%
def corrplot(d: pd.DataFrame) -> None:
  corr = d.corr()
  mask = np.triu(np.ones_like(corr, dtype=bool))
  figsize = (min(d.shape[1], 11), min(d.shape[1], 9))
  f, ax = plt.subplots(figsize=figsize)
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
#%%
corrplot(quartet_df)
#%% md
# Как видно, корреляции одинаковы. Однако, если построить модели и графики предсказаний, результаты будут знгачительно отличаться.
# ЭТакже выведем значения коэффициентов модели.
#%% md
# **Задание**: Постройте линейные модели для каждого из наборов данных, не смотря на графики. Отличаются ли модели?
#%%
def fit_and_plot_anscombe(x_series, y_series, title):
    x = np.array(x_series).reshape((-1, 1))
    y = np.array(y_series)
    model = LinearRegression().fit(
        x, y
    )
    y_pred = model.predict(x)
    r_sq = model.score(x, y)
    
    print(f"{title}")
    print(f"R²: {r_sq:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Slope: {model.coef_[0]:.4f}\n")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="green", alpha=0.7)
    plt.plot(x, y_pred, color="blue", linewidth=2)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model
#%%
fit_and_plot_anscombe(quartet_df['x123'], quartet_df['y1'], "Anscombe 1")
#%%
# Anscombe 2
fit_and_plot_anscombe(quartet_df['x123'], quartet_df['y2'], "Anscombe 2")
#%%
fit_and_plot_anscombe(quartet_df['x123'], quartet_df['y3'], "Anscombe 3")
#%%
fit_and_plot_anscombe(quartet_df['x4'], quartet_df['y4'], "Anscombe 3")
#%% md
# Как видно, все эти зависимости имеют разную природу, но при этом порождают модели с практически одинаковыми параметрами. Таким образом, чрезвычайно важно проводить достаточно объемный EDA.
# Датасте же - известный набор [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)
#%% md
# # Положения линейной регрессии
# Далее рассмотрим некоторые положения линейной регрессии, которые задают границы ее применимости (а также что бывает, когда они нарушаются).
# 
# **Линейность**: взаимосвязь между зависимыми и независимыми переменными линейна.
# 
# **Независимость**: Наблюдения независимы друг от друга.
# 
# **Гомоскедастичность**: дисперсия ошибок постоянна на всех уровнях независимых переменных.
# 
# **Нормальность**: ошибки подчиняются нормальному распределению.
# 
# **Нет мультиколлинеарности**: независимые переменные не сильно коррелируют друг с другом.
# 
# **Нет эндогенности**: между ошибками и независимыми переменными нет связи.
#%% md
# ## Нелинейность
#%% md
# Рассмотрим набор данных Bluegills.
# В 1981 году n = 78 [cинежаберных солнечников](https://ru.wikipedia.org/wiki/%D0%A1%D0%B8%D0%BD%D0%B5%D0%B6%D0%B0%D0%B1%D0%B5%D1%80%D0%BD%D1%8B%D0%B9_%D1%81%D0%BE%D0%BB%D0%BD%D0%B5%D1%87%D0%BD%D0%B8%D0%BA) были случайным образом отобраны из озера Мэри в Миннесоте (Кук и Вайсберг, 1999):
# * Длина (в мм) рыбы
# * Возраст (в годах) рыбы
# 
#%%
df = pd.read_csv(f"{data_path}/bluegills.txt", sep="\t")
#%%
corrplot(df)
#%%
sns.scatterplot(df, x="age", y="length")
#%% md
# График позволяет предположить, что в данных наблюдается положительная динамика. Неудивительно, что с увеличением возраста рыб их длина имеет тенденцию к увеличению. Однако эта тенденция не выглядит вполне линейной. Кажется, что отношения слегка искривлены.
# 
# Один из способов моделирования кривизны этих данных — сформулировать «полиномиальную модель второго порядка»:
# 
# $ Y=\beta_0+ \sum_i (\alpha_i  X_i^2) + \sum_i (\beta_i  X_i) + \sum_{j \neq i} (\gamma_{ij}  X_i X_j) + \epsilon $
# 
# Сделать это просто, введя в данные полиномиальные признаки (PolynomialFeatures) и обучив обычную линейную модель.
#%%
X = np.array(df["age"]).reshape(-1, 1)
y =np.array(df["length"])
model_basic = LinearRegression()
model_basic.fit(X, df["length"])
y_basic = model_basic.predict(X)
#%%
degree = 2  # Set the polynomial degree
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X=np.array(df.loc[:,"age"]).reshape(-1, 1))
X = np.array(df["age"]).reshape(-1, 1)
y =np.array(df["length"])
model_basic = LinearRegression()
model_basic.fit(X, df["length"])
model_poly = LinearRegression()
model_poly.fit(X_poly,  df["length"])
y_basic = model_basic.predict(X)
y_poly = model_poly.predict(X_poly)
#%%
print(f"coefficient of determination of linear model: {model_basic.score(X, y)}")
print(f"coefficient of determination of polynomial model: {model_poly.score(X_poly, y)}")
#%%
sns.scatterplot(df,x="age", y="length", label='Data', color='blue')
sns.lineplot(x=df["age"], y=y_poly, label='Polynomial Regression', color='red', linewidth=2)
sns.lineplot(x=df["age"], y=y_basic, label='Linear Regression', color='green', linewidth=2)
plt.xlabel('age')
plt.ylabel('length')
plt.legend()
plt.title(f'Polynomial Regression (Degree {degree})')
plt.show()
#%%
degree = 2  # Set the polynomial degree
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X=np.array(df.loc[:,"age"]).reshape(-1, 1))
X = np.array(df["age"]).reshape(-1, 1)
y =np.array(df["length"])
model_basic = LinearRegression()
model_basic.fit(X, df["length"])
model_poly = LinearRegression()
model_poly.fit(X_poly,  df["length"])
y_basic = model_basic.predict(X)
y_poly = model_poly.predict(X_poly)
#%% md
# **Задание**: Постройте модель для наборов Anscombe 1 и Anscombe 2. Напишите, какие степени полинома подойдут лучше всего.
#%% md
# **Задание со звездочкой**: Постройте модель с использованием другого базиса, не полиномиального. Проиллюстрируйте ее применение на примерах из практики.
#%% md
# ## Выбросы
# Как мы уже видели, выбросы могут сильно влиять на предсказания линейной модели. Даже один выброс может нарушить одно из условий линейной регрессии. К счастью, существует много сапособов с ними бороться.
# Простейший вариант - использовать не MSE, а MAE. Почему?
# Однако есть и более сложные варианты. Например, комбинация MSE и MAE дает **HuberLoss**:
# 
# $$L_\delta(r) = \begin{cases}
# \frac{1}{2}r^2 & \text{если } |r| \leq \delta \\
# \delta|r| - \frac{1}{2}\delta^2 & \text{если } |r| > \delta
# \end{cases}$$
# 
# где $r = y_i - \mathbf{x}_i^T \beta$ — остаток, $\delta$ — параметр переключения.
# 
# - При малых ошибках ($|r| \leq \delta$) ведет себя как OLS
# - При больших ошибках ($|r| > \delta$) ведет себя как MAE 
# 
#%%
# Визуализация функции потерь Huber
def huber_loss(r, delta):
    """Вычисляет функцию потерь Huber."""
    return np.where(np.abs(r) <= delta, 0.5 * r**2, delta * np.abs(r) - 0.5 * delta**2)

r = np.linspace(-5, 5, 1000)
delta = 1.35

mse_loss = 0.5 * r**2
mae_loss = np.abs(r)
huber = huber_loss(r, delta)

plt.figure(figsize=(12, 6))
plt.plot(r, mse_loss, label='MSE (квадратичная)', linewidth=2, linestyle='--')
plt.plot(r, mae_loss, label='MAE (линейная)', linewidth=2, linestyle='--')
plt.plot(r, huber, label=f'Huber (δ={delta})', linewidth=3)
plt.axvline(x=delta, color='red', linestyle=':', alpha=0.7, label=f'δ = {delta}')
plt.axvline(x=-delta, color='red', linestyle=':', alpha=0.7)
plt.xlabel('Остаток r', fontsize=12)
plt.ylabel('Потери L(r)', fontsize=12)
plt.title('Сравнение функций потерь', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# Влияние параметра delta на функцию потерь
deltas = [0.5, 1.0, 1.35, 2.0]
r = np.linspace(-5, 5, 1000)

plt.figure(figsize=(12, 6))
for delta in deltas:
    huber = huber_loss(r, delta)
    plt.plot(r, huber, label=f'δ = {delta}', linewidth=2)

plt.xlabel('Остаток r', fontsize=12)
plt.ylabel('Потери L(r)', fontsize=12)
plt.title('Влияние параметра δ на функцию потерь Huber', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% md
# 
# Другой робастный вариант - **Theil-Sen regression**. Этот метод строит медиану наклона модели между всеми парами наблюдений. Метод непараметрический. Проблема этого метода в том, что для больших наборов данных вычислительная сложность метода становится слишком большой ($O(n^2)$ для наклона, $O(n \log n)$ для медианы):
# 
# **Алгоритм**:
# 1. Для всех пар точек $(x_i, y_i)$ и $(x_j, y_j)$, где $i < j$, вычисляется наклон:
#    $$b_{ij} = \frac{y_j - y_i}{x_j - x_i}$$
# 2. Оценка наклона: $\hat{\beta}_1 = \text{median}(\{b_{ij}\})$
# 3. Оценка свободного члена: $\hat{\beta}_0 = \text{median}(\{y_i - \hat{\beta}_1 x_i\})$
#%%
# Иллюстрация работы Theil-Sen
np.random.seed(42)
n = 30
X_clean = np.random.uniform(0, 10, n)
y_clean = 2 * X_clean + 1 + np.random.normal(0, 1, n)

# Добавляем выбросы
X_outliers = np.append(X_clean, [1, 9, 5])
y_outliers = np.append(y_clean, [25, -10, 30])

# OLS
ols = LinearRegression()
ols.fit(X_clean.reshape(-1, 1), y_clean)
ols_outliers = LinearRegression()
ols_outliers.fit(X_outliers.reshape(-1, 1), y_outliers)

# Theil-Sen
theil = TheilSenRegressor(random_state=42)
theil.fit(X_clean.reshape(-1, 1), y_clean)
theil_outliers = TheilSenRegressor(random_state=42)
theil_outliers.fit(X_outliers.reshape(-1, 1), y_outliers)

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Без выбросов
axes[0].scatter(X_clean, y_clean, alpha=0.6, s=50, label='Данные')
x_plot = np.linspace(0, 10, 100)
axes[0].plot(x_plot, ols.predict(x_plot.reshape(-1, 1)), 'r--', linewidth=2, label='OLS')
axes[0].plot(x_plot, theil.predict(x_plot.reshape(-1, 1)), 'g-', linewidth=2, label='Theil-Sen')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('Без выбросов', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# С выбросами
axes[1].scatter(X_outliers, y_outliers, alpha=0.6, s=50, label='Данные')
axes[1].scatter([1, 9, 5], [25, -10, 30], color='red', s=100, marker='x', linewidths=3, label='Выбросы')
axes[1].plot(x_plot, ols_outliers.predict(x_plot.reshape(-1, 1)), 'r--', linewidth=2, label='OLS')
axes[1].plot(x_plot, theil_outliers.predict(x_plot.reshape(-1, 1)), 'g-', linewidth=2, label='Theil-Sen')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('С выбросами', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% md
# 
# Третий метод: **RANSAC**. Его идея в том, чтобы сначала построить много моделей-кандидатов, а потом выбрать из них лучшую.
# 
# **Алгоритм**:
# 1. Случайно выбирается минимальная подвыборка для оценки параметров (для линейной регрессии — 2 точки)
# 2. На подвыборке строится модель
# 3. Вычисляются остатки для всех точек. Точки с остатками меньше порога $t$ считаются inliers (внутренними)
# 4. Если количество inliers достаточно велико, модель переоценивается на всех inliers
# 5. Шаги 1-4 повторяются $N$ раз
# 6. Выбирается модель с наибольшим количеством inliers
# 
# Вероятность того, что на всех итерациях до $N$ хотя бы 1 точка будет выбросом (то есть построится некорректная модель) $(1-q)^N$, где q - доля выбросов. Таким образом, мы можем примерно оценить, сколько нам потребуется итераций, а метод становится наиболее толерантным к числу выбросов. Минус в том, что верхняя граница числа итераций фактически не задана, а наша оценка может быть недостаточной.
#%%
# Иллюстрация работы RANSAC
np.random.seed(42)
n = 100
X = np.random.uniform(0, 10, n)
y = 2 * X + 1 + np.random.normal(0, 0.5, n)

# Добавляем выбросы
n_outliers = 30
X_outliers = np.append(X, np.random.uniform(0, 10, n_outliers))
y_outliers = np.append(y, np.random.uniform(-20, 40, n_outliers))

# Модели
ols = LinearRegression()
ols.fit(X_outliers.reshape(-1, 1), y_outliers)

ransac = RANSACRegressor(random_state=42, residual_threshold=2.0, max_trials=100)
ransac.fit(X_outliers.reshape(-1, 1), y_outliers)

# Разделение на inliers и outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Визуализация
plt.figure(figsize=(12, 6))
x_plot = np.linspace(0, 10, 100)

plt.scatter(X_outliers[inlier_mask], y_outliers[inlier_mask], 
           color='blue', alpha=0.6, s=50, label='Inliers')
plt.scatter(X_outliers[outlier_mask], y_outliers[outlier_mask], 
           color='red', alpha=0.6, s=50, marker='x', linewidths=2, label='Outliers')
plt.plot(x_plot, ols.predict(x_plot.reshape(-1, 1)), 'r--', linewidth=2, label='OLS')
plt.plot(x_plot, ransac.predict(x_plot.reshape(-1, 1)), 'g-', linewidth=2, label='RANSAC')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Сравнение OLS и RANSAC при наличии выбросов', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Количество inliers: {np.sum(inlier_mask)}")
print(f"Количество outliers: {np.sum(outlier_mask)}")

#%% md
# 
# Почему мы захотим применять тот или иной метод, зависит от наших требований (на скорость обучения, например) и толерантности к выбросам, которую мы можем себе позволить.
# 
# **Breakdown point** (точка разрушения) — минимальная доля выбросов, которая может сделать оценку сколь угодно плохой (то есть изменение параметров перестает быть ограниченным при добавлении следующих выбросов).
# 
# - OLS: breakdown point = 0% (один выброс может разрушить оценку)
# - Медианная регрессия: breakdown point = 50%
# - Theil-Sen: breakdown point ≈ 29.3%
# - RANSAC: зависит от параметров, обычно 30-50%
# - Huber: breakdown point зависит от параметра $\epsilon$, обычно 10-20%
# 
# 
# Рассмотрим синтетические данные. Пусть это будут синусоидальные данные, но некоторые наблюдения будут "испорчены"
#%%
X = np.random.uniform(0, 1.5, size=300)
y = np.sin(1.5 * np.pi * X) + np.random.normal(scale=0.1, size=X.shape)
X = X.reshape(-1,1)

X_test = np.random.uniform(0, 1.5, size=300) 
y_test = np.sin(1.5 * np.pi * X_test) + np.random.normal(scale=0.1, size=X_test.shape)
X_test = X_test.reshape(-1,1)

y_errors = y.copy()
y_errors[::5] = 3

X_errors = X.copy()
X_errors[::5] = 3

y_errors_large = y.copy()
y_errors_large[::5] = 10

X_errors_large = X.copy()
X_errors_large[::5] = 10
#%%
plt.scatter(X, y_errors)
#%% md
# Задание: Последовательно постройте различные модели для разных видов испорченности данных.
#%%
estimators = {
    "OLS": LinearRegression(),
    "Theil-Sen": TheilSenRegressor(random_state=42),
    "RANSAC": RANSACRegressor(random_state=42),
    "Huber": HuberRegressor(),
}

colors = {
    "OLS": "turquoise",
    "Theil-Sen": "gold",
    "RANSAC": "lightgreen",
    "Huber": "black",
}
linestyle = {"OLS": "-", "Theil-Sen": "-.", "RANSAC": "--", "Huber": "--"}
lw = 2
fig, ax = plt.subplots(nrows=3, ncols=2, figsize= (12,8))
fig.tight_layout()
x_plot = np.linspace(X.min(), X.max())
time_dict = {name: 0. for name in estimators.keys()}
errors_df = pd.DataFrame(columns = list(estimators.keys()))
time_dict["Problem"] = "Sum time"
for i, (title, this_X, this_y) in enumerate([
    ("Modeling Errors Only", X, y),
    ("Corrupt X, Small Deviants", X_errors, y),
    ("Corrupt y, Small Deviants", X, y_errors),
    ("Corrupt X, Large Deviants", X_errors_large, y),
    ("Corrupt y, Large Deviants", X, y_errors_large),
]):
    ax[i % 3, i // 3].plot(this_X[:, 0], this_y, "b+")
    errors_dict = {"Problem": title}
    for name, estimator in estimators.items():
        model = make_pipeline(PolynomialFeatures(3), estimator)
        t0 = time.time()
        model.fit(this_X, this_y)
        time_dict[name] += time.time() - t0
        mse = mean_squared_error(model.predict(X_test), y_test)
        errors_dict[name] = [mse]
        y_plot = model.predict(x_plot[:, np.newaxis])
        ax[i % 3, i // 3].plot(
            x_plot,
            y_plot,
            color=colors[name],
            linestyle=linestyle[name],
            linewidth=lw,
            label="%s: error = %.3f" % (name, mse),
        )
    errors_df = pd.concat([errors_df,pd.DataFrame.from_dict(data=errors_dict)])
    ax[i % 3, i // 3].title.set_text(title)
errors_df = pd.concat([errors_df, pd.DataFrame(time_dict, index=[0])]) 
plt.show()
#%%
colors = {
    "OLS": "turquoise",
    "Theil-Sen": "gold",
    "RANSAC": "lightgreen",
    "Huber": "black",
}
#%%
errors_df
#%% md
# **Задание**: Поиграйте с этими датасетами и параметрами моделей. 
# Ответьте на вопросы:
# - При каких размерах датасетов лучше использовать каждый из алгоритмов
# - При каких процентах выбросов и их значениях лучше использовать каждый из алгоритмов
# - Влияет ли то, как распределены выбросы, на выбор метода
# - Влияет ли отношение шума к сигналу на выбор алгоритма
# - Какие алгоритмы более чувствительны к параметрам, какие менее
# Для ответов покажите таблицы и/или графики
#%% md
# ## Гетероскедастичность
#%% md
# **Гетероскедастичность** (heteroscedasticity) — это ситуация, когда дисперсия ошибок регрессионной модели не является постоянной для всех наблюдений. 
# 
# В классической линейной регрессии предполагается **гомоскедастичность** (homoscedasticity):
# 
# $$\text{Var}(\epsilon_i | \mathbf{x}_i) = \sigma^2 = \text{const}$$
# 
# При гетероскедастичности дисперсия ошибок зависит от значений независимых переменных или предсказанных значений:
# 
# $$\text{Var}(\epsilon_i | \mathbf{x}_i) = \sigma_i^2 \neq \text{const}$$
# 
# 
# Гетероскедастичность не делает оценки коэффициентов OLS смещенными, но приводит к следующим проблемам:
# 
# 1. **Неэффективность оценок**: OLS-оценки перестают быть эффективными (не имеют минимальной дисперсии среди всех несмещенных оценок)
# 
# 2. **Некорректные стандартные ошибки**: Стандартные ошибки коэффициентов становятся смещенными, что приводит к:
#    - Некорректным доверительным интервалам
#    - Некорректным t-статистикам и p-значениям
#    - Неправильным выводам о статистической значимости
# 
# 3. **Несостоятельность оценок ковариационной матрицы**: Матрица ковариаций оценок коэффициентов оценивается неверно
# 
# 
# **Вопрос**: Мы наблюдаем гетероскедастичсность. Предложите примеры гетероскедастичности. В каких  случаях это нарушение возникает из-за нарушения других положений? 
#%% md
# Посмотрим на несколько типов гетероскедастичности. Какие можно придумать примеры для каждого из типов?
#%%
n = 200

# Тип 1: Дисперсия зависит от Х
X1 = np.random.uniform(0, 10, n)
sigma1 = 2.5 - np.clip(0.3 * X1, 0, 2.5)
y1 = 2 * X1 + 1 + np.random.normal(0, sigma1, n)

# Тип 2: Дисперсия зависит от квадрата X
X2 = np.random.uniform(0, 10, n)
sigma2 = 0.5 + 0.05 * X2**2
y2 = 2 * X2 + 1 + np.random.normal(0, sigma2, n)

# Тип 3: Дисперсия зависит от предсказанного значения Y
X3 = np.random.uniform(0, 10, n)
y_true3 = 2 * X3 + 1
sigma3 = 0.3 + 0.1 * np.abs(y_true3)
y3 = y_true3 + np.random.normal(0, sigma3, n)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (X, y, title) in enumerate([(X1, y1, 'Дисперсия зависит от X'),
                                     (X2, y2, 'Дисперсия зависит от X²'),
                                     (X3, y3, 'Дисперсия зависит от Y')]):
    axes[idx].scatter(X, y, alpha=0.6, s=30)
    axes[idx].set_xlabel('X', fontsize=12)
    axes[idx].set_ylabel('y', fontsize=12)
    axes[idx].set_title(title, fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% md
# В этой части рассмотрим данные о стоимости домовладения. При этом будем использовать следующие признаки:
# * Y = стоимость дома
# * X1 = площадь дома
# * X2 = площадь участка
# Подмножество набора данных об исследовании американского сообщества, полученного из Бюро переписей США.
#%%
data = pd.read_csv(f"{data_path}/realestate.txt", sep="\t") # pd.read_csv(f"{data_path}/realestate.csv").dropna()
#%%
# Делаем доп.датасет со всеми положениями
linear_X, linear_y = datasets.make_regression(n_samples=data.shape[0],
                                              n_features=data.shape[1],
                                              noise=75, random_state=46)

# Setting feature names to x1, x2, x3, etc. if they are not defined
linear_feature_names = ['X'+str(feature+1) for feature in range(linear_X.shape[1])]
#%%
data.columns
#%%
X_train, X_test, y_train, y_test = train_test_split(data[["SqFeet"]], np.log(data["SalePrice"]))
#%%

#X_train, X_test, y_train, y_test = train_test_split(data.drop(["Percent_Households_Below_Poverty_Level", "County"], axis=1), data["Percent_Households_Below_Poverty_Level"])
#%%
X_train
#%% md
# 
# Поскольку все переменные сильно асимметричны, мы сначала логарифмизируем каждую переменную. Затем, если провести регрессионный анализ и оценить график остатков в сравнении с предсказанными значениями, можно отметить несколько «коническую» форму остатков.
#%%
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
#%%
sns.scatterplot(x=X_test["SqFeet"], y=y_test, label='sale cost', color='blue')
sns.lineplot(x=X_test["SqFeet"], y=preds, label='Ordinary Linear Regression model', color='red', linewidth=2)
plt.xlabel('log(SqFeet)')
plt.ylabel('log(SalePrice)')
plt.legend()
plt.title(f'Use of weighted linear regression on data with heteroscedastic residuals')
plt.show()
#%%
print(f"The score of prediction is: {model.score(X_test, y_test)}")
#%% md
# Другой способ построить график остатков предлагает Sklearn
#%%
from sklearn.metrics import PredictionErrorDisplay
display = PredictionErrorDisplay(y_true=y_train, y_pred=model.predict(X_train))
display.plot()
plt.show()
#%%

#%% md
# Более объективную оценку могут дать специальные статистические тесты, например, тест Уайта или тест Брейша-Пагана
#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.tools as smt
import statsmodels.stats.diagnostic as smd
#%% md
# Тест Бреуша-Пагана  можно выполнить с помощью функции het_breuschpagan модуля statsmodels.stats.diagnostic для оценки того, объясняют ли независимые переменные дисперсию ошибок. 
# 
# Основные параметры функции het_breuschpagan resid - остатки модели, а exog_het — независимые переменные. Обратите внимание, что ему нужна также константная переменная. 
# Этот тест проверяет нулевую гипотезу о том, что все дисперсии ошибок равны, в сравнении с альтернативой, согласно которой дисперсии ошибок являются мультипликативной функцией одной или нескольких переменных. Этот тест затрагивает линейную форму гетероскедастичности.
# 
# Тест Уайта (het_white) оценивает то, объясняют ли независимые переменные линейной регрессии, квадратичные независимые переменные и перемножения независимых переменных дисперсию ошибок. Он не накладывает ограничений на форму гетероскедастичности.  Внутри происходит регрессия квадратов остатков по отношению к исходным переменным, их квадратам и попарным произведениям. Он более общий как раз из-за добавления дополнительных переменных. Например, квадраты позволяют найти гетероскедастичность в форме "песочных часов". Однако, у него есть минус в том, что исходная гипотеза разбивается на несколько.
# 
# **Вопрос**: почему это минус? Укажите теоретическую причину.
# 
# 
# Основные параметры в функции het_white resid - остатки, а exog - независимые переменные для объяснения дисперсии ошибок модели. 
# 
# Выполним тесты для нашего датасета, чтобы понять, реально ли в модели присутствует гетероскедастичность.
#%%
def get_residual_df(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = df_results['Actual'] - df_results['Predicted']
    
    return df_results

#%%
res_train = get_residual_df(model, X_train, y_train)
res_test = get_residual_df(model, X_test, y_test)
#%%
X_test_new = X_test.copy()
X_test_new["const"]=1.
#%%
bptest = smd.het_breuschpagan(resid=res_test["Residuals"], exog_het=X_test_new)
print("lm:", bptest[0], "lm_pvalue:", bptest[1])
#%% md
# **Вопрос**: что означает pvalue? Можем ли мы сказать, что остатки гетероскедачстичны?
#%%
wtest = smd.het_white(resid=res_test["Residuals"], exog=X_test_new)
print("lm:", wtest[0], "lm_pvalue:", wtest[1])
#%% md
# ### Решение проблемы
# **Вопрос**: Как можно избавиться от гетероскедастичности (или уменьшить ее влияние)? 
#%% md
# 
# ### Обобщенная линейная модель
# Линейная модель описывается уравнением $\textbf{Y}=\textbf{X}\beta+\epsilon^{*}$.
# 
# При этом $ \epsilon^{*} $ распределены нормально с нулевым средним и изменяющейся дисперсией:
# \begin{equation*} \left(\begin{array}{cccc} \sigma^{2}_{1} & 0 & \ldots & 0 \\ 0 & \sigma^{2}_{2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \ldots & \sigma^{2}_{n} \\ \end{array} \right) \end{equation*}
# Тогда мы можем определить веса $ w_i = 1/\sigma^{2}_{i} $ так чтобы
# \begin{equation*}\textbf{W}=\left( \begin{array}{cccc} w_{1} & 0 & \ldots & 0 \\ 0& w_{2} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0& 0 & \ldots & w_{n} \\ \end{array} \right) \end{equation*}
# 
# 
# Поскольку каждый вес обратно пропорционален дисперсии ошибки, он отражает информацию, содержащуюся в этом наблюдении. Итак, наблюдение с небольшой дисперсией ошибки имеет большой вес, поскольку оно содержит относительно больше информации, чем наблюдение с большой дисперсией ошибки (малый вес).
# Веса должны быть известны (или, чаще всего, оценены) с точностью до константы пропорциональности.
# 
# Тогда можно поставить задачу взвешенной регрессиии $\sum_{i=1}^n w_i^2(y_i - (\alpha 1 + \beta x_i))^2 = \sum_{i=1}^n (w_i y_i - (\alpha w_i + \beta w_i x_i))^2
# $
# 
# 
# Финальная модель называется White’s Heteroskedasticity Consistent Estimator https://www.hbs.edu/research-computing-services/Shared%20Documents/Training/hac.pdf
#%% md
# Так как остатки в нашей задаче зависят от величины признака X, мы можем предсказать их по этому признаку.
#%% md
# **Задание**: Постройте взвешенную модель, учитывающую гетероскедастичность. Для построения весов используйте аппроксимацию дисперсии ошибки.
#%%
add_x_train = model.predict(X_train)
add_x_test = model.predict(X_test)
# log is used to force all estimates to be positive
add_y_train = np.sqrt(np.abs(model.predict(X_train) - y_train))
add_y_test = np.sqrt(np.abs(model.predict(X_test) - y_test))
#%%
weight_regressor = LinearRegression().fit(X_train, add_y_train)
residual_estimates_train = weight_regressor.predict(X_train)
residual_estimates_test = weight_regressor.predict(X_test)
display = PredictionErrorDisplay(y_true=add_y_test, y_pred=residual_estimates_test)
display.plot()
plt.show()
#%%
weights_train = np.sqrt(1/(residual_estimates_train**2))
weights_test = np.sqrt(1/(residual_estimates_test**2))
#%%
weights_train = 1 / ((residual_estimates_train))
weights_test = 1 / ((residual_estimates_test))
#%%
updated_model = LinearRegression().fit(X_train, y_train, weights_train)
updated_preds = updated_model.predict(X_test)
#%%
display = PredictionErrorDisplay(
    y_true=y_test, y_pred = updated_preds
    )
display.plot()
plt.show()
#%%
print(f"The score of prediction is: {updated_model.score(X_test, y_test)}")
#%%
X_test_new = X_test.copy()
X_test_new["const"]=1.
#%%
bptest = smd.het_breuschpagan(resid=(y_test-updated_preds), exog_het=X_test_new)
print("lm:", bptest[0], "lm_pvalue:", bptest[1])
#%%
wtest = smd.het_white(resid=(y_test-updated_preds), exog=X_test_new)
print("lm:", wtest[0], "lm_pvalue:", wtest[1])
#%%
sns.scatterplot(x=X_test["SqFeet"], y=y_test, label='sale cost', color='blue')
sns.lineplot(x=X_test["SqFeet"], y=preds, label='Ordinary Linear Regression model', color='red', linewidth=2)
sns.lineplot(x=X_test["SqFeet"], y=updated_preds, label='Weighted Linear Regression', color='green', linewidth=2)
plt.xlabel('log(SqFeet)')
plt.ylabel('log(SalePrice)')
plt.legend()
plt.title(f'Use of weighted linear regression on data with heteroscedastic residuals')
plt.show()
#%%
from sklearn.metrics import r2_score

print(f"R^2: {r2_score(y_test, preds):.4f}")
print(f"MSE: {mean_squared_error(y_test, preds):.4f}")

#%% md
# ### Box-Cox
# При неизвестном типе распределения одним из лучших методов приведения распределения к нормальному является преобразование Бокса-Кокса. Суть метода заключается в преобразовании переменной с помощью параметра $\lambda$ так, что:
# 
# $ \hat{y_i} = { {y^\lambda}_i \over {\lambda}}$ - если $\lambda \neq 0 $
# 
# $ \hat{y_i} = log(y_i) $ - если $\lambda = 0 $
# 
# Параметр $\lambda$ подбирается с помощью максимизации логарифма правдоподобия.
#%%
def boxcox_transform(y, lmbda):
    if lmbda == 0:
        return np.log(y)
    else:
        return (y**lmbda - 1) / lmbda

def inverse_boxcox(y_transformed, lmbda):
    if lmbda == 0:
        return np.exp(y_transformed)
    else:
        return (lmbda * y_transformed + 1)**(1/lmbda)

#%%

from scipy.stats import boxcox

# делаем все значения строго положительными и находим лучшую лямбду
y_min_train = y_train.min() 
y_boxcox_pos = y_train - y_min_train + 1
y_transformed, fitted_lambda = boxcox(y_boxcox_pos)
print(f"fitted lambda: {fitted_lambda:.4f}")

#%%
box_cox_model = LinearRegression()
box_cox_model.fit(X_train, y_transformed)

# Получаем предсказания в преобразованном пространстве и обратно преобразуем
box_cox_preds = inverse_boxcox(box_cox_model.predict(X_test), fitted_lambda) + y_min_train - 1

sns.scatterplot(x=X_test["SqFeet"], y=y_test, label='sale cost', color='blue')
sns.lineplot(x=X_test["SqFeet"], y=preds, label='Ordinary Linear Regression model', color='red', linewidth=2)
sns.lineplot(x=X_test["SqFeet"], y=box_cox_preds, label='Box-Cox Linear Regression', color='green', linewidth=2)
plt.xlabel('log(SqFeet)')
plt.ylabel('log(SalePrice)')
plt.legend()
plt.title(f'Use of weighted linear regression on data with heteroscedastic residuals')
plt.show()
#%%
from sklearn.metrics import r2_score

print(f"R^2: {r2_score(y_test, box_cox_preds):.4f}")
print(f"MSE: {mean_squared_error(y_test, box_cox_preds):.4f}")

#%%
display = PredictionErrorDisplay(
    y_true = boxcox_transform(y_test - y_min_train + 1, fitted_lambda),
    y_pred = box_cox_model.predict(X_test)
    )
display.plot()
plt.show()
#%% md
# ### Квантильная регрессия
# 
# В нашем случае остатки линейной регрессии симметричны, поэтому мы можем легко перейти на квантильную регрессиию, которая позволяет оценить заданные квантили таргета. Этот регрессор минимизирует MAE, в отличие от стандартной регрессии.
# 
# Вопрос: Почему он минимизирует MAE?
# 
#%%
from sklearn.linear_model import QuantileRegressor
#%%
quantiles = [0.05, 0.5, 0.95]
predictions = {}
out_bounds_predictions_test = np.zeros_like(y_test, dtype=np.bool_)
for quantile in quantiles:
    qr = QuantileRegressor(quantile=quantile, alpha=0)
    y_pred = qr.fit(X_train, y_train).predict(X_test)
    predictions[quantile] = y_pred

    if quantile == min(quantiles):
        out_bounds_predictions_test = np.logical_or(
            out_bounds_predictions_test, y_pred >= y_test
        )
    elif quantile == max(quantiles):
        out_bounds_predictions_test = np.logical_or(
            out_bounds_predictions_test, y_pred <= y_test
        )
#%%
for quantile, y_pred in predictions.items():
    plt.plot(X_test, y_pred, label=f"Quantile: {quantile}")

plt.scatter(
    X_test[out_bounds_predictions_test],
    y_test[out_bounds_predictions_test],
    color="black",
    marker="+",
    alpha=0.5,
    label="Outside interval",
)
plt.scatter(
    X_test[~out_bounds_predictions_test],
    y_test[~out_bounds_predictions_test],
    color="black",
    alpha=0.5,
    label="Inside interval",
)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
_ = plt.title("Quantiles of heteroscedastic Normal distributed target")

#%% md
# Почитать: https://www3.nd.edu/~rwilliam/stats2/l25.pdf
#%% md
# **Задание**: Постройте модели, учитывающие гетероскедастичность  для  Anscombe 3 и 4. Предложите сами, как вы будете задавать веса.
#%% md
# **Задание**: Постройте синтетические данные, подходящие для линейной модели. Покажите, поможет ли использование QuantileRegressor для удаления выбросов получить лучшую модель и для каких видов гетероскедастичности (выбросы, распределение, "форма" шума)
#%% md
# ### Преобразования переменных
# 
# Иногда гетероскедастичность можно устранить простыми преобразованиями переменных.
# 
# **Логарифмическое преобразование**:
# 
# Часто используется, когда дисперсия пропорциональна квадрату среднего значения:
# 
# $$\text{Var}(Y | X) \propto [E(Y | X)]^2$$
# 
# После логарифмирования: $\text{Var}(\ln Y | X) \approx \text{const}$
# 
# **Квадратный корень**:
# 
# Полезно, когда дисперсия пропорциональна среднему:
# 
# $$\text{Var}(Y | X) \propto E(Y | X)$$
# 
# После преобразования: $\text{Var}(\sqrt{Y} | X) \approx \text{const}$
# 
#%% md
# ## Проверка нормальности остатков
# 
# Одним из важных предположений линейной регрессии является **нормальность распределения ошибок**. Это предположение необходимо для:
# - Корректного построения доверительных интервалов для коэффициентов
# - Проведения статистических тестов (t-тесты, F-тесты)
# - Получения эффективных оценок методом наименьших квадратов (теорема Гаусса-Маркова)
# - Качественной интерпретации результатов
# 
# Для проверки нормальности остатков используются:
# 1. **Визуальные методы**: Q-Q plots (квантиль-квантиль графики), гистограммы
# 2. **Статистические тесты**: Shapiro-Wilk (для малых выборок), Jarque-Bera (для больших выборок)
# 
# ### Визуальная проверка нормальности
# 
# Q-Q plot (quantile-quantile plot) сравнивает квантили остатков с квантилями нормального распределения. Если остатки нормально распределены, точки должны лежать примерно на прямой линии.
# 
#%%
from scipy import stats
def plot_normality_check(residuals, title="Проверка нормальности остатков"):
    residuals = np.array(residuals)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title('Q-Q Plot (Quantile-Quantile Plot)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Теоретические квантили', fontsize=12)
    axes[0].set_ylabel('Выборочные квантили', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Гистограмма остатков с наложенным нормальным распределением
    axes[1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black', label='Остатки')
    
    # Наложим нормальное распределение с теми же параметрами
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Нормальное распределение')
    axes[1].set_xlabel('Остатки', fontsize=12)
    axes[1].set_ylabel('Плотность', fontsize=12)
    axes[1].set_title('Гистограмма остатков', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

#%% md
# 
# **Статистические тесты:**
# - **Shapiro-Wilk test**: подходит для выборок размером до 5000 наблюдений. Нулевая гипотеза: остатки распределены нормально. Если p-value > 0.05, мы не можем отвергнуть гипотезу о нормальности.
# - **Jarque-Bera test**: подходит для больших выборок. Проверяет нормальность на основе асимметрии и эксцесса. Нулевая гипотеза: остатки распределены нормально.
# 
#%%
from scipy.stats import shapiro, jarque_bera

def test_normality(residuals, alpha=0.05, use_shapiro: bool = True):
    """ Используйте тест Шапиро-Уилка, если маленькая выборка"""
    residuals = np.array(residuals)
    
    if use_shapiro:
        shapiro_stat, shapiro_p = shapiro(residuals)
        results = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > alpha
        }
        print(f"Shapiro-Wilk test:")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  p-value: {shapiro_p:.4f}")
        print(f"  Нормальность: {'Да' if shapiro_p > alpha else 'Нет'}")
    else:
        # Тест Жарке-Бера (для больших выборок)
        jb_stat, jb_p = jarque_bera(residuals)
        results = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > alpha
        }
        print(f"\nJarque-Bera test:")
        print(f"  Statistic: {jb_stat:.4f}")
        print(f"  p-value: {jb_p:.4f}")
        print(f"  Нормальность: {'Да' if jb_p > alpha else 'Нет'}")
        
    return results

#%% md
# Проверим нормальность для модели, которая была построена ранее в разделе о гетероскедастичности.
# 
#%%
y_pred = model.predict(X)
residuals = y - y_pred

plot_normality_check(residuals, title="Normality check")

#%%
# Статистические тесты
print("\nСтатистические тесты:")
print("-" * 60)
test_results = test_normality(residuals)

# Дополнительная статистика
print(f"\nДополнительная статистика:")
print(f"  Среднее остатков: {np.mean(residuals):.6f}")
print(f"  Медиана остатков: {np.median(residuals):.6f}")
print(f"  Стандартное отклонение: {np.std(residuals):.6f}")
print(f"  Асимметрия (skewness): {stats.skew(residuals):.4f}")
print(f"  Эксцесс (kurtosis): {stats.kurtosis(residuals):.4f}")
#%% md
# ### Интерпретация результатов
# 
# **Q-Q plot:**
# - Если точки лежат близко к прямой линии, остатки распределены нормально
# - Отклонения от прямой указывают на отклонение от нормальности
# - S-образная кривая указывает на асимметрию распределения
# - Выпуклость/вогнутость указывает на тяжелые/легкие хвосты распределения
# 
# **Важно:** Нарушение нормальности остатков не всегда критично для предсказательной способности модели, но может влиять на статистические выводы (доверительные интервалы, p-значения).
# 
#%% md
# ## Мультиколлинеарность
#%% md
# Мультиколлинеарность представляет собой проблему, поскольку она дает менее надежные результаты регрессионной модели. Это связано с более широкими доверительными интервалами (большими стандартными ошибками), которые могут снизить статистическую значимость коэффициентов регрессии. При анализе данных это может привести к ложным впечатлениям.
#%% md
# Рассмотрим уже знакомый нам набор данных, посвященный недвижимости.
#%%
data = pd.read_csv(f"{data_path}/realestate.txt", sep="\t")
#%%
data["SalePrice"] = np.log(data["SalePrice"])
data.describe()
#%% md
# Обучим базовую модель на одном признаке.
#%%
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data[["SqFeet"]], data["SalePrice"], random_state=42)
model.fit(x_train, y_train)
print(f"Model score is: {model.score(x_test, y_test)}")
print(f"Coefficients are: {model.coef_}")
#%% md
# Теперь добавим несколько видов мультиколлинеарности в данных.
#%% md
# ### Совершенная коллинеарность
#%% md
# Если две или более независимых переменных имеют между собой линейную связь, то
# у нас получается идеальная мультиколлинеарность.
# Примеры: включение одной и той же информации дважды (стоимость в рублях и долларах), неправильное использование фиктивных переменных (dummy variables trap) и т. д.
# Вот пример идеальной мультиколлинеарности в модели с двумя объясняющими
# переменными:
# \begin{equation}
# y = α_{1} X_1 + \alpha_{2} X_2
# \end{equation}
# \begin{equation}
# X_1 = \beta_1 X_2 + \beta_2
# \end{equation}
# 
#%%
data["SqFeet_changed"] = data["SqFeet"]* 0.8 + 10
corrplot(data)
#%%
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data[["SqFeet_changed", "SqFeet"]], data["SalePrice"], random_state=42)
model.fit(x_train, y_train)
model.score(x_test, y_test)
print(f"Model score is: {model.score(x_test, y_test)}")
print(f"Coefficients are: {model.coef_}")
#%%
data["SqFeet_changed_2"] = data["SqFeet"]* 0.5 - 10
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data[["SqFeet_changed", "SqFeet_changed_2", "SqFeet"]], data["SalePrice"], random_state=42)
model.fit(x_train, y_train)
model.score(x_test, y_test)
print(f"Model score is: {model.score(x_test, y_test)}")
print(f"Coefficients are: {model.coef_}")
#%% md
# По идее, совершенная коллинеарность не дает в принципе построить линейную модель, так как невозможно оценить маржинальный эффект воздействия признака на целевую переменную, сохраняя остальные константными.
# )
# На практике же, даже если конкретная реализация OLS позволит построить коэффициенты, это ведет к тому, что модель перестает быть устойчивой.
#%% md
# ### Несовершенная мультиколлинеарность
# Несовершенная мультиколлинеарность не нарушает базовых предположений ЛР. Следовательно, теорема Гаусса-Маркова говорит нам, что оценки МНК состоятельны.
# 
# Так почему же нас волнует мультиколлинеарность?
# 2. Дисперсия и стандартные ошибки оценок коэффициентов регрессии будут
# увеличиваться. Это означает более низкую t-статистику.
# 3. На общую точность уравнения регрессии мультиколлинеарность практически не повлияет.
# 4. Коэффициенты регрессии будут сильно зависеть от признаков. Таким образом, коэффициенты существенно измениться при добавлении или удалении переменных.
#%%
data["SqFeet_changed"] = data["SqFeet"] * 0.8 - 20
data["SqFeet_changed_2"] = data["SqFeet_changed"] * 1.5 + 10

data["SqFeet_changed"] += np.random.normal(+1, 0.5, len(data["SqFeet_changed"]))
data["SqFeet_changed_2"] += np.random.normal(-1, 0.5, len(data["SqFeet_changed"]))

#%%
corrplot(data)
#%%
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data[["SqFeet_changed", "SqFeet_changed_2", "SqFeet"]], data["SalePrice"], random_state=42)
model.fit(x_train, y_train)
model.score(x_test, y_test)
print(f"Model score is: {model.score(x_test, y_test)}")
print(f"Coefficients are: {model.coef_}")
#%%
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data[["SqFeet_changed", "SqFeet"]], data["SalePrice"], random_state=42)
model.fit(x_train, y_train)
model.score(x_test, y_test)
print(f"Model score is: {model.score(x_test, y_test)}")
print(f"Coefficients are: {model.coef_}")
#%% md
# Как видно, только один признак имеет большой вес, точность при этом практически не меняется. Однако значимость многих признаков становится неудовлетворительной. Поэтому, если оценить значимость коэффициентов модели, можно обнаружить, что многие признаки просто не нужны. При этом получается, что даже при очень небольшом изменении модели мы можем сделать совершенно противоположные выводы о том, какие признаки важны.
# 
# Итак, первый способ найти коллинеарные признаки - оценить значимость коэффициентов (t-statistic + p-value)
#%% md
# ![image.png](data:image/png;base64,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x44dlg1EZ8+erayspM62YKxRNhZkV+4ItLa2rl+/vqICf567ra3t5ZdfduO/UvxSY88J0Q0LO281Zv8uevEVAImjjkARKMtrr71G32gdXdYVRfnpT3+6du3a999//5e//OWEf4Xt6ekpd99bejDmqFfF4WdIP7BC5/ko/1BVdfbs2ddddx09oMWx1u3atevqq69esWKFG8rS3t4uztMMX1dICQiUOQKqqra1tT3//POhUGjVqlUbNmzYuHGjm0ZUqnJjD4yBviP9yZH4WpVKOZBbZASGT1not5P3799fW1srSdLdd9/Nv6Ys6rhnz57f//73mqY9+eSTv/vd7xwHDzF+uV9v2LDhiSeeKN9SzJ49OxQKybLs+H2Q8Vmud999V5Kk+vp6sXZR/6oPP/zQUWdFUVatWjVnzhxLV0tdYSxJVFWVJOmtt96yhMMtIAAIAAKAwGgiMEzKQr0Up02bNoP9vOR3++2319bWiiMHLYyqqn/+85/r6ursj0aztKXOS1XVhx566PXXXy91RiWS/+mnnz766KOpVKqvr69cvFDpV04kSdq3bx+HJZPJNDY2SpJ09qztCzfkwyivvfbatGnTLKtFeI8e+e6EhcdwsXABCAACgAAgMIYIDJOy0M6df16LXtDdGY6kRFGUO+64I+9HQccQiKJkrarq7373u/LyvaWf2uHFX7t2Lb2WJGlggJ3axR+PvwtFURYsWGAhWJlM5s4777z11lvt5ENRlL17995www12vqJpWjwe//Wvfy2mUlV17969Ysj4wwA0AgQAAUBgUiAwfMpSEDxnz56VJOkf//hHQanKLnLZbZGl47H9yLVXXnmlqqqqLPDPZDL3339/RUWFyCo2btwoSdJrr71mKYKqqvv375ckydHvSlXVUCjk8Xi4KDrpcsMNN1RVVfFAi0y4BQQAAUAAEBgdBEaJsrzxxhtFP1R0dAAqKJdz58796le/cpxnKkjOqEVeu3btW2+99eabbz7yyCM8U/qFy3IZoRVFCYVCkiR98803tAjpdFqSpPnz59uLcPDgQUmSEokEdTGm/w8NDWUymdOnTy9ZskSSJL/fTy2oqurUqVMvXLhAP05kl8YRgwtAABAABACBUUBgNCgL9fBobGwso7F8eNBv3779wQcfLOtiHjhwQJKkTCYTjUYPHTo0PBxGOdVHH31Etwvt2rUrEolIkvToo4/a132GhoboFxOZ/5X+t6qq6pZbbvm///f/0o1voVCI6n/y5EnqSf3YY4/NmDGjrM06yhaB7AABQAAQKAUCpaUsW7ZsOX78+FdffXX99dfv3bu3FAUYVzIfeeSRzZs3jyuVcitDHVnoRMtbb71F+Qo/hM0+6ueWNoZPT5069fjjj8+YMWPt2rXd3d2OMyI7d+5cvXr1k08++ePnFB577LGGhob6+vqHHnpo4cKFQeFn+SKuoiizZ89+5plnxrB0kDUgAAgAAoCApmmlpSw/Og3U1NQ888wzjY2NjqPIRLKBG99bRVG4f+solL23t7ezszNbRtTzRpKkqqqq+vr6J5988sCBA+8Lv/KaV6AfOslW2GGHDw0NXX311W+//fawJUBCQAAQAAQAgaIgUFrK8tprr9XU1Lz55pvlNfgND1nKAHLMTNAI77//PpdPz5hvbm5uaGjglO7AgQMLFy5ctGhRKBRavnw5jzyMC1VV77777r/97W+OaV9//fXp06fTiRZJkhzjQGAsFrv66qu//vprgAIQAAQAAUBgbBEoLWUZ27KNcu6Dg4O/+MUvOPOw5/7ss882NDSI4QcOHNi+ffuMGTMkSeIrSqqqvvTSS5Ik1dXVjfzE1ffff1+SpJMnT4r50mufz/fmm29qmrZ161aLYvbIkzbkiSeemDdv3mTg3JPWxFBwQAAQKBcEgLIUzVLbt293PEaPZvDdd99Nmzbt8OHDlvwOHjy4bt26W2655Q9/+AMfFxVFefjhh/mtJUmht48//nhLS4s91U033dTR0XHy5ElJkrKdEmtPNalCVFWtqKhYs2bNpCo1FBYQAAQAgfGJAFCWotll5cqVkUgkm7jm5mbHXSerVq06cOAAPV2eHwn/3nvvtbe3ZxOVN9zCdQ4fPnzPPffYp3+2b99OPW3tj/JmMeEjeDyeqqqq3t7e22677fjx4xO+vFBAQAAQAATGPwJAWYpjI1VV77nnngMHDmQTV1dXFw6H7U9vuukmRVGOHz/+05/+dPbs2ZRtBAKBo0eP2iPnCFm3bt2SJUu++uqr++67j+745ZG/+OKLa665Jp1O8xC4yIvA//7v/9LN0h999FHeyBABEAAEAAFAYBQQAMpSHJAVRbnyyisvXrzoKE5RFI/HY584EU/LXbx4sSRJBw8eVFV1ypQplpkSR7E88Omnn960aVNVVZUkSZ2dnYqi3HjjjZw/qap67bXXHjt2jMeHi7wIqKoKk095UYIIgAAgAAiMJgJAWYqDdiqVuuaaa7INcoqi/PrXv965c6clszNnzkiSRNnJ7t27JUlqamq6cOHCTTfdJFIW8ahWyzWNRg9/u+6667jXhcfj4R+hVFX1tttuE/cN7Snw19PTY9EcbgEBQAAQAAQAgVFGAChLcQA/cODAzJkzRZ4hylUU5eqrr37nnXfEQE3TwuHwypUraaCqqtXV1bfccsu+ffuWLl0qxqSntFY5/bZu3Upj/vOf/5QkiS4nKYpy/fXX19XVccnTp08XPXDb29v5eXFuLoCyiOaAa0AAEAAEAIExQQAoS3Fgf/HFF3MckKqq6h//+Ef7qW7Tpk3jyzeaptGJlueeey4ej4tq0aNTsv1PY27atOm3v/0tneb58MMPRXcWVVV/+9vfbtu2jcu8VPiPp4ULQAAQAAQAAUBgTBAAylIc2JcuXZrjgNQf2cZf/vKXTZs2iZmpqnr77benUikeqCjKrFmzEEKXLl3igS4v5s2bV1VVRad5mpubb7jhBr5KNTQ09JOf/EQ8ws6lTIgGCAACgAAgAAiMHwSAshTBFqqq3nHHHY7HtXHpra2tf/nLX/htW1sbPUHuoYceOnHiBA+PRqP33ntvtgUmHs1yoarqjTfe+Itf/OKNN9544YUX5s2bx/mKpmkDAwM/ftk4mUxaUmW7pdM52Z5COCAACAACgAAgMCYIAGUpAuyXL1/+6U9/KrIEu9B33313ypQpPA499/bNN9/cvn27GPnHg0CGceLtV199RT1wn3nmmVdffZXnQiXH4/G5c+daAsVMxWuv11tXV3f33XfnZmBiErgGBAABQAAQAARGAQGgLEUA+fDhw7Is5+YEqqr6/f7du3cXIT+biPfff//mm292/LyRqqpz587ds2ePLZFDQCaTWbt27Y9nzPT19eXYAOWQEoIAAUAAEAAEAIESIwCUpQgAv/LKKw899FDe1Zy+vr45c+bkZjbD0ObZZ59ds2ZNVVXVwoUL7ck3bNhQW1vrPtOqqipN01RVlSQJjnKx4wkhgAAgAAgAAmOFAFCWIiAfCASee+65vIJUVd25cyfflpw3vssI77777pvkx8/75wkPHTpUEF/hCQ8cOCB+84iHwwUgAAgAAoAAIDBWCABlGSnyqqpWVlbu2rXLpaBhuKq4lGyP9u67754+fdoenjdEkiT3EzN5pUEEQAAQAAQAAUBg5AgAZRkphoqiXHPNNQMDAyMVNEbp6cn09FBduiR01VVXKYqyZ8+e7u7uMVIKsgUEAAFAABAABKwIAGWxImK///7771977TV7OA05efLkNddcMzQ0lC3CuA1XFKW2tpaefvvLX/5y9uzZn3zySU1NDT8PFyZaxq3tQDFAABAABCYhAkBZ8ht906ZNtbW12eJ1dnbOmDEjr+9ttuRjGL53794bb7zx7Nmz1NlW07QDBw6sXbs2EAg8/PDDwWCwHAs1hnhC1oAAIAAIAAIlRQAoSx54v/zyy5tvvnn69OnZxu+nn3561apVeaSMy8eLFi2iB/Lu3r37vvvuG5c6glKAACAACAACgICOAFCWPFXhjTfeoAsljqskqqrOmTPH/vGgPELHx2PuYytJkvgFovGhHWgBCAACgAAgAAiYEADKYoLDcqMoyvHjxyllOXPmjOWppmmKotx00039/f32R+M/ZMuWLbRoFRUV2eaQxn8pQENAABAABACBSYIAUJZcht68efPg4OCNN94oSZL4yWWe5t///vfPfvYzxwkYHgcuAAFAABAABAABQGDkCABlyYqhoihTp04dGBjw+XySJL3xxhv2qHv37vV6vTBFYUcGQgABQAAQAAQAgeIiAJQlK56qqj7yyCOqqq5YsUKSJMfzbdevX79y5cqsIjTt44n4y1FeeAQIAAKAACAACJQIAaAs+YF95ZVXJElaunSpfTZl0aJFjrMvXOicOXP4MScT4wLOl+PGhQtAABAABACB0UQAKEt+tHt7eyVJmjp1qsVnRVXVP/zhD7l9b49NxF9+yCAGIAAIAAKAACBQbASAsuRH9IsvvpAk6T//8z8zmYwY++LFi5IkWQLFCHANCAACgAAgAAgAAsVCAChLfiQVRbnuuuskSTp58qQY+9ChQ7IsW6ZexAgjv1bZzyKKfhjIEgi3gAAgAAgAAoDABEYAKEt+46qqOn36dEmS3nrrLTH25s2bly1bZndwEeOM8Lqurm7GjBlXXnmlmFFVVdWtt946Y8aMG264YTS/Cz3CskByQAAQAAQAAUBgJAgAZcmPnqqq9fX1kiSFQiFxTmX58uUvv/xy/vQjiKGq6rJly6qrqyVJ+vrrr7mkhQsXRiKRc+fO8RC4AAQAAUAAEAAEJjYCQFlc2Xf9+vWSJNXW1vI5FVVV77nnnoMHD7pKP4JI119//euvvy5J0saNG7mY1atXnzp1it/CBSAACAACgAAgMOERAMriysS7du2SJOnWW2/lsyyKolxzzTXffvutq/TDjaSq6s033/zDDz/cfvvtN910Uzqd1jRNVdWZM2dy8uRSNnd/oe4xLlNBNEAAEAAEAAFAYJwgAJTFlSE+/vhjeqoK3x908uTJ3//+95zBuJJSeKQPPvjg2muv1TRt3bp1kiRt3rxZ07RTp04tWbKkIGHBYHDt2rWLFy9+5JFH5s6de++997733nsFSYDIgAAgAAgAAoDA2CIAlMUV/oqi/OQnP5Ek6aOPPqIJtm/fPn/+/EKnOlxlJkQKBALNzc2app0+ffqqq66aMWOGqqpbtmwJh8NCrDyX1CHm/fffp6xL07S33nqrqqqq1Hwrj1rwGBAABAABQAAQKAQBoCyu0FJV9bbbbpMkafv27TTB008/HQqFXCUeQaSrr776m2++oQIeffTRHxen3n///YaGhsOHD3OpJ06ceD/7T9O03t7evXv3Njc3ezweyrHefPNNKooLgQtAABAABAABQGCcIwCUxa2BHnzwQUmSmpubqS9IdXX1zp073SYeVjxVVSVJ4hM5H374oSRJ9fX1Pp+PB2qaduzYsfey/2jOqqred999y5cvp7fNzc2SJH322WfD0gsSAQKAACAACAACY4AAUBa3oK9du1aSpJqaGurHOmXKlDNnzrhNPKx4H3744dSpU3lSVVXpRMsTTzzBA11efPvtt5Ik/e1vf6Peu9OnT582bZrIe1zKgWiAACAACAACgMBYIQCUxS3yO3bskCTp5ptvVhTl3Llz//Vf/1VSXxBVVV966SXqb8tVjMfjkiRt27aNh7i82LNnjyRJu3bt0jSNOrX09fW5TAvRAAFAABAABACB8YAAUBa3Vujv76fuq99///3evXstqzNupbiLd+DAgVtuuUWSpN/85jcLFizgiVRVnTdv3jDYxoYNG6699tqpU6fOnz9/+vTpx48f5zLhAhAABAABQAAQKAsEgLK4NZOiKFdeeaUkSYcPH37++eebmprcphxWPEVRMuQ38uUbVVXr6urC4bCiKL29vSWdHBpWWSERIAAIAAKAACCQHwGgLPkxojFUVZ02bdqPEy1//etfFy5c2NnZ6TblWMdT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)
#%%
from scipy.stats import t
#%% md
# **Задание**: Закончите функцию.
#%%
def get_p_values(model, x: pd.DataFrame, y: pd.DataFrame, a: int = 0.01) -> None:
  beta_hat = [model.intercept_] + model.coef_.tolist()
  n = len(x)
  print(f"Coeffficents: {beta_hat}")
  x_1 = np.column_stack((np.ones(n), x))
  # standard deviation of the noise.
  sigma_hat = np.sqrt(np.sum(np.square(y - x_1@beta_hat)) / (n - x_1.shape[1]))
  # estimate the covariance matrix for beta
  beta_cov = np.linalg.inv(x_1.T@x_1)
  # the t-test statistic for each variable from the formula from above figure
  t_vals = beta_hat / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
  # compute 2-sided p-values.
  p_vals = t.sf(np.abs(t_vals), n-x_1.shape[1])*2
  print(f"T values: {t_vals}")
  print(f"p-values: {p_vals}")
  print(f"Is valuable: {p_vals > 0.01}")
#%%
def get_p_values(model, x: pd.DataFrame, y: pd.DataFrame, a: int = 0.01) -> None:
  beta_hat = [model.intercept_] + model.coef_.tolist()
  n = len(x)
  print(f"Coeffficents: {beta_hat}")
  x_1 = np.column_stack((np.ones(n), x))
  # standard deviation of the noise.
  sigma_hat = # CAlculate the noise
  # estimate the covariance matrix for beta
  beta_cov = # Get (X^TX)^-1
  # the t-test statistic for each variable from the formula from above figure
  t_vals =  # your code: beta_hat / (sigma_hat * sqrt(diagonal(beta_cov)))
  # compute 2-sided p-values.
  p_vals = t.sf(np.abs(t_vals), n-x_1.shape[1])*2
  print(f"T values: {t_vals}")
  print(f"p-values: {p_vals}")
  print(f"Is valuable: {p_vals > 0.01}")
#%%
get_p_values(model, x_test, y_test)
#%% md
# Мультиколлинеарность можно обнаружить и с помощью коэффициента инфляции дисперсии (VIF). Для каждого признака в построенной модели этот коэффициент показывает, в какой степени коэфициент "испорчен" коллинеарностью с другими признаками.
# 
# В методе VIF мы выбираем каждый признак и сравниваем его со всеми другими признаками. Для каждого признака коэффициент рассчитывается как:
# 
# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAATCAYAAAAH3YpvAAACqklEQVRYCeWYS6hNYRTHfyjySugmAzIRebvqYoI8RgyYGRh4FBOEa0rKxO0OLjIwMVQouTcjI5GiO7hFHlMjAwMlSsnr+2vt2zrf+c527H3srXtW7fZ6fI+91v7Wf629obtpMnAIeADs7O5QlPP+KrCr6BLTgW3uWhYttMnZNDajhU6v+X2Zwe4bIrvfQ/yUaHydYqkA6sHlzGfgcMILBWYY6EnYFgM/gS0Jm1Q3gPuRbRHwCpga6esUSwdQD/8cOJXw4iwwO6GXai/woYVN6idAf8J+OqGrU9WRAOqkXIm8WGVBitTj4hBwb1xqZGYA38LpXWfqScB84/c0Dq1d6kgAtciIc0UOn3dyih0LJ/BkymBV7b2zqcotdfL/wgp+dBCOAivLPNQZ4IVb4ACwwMkxOwv4DqyJDSZfNKy7AAwAb1uM82oVF43Pu/70Uv16lfL7gC+2o1JNAcwjpWEe/j2KTmcKC/PWL2JbCxyp8GoousIqVdR5gEBeKZxHg8DdFgPU7nwNkLDe2Uulh1snj1XGaM9/dQmGrltG9Tp8//1McyyAx9rEglHgeAtvtofq+wlQp/831MkUVotVhBR837NudouoyM61Ey6YaaKPAdfONWmbFUrxH9EJ86OEefo8qoM2Brw9CLwssbmgaabN13o3jd9t9+XA7dT6l4FpKYPT6dvRg7xvopVC3iZ+hZtbJfu64GbqFJ65uWrHBEeCNpHiI+jy8GSmiXUrGkBB2CULhb6W1E1krZo6jxOh7RHcVYHptb6RogG8Bail2xGy5w7gm/6n9nX1BrhWq3cVbB4HUAUvhhfJCpand6HKZj9NhHUPvbGb+DiA7fi+GlD/mtF+9zNEf5+6gtRMK+30War7kja91jw1++ogssK3FXhsGDjhi0abcerssF9lZngpcq/KCwAAAABJRU5ErkJggg==)
# где R-квадрат — коэффициент детерминации в линейной регрессии этого признака, построенной по всем остальным признакам (1-to-all). Его значение лежит между 0 и 1.
# Как мы видим из формулы, чем больше значение R-квадрат, тем больше VIF. Следовательно, больший VIF означает большую корреляцию. Это согласуется с тем фактом, что более высокое значение R-квадрат означает более сильную коллинеарность. Обычно значение VIF выше 5 указывает на высокую мультиколлинеарность.
# 
#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = x_train.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x_train.values, i)
                          for i in range(len(x_train.columns))]

print(vif_data)
#%% md
# Как мы видим, наши переменные имеют очень высокие значения VIF, что означает сильную коллинеарность между ними.
# 
#%%
features = ['SqFeet', 'Beds', 'Baths', 'Air', 'Garage', 'Pool', 'Year',
       'Quality', 'Style', 'Lot', 'Highway']
#%%
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(data[features], data["SalePrice"], random_state=42)
model.fit(x_train, y_train)
model.score(x_test, y_test)
print(f"Model score is: {model.score(x_test, y_test)}")
print(f"Coefficients are: {model.coef_}")
#%%
vif_data = pd.DataFrame()
vif_data["feature"] = x_train.columns

# calculating VIF for each feature
vif_data["VIF"] = [ # Calculate vif for each feature in data]
vif_data
#%% md
# Теперь можно удалить некоторые признаки. Для этого используются разные способы feature selection. Посмотрим, узменится ли предсказание при удалении некоторых признаков
#%% md
# **Задание**: Постройте модель, удалив лишние признаки. Можно ли получить модель с лучшим скором?
#%%
# Your code