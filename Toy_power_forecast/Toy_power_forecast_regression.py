# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# %%
# Load the provided data
df = pd.read_csv("daily_load_toy.csv", parse_dates=["date"], skipinitialspace=True)

# %% -------- Feature builders --------
# Build features using temp squared
def build_features_quadratic(df):
    X = pd.DataFrame({
        "intercept": 1.0,
        "temp_today": df["temp"],
        "temp_today_sq": df["temp"]**2,
        "temp_tom_fc": df["temp_tomorrow_fc"],
        "temp_tom_fc_sq": df["temp_tomorrow_fc"]**2,
        "is_weekend": df["is_weekend"].astype(int),
        "load_yday": df["load_yesterday"],
    })
    y = df["load_tomorrow"].values
    return X, y

# %% Build features using cooling and heating degree days instead
def build_features_dd(df, T_cool=18.0, T_heat=12.0):
    T_today = df["temp"].values
    T_tom_fc = df["temp_tomorrow_fc"].values
    CDD_today = np.maximum(T_today - T_cool, 0.0)
    HDD_today = np.maximum(T_heat - T_today, 0.0)
    CDD_tom = np.maximum(T_tom_fc - T_cool, 0.0)
    HDD_tom = np.maximum(T_heat - T_tom_fc, 0.0)
    X = pd.DataFrame({
        "intercept": 1.0,
        "CDD_today": CDD_today,
        "HDD_today": HDD_today,
        "CDD_tom_fc": CDD_tom,
        "HDD_tom_fc": HDD_tom,
        "is_weekend": df["is_weekend"].astype(int),
        "load_yday": df["load_yesterday"],
    })
    y = df["load_tomorrow"].values
    return X, y

# %%
def ols_fit_predict(X_train, y_train, X_test):
    beta, *_ = np.linalg.lstsq(X_train.values, y_train, rcond=None)
    yhat = X_test.values @ beta
    return beta, yhat

# %%
def metrics(y_true, yhat):
    mae = float(np.mean(np.abs(y_true - yhat)))
    rmse = float(np.sqrt(np.mean((y_true - yhat)**2)))
    ss_res = float(np.sum((y_true - yhat)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res/ss_tot
    mape = float(np.mean(np.abs((y_true - yhat)/y_true)) * 100.0)
    return {"MAE (MW)": mae, "RMSE (MW)": rmse, "R^2": r2, "MAPE (%)": mape}

# %% -------- Train/test split (last 30 days as test) --------
test_days = 30
train_df = df.iloc[:-test_days].copy()
test_df = df.iloc[-test_days:].copy()

XA, yA = build_features_quadratic(train_df)
XA_test, yA_test = build_features_quadratic(test_df)

XB, yB = build_features_dd(train_df)
XB_test, yB_test = build_features_dd(test_df)

# %% -------- Fit & evaluate (OLS) --------
beta_A, yhat_A = ols_fit_predict(XA, yA, XA_test)
beta_B, yhat_B = ols_fit_predict(XB, yB, XB_test)

results = pd.DataFrame({
    "Quadratic_T (OLS)": pd.Series(metrics(yA_test, yhat_A)),
    "HDD_CDD (OLS)": pd.Series(metrics(yB_test, yhat_B)),
})
print("===== Test Metrics (OLS) =====")
print(results.round(3))

coef_A = pd.Series(beta_A, index=XA.columns, name="Quadratic_T (OLS)")
coef_B = pd.Series(beta_B, index=XB.columns, name="HDD_CDD (OLS)")
coefs = pd.concat([coef_A, coef_B], axis=1)
print("\n===== Coefficients (OLS) =====")
print(coefs.round(2))

# Save coefficients to CSV
coefs.to_csv("ols_coefficients.csv", index=True)

# Save metrics to CSV
results.to_csv("ols_metrics.csv", index=True)

# %% -------- Plots (OLS) --------
# (Cast to ndarray for Pylance peace of mind)
yA_plot = np.asarray(yA_test, dtype=float)
yAhat_plot = np.asarray(yhat_A, dtype=float)
yB_plot = np.asarray(yB_test, dtype=float)
yBhat_plot = np.asarray(yhat_B, dtype=float)

plt.figure(figsize=(10, 4))
plt.plot(yA_plot, label="Actual")
plt.plot(yAhat_plot, label="Pred: Quadratic T (OLS)")
plt.title("Actual vs Predicted Daily Load — Quadratic T (OLS) — last 30 days")
plt.xlabel("Day")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 4))
plt.plot(yB_plot, label="Actual")
plt.plot(yBhat_plot, label="Pred: HDD/CDD (OLS)")
plt.title("Actual vs Predicted Daily Load — HDD/CDD (OLS) — last 30 days")
plt.xlabel("Day")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.show()

# %% ===================== Ridge & Lasso (Cross validation) =====================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline

# Time-series CV and alpha grids
tscv = TimeSeriesSplit(n_splits=5)
ridge_alphas = np.logspace(-2, 3, 20)
lasso_alphas = np.logspace(-3, 1, 30)

def ridge_pipeline():
    # Drop explicit intercept column and let model fit intercept
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=ridge_alphas, cv=tscv, fit_intercept=True))
    ])

def lasso_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LassoCV(alphas=lasso_alphas, cv=tscv, max_iter=20000, fit_intercept=True))
    ])

def fit_regularized(name, X_train_df, y_train, X_test_df):
    # Drop the explicit intercept for sklearn models
    Xtr = X_train_df.drop(columns=["intercept"]).to_numpy(dtype=float)
    Xte = X_test_df.drop(columns=["intercept"]).to_numpy(dtype=float)
    ytr = np.asarray(y_train, dtype=float)

    # Ridge and Lasso pipelines
    rp = ridge_pipeline(); rp.fit(Xtr, ytr)
    lp = lasso_pipeline(); lp.fit(Xtr, ytr)

    return {
        "ridge_pipe": rp,
        "lasso_pipe": lp,
        "yhat_ridge": rp.predict(Xte),
        "yhat_lasso": lp.predict(Xte),
        "ridge_alpha": rp.named_steps["model"].alpha_,
        "lasso_alpha": lp.named_steps["model"].alpha_,
        "ridge_coef": rp.named_steps["model"].coef_,
        "lasso_coef": lp.named_steps["model"].coef_,
        "feature_names": X_train_df.drop(columns=["intercept"]).columns.tolist(),
    }


# %% -------- Fit & evaluate (Ridge/Lasso) : Quadratic T --------
fitA = fit_regularized("Quadratic_T", XA, yA, XA_test)
resA_ridge = pd.Series(metrics(yA_test, fitA["yhat_ridge"]), name="Quadratic_T (Ridge)")
resA_lasso = pd.Series(metrics(yA_test, fitA["yhat_lasso"]), name="Quadratic_T (Lasso)")
resA = pd.concat([resA_ridge, resA_lasso], axis=1)
print("\n===== Test Metrics (Ridge/Lasso) — Quadratic T =====")
print(resA.round(3))
print(f"Ridge alpha (Quadratic T): {fitA['ridge_alpha']:.4g}")
print(f"Lasso alpha (Quadratic T): {fitA['lasso_alpha']:.4g}")

# Coefficients (on scaled space; still useful for relative importance)
coefA = pd.DataFrame({
    "Quadratic_T (Ridge)": pd.Series(fitA["ridge_coef"], index=fitA["feature_names"]),
    "Quadratic_T (Lasso)": pd.Series(fitA["lasso_coef"], index=fitA["feature_names"]),
})
print("\n===== Coefficients (Ridge/Lasso) — Quadratic T =====")
print(coefA.round(3))

# Plots
plt.figure(figsize=(10, 4))
plt.plot(yA_plot, label="Actual")
plt.plot(fitA["yhat_ridge"], label="Ridge")
plt.plot(fitA["yhat_lasso"], label="Lasso")
plt.title("Actual vs Predicted — Quadratic T (Ridge & Lasso) — last 30 days")
plt.xlabel("Day"); plt.ylabel("MW"); plt.legend(); plt.tight_layout(); plt.show()

# %% -------- Fit & evaluate (Ridge/Lasso) : HDD/CDD --------
fitB = fit_regularized("HDD_CDD", XB, yB, XB_test)
resB_ridge = pd.Series(metrics(yB_test, fitB["yhat_ridge"]), name="HDD_CDD (Ridge)")
resB_lasso = pd.Series(metrics(yB_test, fitB["yhat_lasso"]), name="HDD_CDD (Lasso)")
resB = pd.concat([resB_ridge, resB_lasso], axis=1)
print("\n===== Test Metrics (Ridge/Lasso) — HDD/CDD =====")
print(resB.round(3))
print(f"Ridge alpha (HDD/CDD): {fitB['ridge_alpha']:.4g}")
print(f"Lasso alpha (HDD/CDD): {fitB['lasso_alpha']:.4g}")

coefB = pd.DataFrame({
    "HDD_CDD (Ridge)": pd.Series(fitB["ridge_coef"], index=fitB["feature_names"]),
    "HDD_CDD (Lasso)": pd.Series(fitB["lasso_coef"], index=fitB["feature_names"]),
})
print("\n===== Coefficients (Ridge/Lasso) — HDD/CDD =====")
print(coefB.round(3))

plt.figure(figsize=(10, 4))
plt.plot(yB_plot, label="Actual")
plt.plot(fitB["yhat_ridge"], label="Ridge")
plt.plot(fitB["yhat_lasso"], label="Lasso")
plt.title("Actual vs Predicted — HDD/CDD (Ridge & Lasso) — last 30 days")
plt.xlabel("Day"); plt.ylabel("MW"); plt.legend(); plt.tight_layout(); plt.show()

# %% -------- Save results to CSV --------
# Save metrics to CSV
resA.to_csv("ridge_lasso_metrics_quadratic.csv", index=True)
resB.to_csv("ridge_lasso_metrics_hdd_cdd.csv", index=True)

# Save coefficients to CSV
coefA.to_csv("ridge_lasso_coefficients_quadratic.csv", index=True)
coefB.to_csv("ridge_lasso_coefficients_hdd_cdd.csv", index=True)
