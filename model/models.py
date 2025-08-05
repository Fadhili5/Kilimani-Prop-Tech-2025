# coding: utf-8
"""
Title: Decoding heat hotspots in Kilimani, Nairobi
Author: Tulu, Serah, Fadhili
Date: 2025-08-04
Description: This script trains and compares multiple regression models to predict temperature (air and land surface)
             using satelite-derived data and local data. Each model undergoes hyperparameter tuning via
             grid search with 5-fold cross-validation. Evaluation metrics include MAE, MAPE, RMSE, R2, and
             Adjusted R2. The best-performing model is used to generate a regression plot.
"""

import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ========== Performance Metrics ==========
def metrics_define(y_true, y_pred):
    n_samples = len(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n_samples - 1)) / (n_samples - 1 - 1)
    return mae, mape, rmse, r2, adj_r2

# ========== Load Dataset ==========
data_path = r"data_filepath"  # Replace with your actual data file path

data = gpd.read_file(data_path)
X = data.iloc[:, "slice"].to_numpy() # Replace "slice" with the actual features
Y = data.iloc[:, "index"].to_numpy() # Replace "index" with the actual target variable

# ========== Dynamic Model Pool with Parameter Grids ==========
model_grid = {
    'CatBoost': (CatBoostRegressor(verbose=0), {'learning_rate': [0.03, 0.04], 'depth': [6, 8]}),
    'AdaBoost': (AdaBoostRegressor(random_state=0), {'learning_rate': [0.02, 0.05], 'n_estimators': [50, 100]}),
    'XGBoost': (XGBRegressor(random_state=2), {'learning_rate': [0.03, 0.05], 'n_estimators': [100, 150]}),
    'GBR': (GradientBoostingRegressor(random_state=3), {'learning_rate': [0.03, 0.05], 'n_estimators': [100, 150]}),
    'RFR': (RandomForestRegressor(random_state=15), {'n_estimators': [30, 50], 'max_depth': [4, 6]}),
    'ETR': (ExtraTreesRegressor(random_state=0), {'n_estimators': [30, 50]}),
    'SVR': (SVR(), {'C': [1, 10], 'gamma': [0.01, 0.1]}),
    'LR': (LinearRegression(), {}),
    'KNR': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
    'DTR': (DecisionTreeRegressor(), {'max_depth': [3, 5, 7]}),
    'HR': (HuberRegressor(), {'epsilon': [1.35, 1.75], 'alpha': [0.001, 0.01]}),
}

# ========== Grid Search with 5-Fold Cross-Validation ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(r2_score)
results_summary = []
best_model = None
best_score = -np.inf

for name, (model, param_grid) in model_grid.items():
    print(f"Running GridSearchCV for {name}...")
    grid = GridSearchCV(model, param_grid, cv=kf, scoring=scorer, n_jobs=-1)
    grid.fit(X, Y)
    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Best R2 Score: {grid.best_score_:.4f}\n")
    results_summary.append((name, grid.best_score_, grid.best_params_))
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_

# ========== Final Evaluation with Train-Test Split ==========
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=115)
pre_trn = best_model.predict(X_train)
pre_tst = best_model.predict(X_test)
train_metrics = metrics_define(Y_train, pre_trn)
test_metrics = metrics_define(Y_test, pre_tst)

# ========== Regression Plot ==========
def generate_regression_plot(Y_train, pre_trn, Y_test, pre_tst, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_train, pre_trn, c="#70CDBE", edgecolors="yellow", s=280, alpha=0.8, label="Training")
    plt.scatter(Y_test, pre_tst, c="#F5AA61", edgecolors="yellow", s=280, alpha=0.8, label="Testing")
    plt.plot([min(Y_train.min(), Y_test.min()), max(Y_train.max(), Y_test.max())],
             [min(Y_train.min(), Y_test.min()), max(Y_train.max(), Y_test.max())],
             linestyle='--', color='gray', linewidth=2, label='1:1 Line')
    linreg = LinearRegression().fit(Y_train.reshape(-1, 1), pre_trn)
    plt.plot(Y_train, linreg.predict(Y_train.reshape(-1, 1)), color='lightblue', linewidth=3, label='Fitting Line')
    plt.xlabel('True Value', fontsize=16)
    plt.ylabel('Predicted Value', fontsize=16)
    plt.legend(frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

# ========== Save Plot ==========
save_path = 'plot_filepath.png'

generate_regression_plot(Y_train, pre_trn, Y_test, pre_tst, save_path)

# ========== Print Summary ==========
print("\nModel Comparison Summary:")
for name, score, params in results_summary:
    print(f"{name}: R2 = {score:.4f}, Best Params = {params}")

print("\nBest Model Performance")
print(f"Best Model: {best_model.__class__.__name__}")
print("Train Metrics (MAE, MAPE, RMSE, R2, Adjusted R2):", train_metrics)
print("Test Metrics (MAE, MAPE, RMSE, R2, Adjusted R2):", test_metrics)

# Save the best model for future use
import joblib
model_save_path = 'best_model.pkl'
joblib.dump(best_model, model_save_path)
print(f"Best model saved to {model_save_path}")