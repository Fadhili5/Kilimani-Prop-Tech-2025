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