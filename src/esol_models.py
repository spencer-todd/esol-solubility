# -*- coding: utf-8 -*-
"""esol_models.py: The model pipelines and their parameter ranges, shared by notebooks 02 to 05.

Each builder function returns a new, untrained scikit-learn pipeline, as make_knn_pipeline() and
make_rf_pipeline() do in Tutorial 4. Keeping them in one file means every notebook trains exactly
the same models.
"""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from esol_data import RANDOM_SEED, FEATURES_9


# Source: Tutorial 4 (7.1 Pipeline with StandardScaler)
# Changes (final project): only the descriptors are standardized. The 0/1 fingerprint bits are kept as they are
def make_preprocessing(feature_set):
    """Standardize the descriptors only. The 0/1 fingerprint bits are kept as they are."""
    if feature_set in ['8 descriptors', '9 descriptors']:
        return StandardScaler()
    if feature_set == 'fingerprints':
        return 'passthrough'   # no change
    # descriptors + fingerprints: scale the 9 descriptor columns, pass the bits through
    return ColumnTransformer([('scaling', StandardScaler(), FEATURES_9)], remainder='passthrough')


# Baselines of 02_baselines.ipynb
# Source: Tutorial 4 (4.1 LinearRegression, 7.1 Pipeline with StandardScaler, 11.1 kNN). Same models as the proposal
def make_linear_pipeline():
    """Pipeline: scaling, then linear regression (8 descriptors)."""
    return Pipeline([
        ('scaling', StandardScaler()),
        ('linear', LinearRegression()),
    ])


def make_ridge_pipeline(feature_set='8 descriptors'):   # default: the 8-descriptor baseline of notebook 02
    """Pipeline: preprocessing, then ridge regression."""
    # Changes (final project): takes the feature set, so ridge can also use the fingerprints (notebooks 03, 04, 05)
    return Pipeline([
        ('preprocessing', make_preprocessing(feature_set)),
        ('ridge', Ridge()),
    ])


def make_knn_pipeline():
    """Pipeline: scaling, then k-nearest neighbours (kNN) with the Euclidean distance (8 descriptors)."""
    return Pipeline([
        ('scaling', StandardScaler()),
        ('knn', KNeighborsRegressor(metric='euclidean')),
    ])


# Source: Tanimoto similarity from Tutorial 3 (4. Molecular Fingerprints & Similarity)
# Changes (final project): the Jaccard distance (= 1 - Tanimoto similarity) on the 0/1 fingerprint bits
def make_knn_fingerprint_pipeline():
    """Pipeline: kNN with the Jaccard (1 - Tanimoto) distance on the fingerprint bits."""
    return Pipeline([
        ('knn', KNeighborsRegressor(metric='jaccard')),
    ])


# The three models of our proposal (03_models.ipynb)
# Source: builder functions like make_knn_pipeline() / make_rf_pipeline() in Tutorial 4 (11., 12.)
def make_krr_pipeline(feature_set):
    """Pipeline: preprocessing, then kernel ridge regression (RBF kernel) on standardized log S."""
    # KernelRidge has no intercept, so log S is standardized (mean 0) before the fit
    return Pipeline([
        ('preprocessing', make_preprocessing(feature_set)),
        ('krr', TransformedTargetRegressor(regressor=KernelRidge(kernel='rbf'), transformer=StandardScaler())),
    ])


def make_rf_pipeline(feature_set):
    """Pipeline: preprocessing, then a random forest with 300 trees."""
    return Pipeline([
        ('preprocessing', make_preprocessing(feature_set)),
        # n_jobs=1: the search already uses all CPU cores
        ('forest', RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1)),
    ])


def make_xgb_pipeline(feature_set):
    """Pipeline: preprocessing, then gradient-boosted trees (XGBoost, as in Tutorial 5)."""
    return Pipeline([
        ('preprocessing', make_preprocessing(feature_set)),
        # n_jobs=1: the search already uses all CPU cores. 'hist' is the default tree method
        ('xgb', XGBRegressor(random_state=RANDOM_SEED, n_jobs=1, tree_method='hist')),
    ])


# Parameter ranges. Ridge and kNN are searched on the full grid (GridSearchCV), the three proposed
# models with 20 random settings (RandomizedSearchCV). The reasons for each range are in notebook 03 (section 2)
# Source: Tutorial 4 (9.1 GridSearchCV), the grids of the proposal for ridge and kNN
ridge_param_grid = {'ridge__alpha': np.logspace(-3, 3, 13)}
knn_param_grid = {
    'knn__n_neighbors': [3, 5, 7, 10, 15, 20, 30],
    'knn__weights': ['uniform', 'distance'],
}
krr_param_grid = {
    'krr__regressor__alpha': np.logspace(-3, 1, 9),
    'krr__regressor__gamma': np.logspace(-5, 1, 13),
}
rf_param_grid = {
    'forest__max_features': [0.1, 0.2, 0.33, 0.5, 0.75, 1.0],
    'forest__min_samples_leaf': [1, 2, 3, 5],
}
xgb_param_grid = {
    'xgb__n_estimators': [200, 400, 800, 1200],
    'xgb__max_depth': [3, 4, 6, 8],
    'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.85, 1.0],
    'xgb__colsample_bytree': [0.3, 0.5, 0.8, 1.0],
    'xgb__min_child_weight': [1, 3, 5, 10],
}
