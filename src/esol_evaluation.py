# -*- coding: utf-8 -*-
"""esol_evaluation.py: Regression metrics, the parity plot and the figure style (sizes and model colours),
shared by all notebooks."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score

from esol_data import TARGET_LABEL


# Source: Tutorial 4 (3.1, get_regression_metrics), as in the proposal
# Changes (proposal): RMSE and R2 instead of MSE and MAPE (MAPE is not meaningful because
#                     log S crosses zero). The function regression_metrics() also works for predictions
#                     that do not come from a model (e.g. the ESOL equation)
def get_regression_metrics(model, X, y_true):
    """
    Get a dictionary with regression metrics:
    model: sklearn model with predict method
    X: feature matrix
    y_true: ground truth values
    """
    y_predicted = model.predict(X)
    return regression_metrics(y_true, y_predicted)


def regression_metrics(y_true, y_predicted):
    """Regression metrics from true and predicted values."""
    metrics_dict = {
        'mae': mean_absolute_error(y_true, y_predicted),
        'rmse': np.sqrt(mean_squared_error(y_true, y_predicted)),
        'max_error': max_error(y_true, y_predicted),
        'r2': r2_score(y_true, y_predicted),
    }
    return metrics_dict


# Source: parity plot idea from Tutorial 4 (6. Evaluate the model performance, hist_density)
# Changes (proposal): a scatter plot instead of a 2D histogram, because a random test set only has 223 molecules
def parity_plot(ax, y_true, y_pred, title, lims, color=None):
    """Parity plot (predicted vs. true values) with the y = x line. color: the model's colour (MODEL_COLOURS)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.scatter(y_true, y_pred, s=15, alpha=0.7, color=color)
    ax.plot(lims, lims, 'k--', lw=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(TARGET_LABEL)
    ax.set_ylabel('Predicted log S (log mol/L)')
    ax.set_title(f"{title} (RMSE = {rmse:.2f})")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


# Source: plotting style from Tutorial 3 (6. Data Visualization)
# Changes (final project): figures are drawn at the size they are printed in the paper
# (full width 6.5 in, half width 3.2 in) with 9 pt text, and each model always has the same
# colour (Okabe-Ito colour-blind safe palette, plus greys for the baselines)
FULL_WIDTH = 6.5   # inches
HALF_WIDTH = 3.2   # inches

MODEL_COLOURS = {
    'ESOL equation (Delaney, 2004)': '#999999',
    'ESOL refit (Crippen logP)': '#7f7f7f',
    'Ridge (8 descriptors)': '#56B4E9',
    'Ridge (9 descriptors)': '#56B4E9',
    'Ridge (descriptors + fingerprints)': '#0072B2',   # darker blue than ridge on the descriptors only
    'kNN (8 descriptors)': '#000000',
    'kNN (fingerprints)': '#555555',
    'Kernel ridge': '#E69F00',
    'Random forest': '#009E73',
    'XGBoost': '#D55E00',
    'Random forest (8 descriptors)': '#009E73',
    'Random forest (9 descriptors)': '#009E73',
}


def set_paper_style():
    """Matplotlib settings for the paper figures (9 pt text, no top/right frame)."""
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    })
