#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR      = "."
HELD_OUT_FILE = os.path.join(WORK_DIR, "models/held_out_data.pkl")
RF_DIR        = os.path.join(WORK_DIR, "models/models_rf")
XGB_DIR       = os.path.join(WORK_DIR, "models/models_xgb")
OUTPUT_DIR    = os.path.join(WORK_DIR, "outputs")

PARAMS_TO_TRAIN = ['i', 't2_t1', 'q', 'p1', 'p2']
N_FOLDS = 5

PARAM_LABELS = {
    'i':     'Inclination i (\u00b0)',
    't2_t1': 'Temperature Ratio T\u2082/T\u2081',
    'q':     'Mass Ratio q',
    'p1':    'Surface Potential \u03a9\u2081',
    'p2':    'Surface Potential \u03a9\u2082'
}

# Uniform 10% relative error threshold for all parameters (reviewer m5)
ERROR_THRESHOLD = 0.10

# =============================================================================
# HELPERS
# =============================================================================

def ensemble_predict(model_dir, model_type, X_held_out):
    """
    Load all fold models, apply each fold's scaler, average predictions.
    Each scaler was fit only on that fold's training data — no leakage.
    """
    all_preds = {param: [] for param in PARAMS_TO_TRAIN}

    for fold_idx in range(N_FOLDS):
        fold_file = os.path.join(model_dir, f"{model_type}_fold_{fold_idx}.pkl")
        with open(fold_file, 'rb') as f:
            fold_data = pickle.load(f)

        scaler = fold_data['scaler']
        models = fold_data['models']

        X_scaled = scaler.transform(X_held_out)

        for param in PARAMS_TO_TRAIN:
            pred = models[param].predict(X_scaled)
            all_preds[param].append(pred)

    # Ensemble: mean across folds
    return {param: np.mean(all_preds[param], axis=0) for param in PARAMS_TO_TRAIN}


def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


MORPH_STYLE = {
    'detached':     {'color': 'gold',       'marker': 'x', 'label': 'Detached'},
    'contact':      {'color': 'tab:blue',   'marker': 'o', 'label': 'Contact'},
    'semidetached': {'color': 'tab:green',  'marker': 's', 'label': 'Semidetached'},
}


def make_scatter_plots(preds, params_held, model_name, out_file):
    """Generate prediction vs true scatter plots with morphology colours."""
    morphology = params_held['morphology'].values

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(PARAMS_TO_TRAIN):
        ax = axes[idx]
        y_true = params_held[param].values
        y_pred = preds[param]

        r2, mae, rmse = compute_metrics(y_true, y_pred)

        # Plot each morphology class separately for legend
        for morph, style in MORPH_STYLE.items():
            mask = morphology == morph
            if mask.sum() == 0:
                continue
            ax.scatter(y_true[mask], y_pred[mask],
                       alpha=0.6, s=20,
                       color=style['color'],
                       marker=style['marker'],
                       label=style['label'],
                       zorder=3)

        lim_min = min(y_true.min(), y_pred.min())
        lim_max = max(y_true.max(), y_pred.max())

        # 1:1 line
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'k-', linewidth=1.5, label='1:1')

        # Uniform ±10% relative error lines (reviewer m5)
        x_line = np.linspace(lim_min, lim_max, 200)
        ax.plot(x_line, x_line * (1 + ERROR_THRESHOLD),
                'r--', linewidth=1, alpha=0.7, label='\u00b110%')
        ax.plot(x_line, x_line * (1 - ERROR_THRESHOLD),
                'r--', linewidth=1, alpha=0.7)

        ax.set_xlabel(f'True {PARAM_LABELS.get(param, param)}')
        ax.set_ylabel(f'Predicted {PARAM_LABELS.get(param, param)}')
        ax.set_title(f'{param.upper()} — R\u00b2={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.suptitle(f'{model_name} Predictions vs True Values\n'
                 f'Held-Out Test Set ({params_held.shape[0]} systems)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


def make_residual_plots(preds, params_held, model_name, out_file):
    """
    Residual distributions (predicted - true) with Gaussian fits.
    Matches style of original Fig 5 in the manuscript but uses held-out data.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, param in enumerate(PARAMS_TO_TRAIN):
        ax = axes[idx]
        y_true = params_held[param].values
        y_pred = preds[param]
        residuals = y_pred - y_true

        mu, sigma = residuals.mean(), residuals.std()

        # Histogram
        ax.hist(residuals, bins=20, density=True,
                color='steelblue', alpha=0.6, edgecolor='white')

        # Gaussian fit overlay
        x_fit = np.linspace(residuals.min(), residuals.max(), 300)
        ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sigma),
                'r-', linewidth=2)

        # Zero line
        ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Residual')
        ax.set_ylabel('Density')
        label = PARAM_LABELS.get(param, param)
        ax.set_title(f'{label}\n(\u03bc={mu:.3f}, \u03c3={sigma:.3f})')
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    plt.suptitle(f'{model_name} Residual Distributions\n'
                 f'Held-Out Test Set ({params_held.shape[0]} systems)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SCRIPT 5g: HELD-OUT TEST SET EVALUATION")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load held-out data (features + true labels)
    print(f"\nLoading held-out data: {HELD_OUT_FILE}")
    with open(HELD_OUT_FILE, 'rb') as f:
        held_out_data = pickle.load(f)

    X_held     = held_out_data['held_out_features']
    params_held = held_out_data['held_out_params']

    print(f"Held-out samples: {len(X_held)}")
    print(f"Morphology distribution:")
    print(params_held['morphology'].value_counts())

    # Ensemble predictions
    print("\nEnsembling RF predictions across 5 folds ...")
    rf_preds = ensemble_predict(RF_DIR, 'rf', X_held)

    print("Ensembling XGB predictions across 5 folds ...")
    xgb_preds = ensemble_predict(XGB_DIR, 'xgb', X_held)

    # ---- Metrics table -------------------------------------------------------
    print("\n" + "=" * 80)
    print("HELD-OUT TEST RESULTS")
    print("=" * 80)
    rf_r2_label  = 'RF R\u00b2'
    xgb_r2_label = 'XGB R\u00b2'
    header = (f"\n{'Parameter':<12} | {rf_r2_label:>8} | {'RF MAE':>8} | {'RF RMSE':>9} |"
              f" {xgb_r2_label:>8} | {'XGB MAE':>8} | {'XGB RMSE':>9}")
    print(header)
    print("-" * 82)

    table_rows = []
    for param in PARAMS_TO_TRAIN:
        y_true = params_held[param].values

        rf_r2,  rf_mae,  rf_rmse  = compute_metrics(y_true, rf_preds[param])
        xgb_r2, xgb_mae, xgb_rmse = compute_metrics(y_true, xgb_preds[param])

        print(f"{param:<12} | {rf_r2:>8.4f} | {rf_mae:>8.4f} | {rf_rmse:>9.4f} |"
              f" {xgb_r2:>8.4f} | {xgb_mae:>8.4f} | {xgb_rmse:>9.4f}")

        table_rows.append({
            'parameter': param,
            'rf_r2':  rf_r2,  'rf_mae':  rf_mae,  'rf_rmse':  rf_rmse,
            'xgb_r2': xgb_r2, 'xgb_mae': xgb_mae, 'xgb_rmse': xgb_rmse
        })

    results_df = pd.DataFrame(table_rows)
    csv_file = os.path.join(OUTPUT_DIR, 'held_out_results.csv')
    results_df.to_csv(csv_file, index=False)
    print(f"\nSaved: {csv_file}")

    # ---- Scatter plots -------------------------------------------------------
    make_scatter_plots(
        rf_preds, params_held,
        model_name='Random Forest',
        out_file=os.path.join(OUTPUT_DIR, 'fig4_rf_held_out_scatter.png')
    )

    make_scatter_plots(
        xgb_preds, params_held,
        model_name='XGBoost',
        out_file=os.path.join(OUTPUT_DIR, 'fig5_xgb_held_out_scatter.png')
    )

    make_residual_plots(
        xgb_preds, params_held,
        model_name='XGBoost',
        out_file=os.path.join(OUTPUT_DIR, 'fig6_xgb_held_out_residuals.png')
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
