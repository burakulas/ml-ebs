#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            accuracy_score, confusion_matrix, f1_score, classification_report)
import xgboost as xgb

# =============================================================================
# GPU DETECTION
# =============================================================================
try:
    import cupy as cp
    cp.array([1])  # test actual GPU access
    GPU_AVAILABLE = True
    print("GPU (cupy) detected — XGBoost will use CUDA.")
except Exception:
    GPU_AVAILABLE = False
    print("No GPU detected — XGBoost will run on CPU.")

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = "."
INPUT_FILE = os.path.join(WORK_DIR, "processed_data/training_features.pkl")
OUTPUT_DIR_RF = os.path.join(WORK_DIR, "models/models_rf")
OUTPUT_DIR_XGB = os.path.join(WORK_DIR, "models/models_xgb")
HELD_OUT_DIR = os.path.join(WORK_DIR, "models")

PARAMS_TO_TRAIN = ['i', 't2_t1', 'q', 'p1', 'p2']
N_FOLDS = 5
RANDOM_SEED = 42
HELD_OUT_FRACTION = 0.15  # 15% held-out test set

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 25,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'device': 'cuda' if GPU_AVAILABLE else 'cpu'
}

# Classification parameters
RF_CLF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

XGB_CLF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'objective': 'multi:softmax',
    'num_class': 3,
    'device': 'cuda' if GPU_AVAILABLE else 'cpu'
}

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_fold(X_train, X_val, y_train, y_val, model_type='xgb'):
    """Train a single regression model and return predictions."""

    if model_type == 'rf':
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
    else:  # xgb
        X_train_xgb = cp.array(X_train, dtype='float32')
        X_val_xgb = cp.array(X_val, dtype='float32')
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train_xgb, y_train,
                 eval_set=[(X_val_xgb, y_val)],
                 verbose=False)

    y_pred = model.predict(X_val_xgb if model_type != 'rf' else X_val)

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return model, y_pred, r2, mae, rmse


def train_classification_fold(X_train, X_val, y_train, y_val, model_type='xgb'):
    """Train a single classification model and return predictions."""

    if model_type == 'rf':
        model = RandomForestClassifier(**RF_CLF_PARAMS)
        model.fit(X_train, y_train)
    else:  # xgb
        X_train_xgb = cp.array(X_train, dtype='float32')
        X_val_xgb = cp.array(X_val, dtype='float32')
        model = xgb.XGBClassifier(**XGB_CLF_PARAMS)
        model.fit(X_train_xgb, y_train,
                 eval_set=[(X_val_xgb, y_val)],
                 verbose=False)

    y_pred = model.predict(X_val_xgb if model_type != 'rf' else X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_val, y_pred)

    return model, y_pred, accuracy, f1_macro, f1_weighted, conf_matrix


def train_model_type(model_type, X, params_df, feature_names, folds, output_dir):
    """Train all folds for a given model type (regression + classification)."""

    print("\n" + "=" * 80)
    print(f"TRAINING {model_type.upper()}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    cv_results = {param: [] for param in PARAMS_TO_TRAIN}
    all_val_preds = {param: [] for param in PARAMS_TO_TRAIN}
    all_val_targets = {param: [] for param in PARAMS_TO_TRAIN}
    all_val_idx = []

    # Classification storage
    clf_results = []
    all_clf_preds = []
    all_clf_targets = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*80}")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        fold_models = {}
        fold_preds = {}

        for param in PARAMS_TO_TRAIN:
            y_train = params_df.loc[train_idx, param].values
            y_val = params_df.loc[val_idx, param].values

            # Train
            model, y_pred, r2, mae, rmse = train_fold(
                X_train_scaled, X_val_scaled, y_train, y_val, model_type
            )

            print(f"  {param:6s}: R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

            cv_results[param].append({
                'fold': fold_idx,
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            })

            fold_models[param] = model
            fold_preds[param] = y_pred

            # Store for overall analysis
            all_val_preds[param].extend(y_pred)
            all_val_targets[param].extend(y_val)

        # Train morphology classification
        y_train_morph = params_df.loc[train_idx, 'morphology'].values
        y_val_morph = params_df.loc[val_idx, 'morphology'].values

        # Encode morphology labels
        morph_mapping = {'detached': 0, 'semidetached': 1, 'contact': 2}
        y_train_morph_encoded = np.array([morph_mapping[m] for m in y_train_morph])
        y_val_morph_encoded = np.array([morph_mapping[m] for m in y_val_morph])

        clf_model, clf_pred, accuracy, f1_macro, f1_weighted, conf_matrix = train_classification_fold(
            X_train_scaled, X_val_scaled, y_train_morph_encoded, y_val_morph_encoded, model_type
        )

        print(f"  MORPH : Acc={accuracy:.4f}  F1_macro={f1_macro:.4f}  F1_weighted={f1_weighted:.4f}")

        clf_results.append({
            'fold': fold_idx,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix
        })

        fold_models['morphology_classifier'] = clf_model
        all_clf_preds.extend(clf_pred)
        all_clf_targets.extend(y_val_morph_encoded)

        all_val_idx.extend(val_idx)

        # Save fold
        fold_data = {
            'models': fold_models,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'fold': fold_idx,
            'predictions': fold_preds
        }

        fold_file = os.path.join(output_dir, f"{model_type}_fold_{fold_idx}.pkl")
        with open(fold_file, 'wb') as f:
            pickle.dump(fold_data, f)

    # Summary
    print("\n" + "=" * 80)
    print(f"{model_type.upper()} CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    summary = []
    for param in PARAMS_TO_TRAIN:
        results = cv_results[param]
        r2_scores = [r['r2'] for r in results]
        mae_scores = [r['mae'] for r in results]

        print(f"\n{param.upper()}:")
        print(f"  R² =  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"  MAE = {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")

        summary.append({
            'parameter': param,
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores)
        })

    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, f"{model_type}_cv_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # Classification summary
    print(f"\nMORPHOLOGY CLASSIFICATION:")
    accuracy_scores = [r['accuracy'] for r in clf_results]
    f1_macro_scores = [r['f1_macro'] for r in clf_results]
    f1_weighted_scores = [r['f1_weighted'] for r in clf_results]

    print(f"  Accuracy =    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"  F1 (macro) =  {np.mean(f1_macro_scores):.4f} ± {np.std(f1_macro_scores):.4f}")
    print(f"  F1 (weighted) = {np.mean(f1_weighted_scores):.4f} ± {np.std(f1_weighted_scores):.4f}")

    # Overall confusion matrix
    overall_conf_matrix = confusion_matrix(all_clf_targets, all_clf_preds)
    print(f"\nOverall Confusion Matrix:")
    print(f"                Pred: Det   Semi   Cont")
    morph_names = ['detached', 'semidetached', 'contact']
    for i, name in enumerate(morph_names):
        print(f"  True {name:12s}: {overall_conf_matrix[i]}")

    # Per-class F1 scores
    f1_per_class = f1_score(all_clf_targets, all_clf_preds, average=None)
    print(f"\nPer-Class F1 Scores:")
    for i, name in enumerate(morph_names):
        print(f"  {name:12s}: {f1_per_class[i]:.4f}")

    # Save overall predictions for plotting
    overall_summary = {
        'all_val_preds': all_val_preds,
        'all_val_targets': all_val_targets,
        'all_val_idx': all_val_idx,
        'cv_results': cv_results,
        'feature_names': feature_names,
        'clf_results': clf_results,
        'clf_preds': all_clf_preds,
        'clf_targets': all_clf_targets,
        'clf_confusion_matrix': overall_conf_matrix,
        'clf_f1_per_class': f1_per_class,
        'morph_names': morph_names
    }

    overall_file = os.path.join(output_dir, f"{model_type}_summary.pkl")
    with open(overall_file, 'wb') as f:
        pickle.dump(overall_summary, f)

    print(f"\n" + "=" * 80)
    print(f"SAVED: {output_dir}")
    print("=" * 80)
    print(f"\n{len(folds)} folds saved")
    print(f"Summary: {summary_file}")
    print(f"Overall: {overall_file}")

    return cv_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SCRIPT 03: TRAIN RANDOM FOREST AND XGBOOST MODELS")
    print("  WITH 15% HELD-OUT TEST SET")
    print("=" * 80)

    # Load
    print(f"\nLoading: {INPUT_FILE}")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)

    X = data['features']
    params_df = data['params']
    feature_names = data['feature_names']

    print(f"Features: {X.shape}")
    print(f"Parameters to train: {PARAMS_TO_TRAIN}")

    # =========================================================================
    # HELD-OUT TEST SPLIT
    # If held_out_data.pkl exists, load it directly (reproduces paper exactly).
    # Otherwise, create a fresh 15% stratified split and save it.
    # =========================================================================
    os.makedirs(HELD_OUT_DIR, exist_ok=True)
    held_out_file = os.path.join(HELD_OUT_DIR, "held_out_data.pkl")

    if os.path.exists(held_out_file):
        print(f"\n{'='*80}")
        print(f"HELD-OUT SPLIT")
        print(f"{'='*80}")
        print(f"  Loading pre-defined split from: {held_out_file}")
        print(f"  (Reproduces paper results exactly)")
        with open(held_out_file, 'rb') as f:
            held_out_data = pickle.load(f)
        X_cv = held_out_data['cv_features']
        params_cv = held_out_data['cv_params']
        held_out_indices = held_out_data['held_out_indices']
        cv_indices = held_out_data['cv_indices']
        print(f"  CV set:        {len(X_cv)} samples")
        print(f"  Held-out test: {len(held_out_indices)} samples")
    else:
        print(f"\n{'='*80}")
        print(f"HELD-OUT SPLIT")
        print(f"{'='*80}")
        print(f"  No held_out_data.pkl found — creating fresh 15% split.")
        morphology = params_df['morphology'].values
        all_indices = np.arange(len(X))

        cv_indices, held_out_indices = train_test_split(
            all_indices,
            test_size=HELD_OUT_FRACTION,
            stratify=morphology,
            random_state=RANDOM_SEED
        )
        cv_indices = np.sort(cv_indices)
        held_out_indices = np.sort(held_out_indices)

        print(f"  Total samples:    {len(X)}")
        print(f"  CV set:           {len(cv_indices)} ({100*len(cv_indices)/len(X):.1f}%)")
        print(f"  Held-out test:    {len(held_out_indices)} ({100*len(held_out_indices)/len(X):.1f}%)")

        print(f"\n  Morphology distribution:")
        for morph in ['detached', 'semidetached', 'contact']:
            n_cv = np.sum(morphology[cv_indices] == morph)
            n_test = np.sum(morphology[held_out_indices] == morph)
            print(f"    {morph:12s}: CV={n_cv}, Test={n_test} "
                  f"(CV {100*n_cv/len(cv_indices):.1f}%, Test {100*n_test/len(held_out_indices):.1f}%)")

        X_cv = X.iloc[cv_indices].reset_index(drop=True)
        params_cv = params_df.iloc[cv_indices].reset_index(drop=True)

        held_out_data = {
            'held_out_indices': held_out_indices,
            'cv_indices': cv_indices,
            'held_out_features': X.iloc[held_out_indices].reset_index(drop=True),
            'held_out_params': params_df.iloc[held_out_indices].reset_index(drop=True),
            'cv_features': X_cv,
            'cv_params': params_cv,
            'feature_names': feature_names,
            'held_out_fraction': HELD_OUT_FRACTION,
            'random_seed': RANDOM_SEED
        }
        with open(held_out_file, 'wb') as f:
            pickle.dump(held_out_data, f)
        print(f"\n  Saved held-out data: {held_out_file}")
        np.save(os.path.join(HELD_OUT_DIR, "held_out_indices.npy"), held_out_indices)
        np.save(os.path.join(HELD_OUT_DIR, "cv_indices.npy"), cv_indices)
        print(f"  Saved index files: held_out_indices.npy, cv_indices.npy")

    # =========================================================================
    # CV FOLDS — only on the CV subset (85%)
    # =========================================================================
    morphology_cv = params_cv['morphology'].values

    print(f"\nCreating {N_FOLDS}-fold CV splits on {len(X_cv)} CV samples...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(X_cv, morphology_cv))

    # Train both models (on CV subset only)
    rf_results = train_model_type('rf', X_cv, params_cv, feature_names, folds, OUTPUT_DIR_RF)
    xgb_results = train_model_type('xgb', X_cv, params_cv, feature_names, folds, OUTPUT_DIR_XGB)

    # Comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Mean R² across folds)")
    print("=" * 80)
    print(f"\n{'Parameter':<12} | {'RF R²':>10} | {'XGB R²':>10} | {'Winner':>10}")
    print("-" * 50)

    for param in PARAMS_TO_TRAIN:
        rf_r2 = np.mean([r['r2'] for r in rf_results[param]])
        xgb_r2 = np.mean([r['r2'] for r in xgb_results[param]])
        winner = 'RF' if rf_r2 > xgb_r2 else 'XGB'

        print(f"{param:<12} | {rf_r2:>10.4f} | {xgb_r2:>10.4f} | {winner:>10}")

    # Classification comparison
    print("\n" + "=" * 80)
    print("MORPHOLOGY CLASSIFICATION COMPARISON")
    print("=" * 80)

    # Load classification results from summary files
    rf_summary_file = os.path.join(OUTPUT_DIR_RF, "rf_summary.pkl")
    xgb_summary_file = os.path.join(OUTPUT_DIR_XGB, "xgb_summary.pkl")

    with open(rf_summary_file, 'rb') as f:
        rf_summary = pickle.load(f)
    with open(xgb_summary_file, 'rb') as f:
        xgb_summary = pickle.load(f)

    rf_clf_results = rf_summary['clf_results']
    xgb_clf_results = xgb_summary['clf_results']

    rf_accuracy = np.mean([r['accuracy'] for r in rf_clf_results])
    xgb_accuracy = np.mean([r['accuracy'] for r in xgb_clf_results])

    rf_f1_macro = np.mean([r['f1_macro'] for r in rf_clf_results])
    xgb_f1_macro = np.mean([r['f1_macro'] for r in xgb_clf_results])

    print(f"\n{'Metric':<20} | {'RF':>10} | {'XGB':>10} | {'Winner':>10}")
    print("-" * 55)
    print(f"{'Accuracy':<20} | {rf_accuracy:>10.4f} | {xgb_accuracy:>10.4f} | {('RF' if rf_accuracy > xgb_accuracy else 'XGB'):>10}")
    print(f"{'F1 (macro)':<20} | {rf_f1_macro:>10.4f} | {xgb_f1_macro:>10.4f} | {('RF' if rf_f1_macro > xgb_f1_macro else 'XGB'):>10}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"  Held-out test set ({len(held_out_indices)} samples) saved for evaluation by 5g_held_out_evaluation.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
