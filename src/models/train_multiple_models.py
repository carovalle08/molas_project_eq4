"""
Multi-Model Training and Evaluation for Bike Sharing Demand Prediction.

This script:
1. Trains 3 different ML models (Random Forest, Gradient Boosting, Ridge Regression)
2. Performs hyperparameter tuning using GridSearchCV
3. Evaluates models using multiple performance metrics
4. Compares model performance and selects the best model
5. Generates comprehensive visualizations and reports

Models:
- Random Forest Regressor: Ensemble of decision trees, good for non-linear patterns
- Gradient Boosting Regressor: Sequential ensemble, often highest accuracy
- Ridge Regression: Linear model with L2 regularization, interpretable baseline

Author: MLOps Team 4
Date: 2025-10-12
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path):
    """
    Load the dataset and perform initial preprocessing.

    Args:
        data_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")

    # Drop columns based on correlation analysis
    columns_to_drop = [
        'instant',      # Just an index
        'dteday',       # Date string (already have yr, mnth, hr)
        'casual',       # Data leakage (part of target)
        'registered',   # Data leakage (part of target)
        'atemp',        # High correlation with temp (0.97)
        'mixed_type_col'  # Not useful for prediction
    ]

    df = df.drop(columns=columns_to_drop)
    print(f"Shape after dropping columns: {df.shape}")
    print(f"Remaining columns: {list(df.columns)}")

    # Check for missing values
    if df.isnull().sum().any():
        print("\nWarning: Missing values found!")
        print(df.isnull().sum())
    else:
        print("\nNo missing values found.")

    return df


def engineer_features(df):
    """
    Engineer features for the model.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("\nEngineering features...")

    df_engineered = df.copy()

    # Create hour bins
    df_engineered['hour_bin'] = pd.cut(
        df_engineered['hr'],
        bins=[0, 6, 11, 17, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )

    # Create temperature bins
    df_engineered['temp_bin'] = pd.cut(
        df_engineered['temp'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['cold', 'mild', 'warm', 'hot'],
        include_lowest=True
    )

    # One-hot encode categorical variables
    categorical_cols = ['season', 'weathersit', 'weekday', 'holiday',
                       'workingday', 'hour_bin', 'temp_bin']

    for col in categorical_cols:
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].astype(str)

    df_encoded = pd.get_dummies(df_engineered, columns=categorical_cols, drop_first=True)

    print(f"Shape after feature engineering: {df_encoded.shape}")

    return df_encoded


def split_data(df, target_col='cnt', test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting data (test_size={test_size})...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale numerical features.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features

    Returns:
        tuple: Scaled X_train, X_test, and the scaler object
    """
    print("\nScaling numerical features...")

    numerical_cols = ['yr', 'mnth', 'hr', 'temp', 'hum', 'windspeed']
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train_scaled, X_test_scaled, scaler


def get_model_configs():
    """
    Define models and their hyperparameter grids.

    Returns:
        dict: Dictionary with model configurations
    """
    configs = {
        'random_forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            },
            'description': 'Ensemble of decision trees, robust to outliers and non-linear patterns'
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'description': 'Sequential ensemble learning, often achieves highest accuracy'
        },
        'ridge_regression': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'description': 'Linear model with L2 regularization, interpretable and fast'
        }
    }

    return configs


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        float: MAPE value
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_evaluate_model(model_name, model_config, X_train, y_train, X_test, y_test):
    """
    Train a model with hyperparameter tuning and evaluate it.

    Args:
        model_name (str): Name of the model
        model_config (dict): Model configuration with model and params
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        dict: Results including best model, metrics, and timing
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper().replace('_', ' ')}")
    print(f"Description: {model_config['description']}")
    print(f"{'='*70}")

    # GridSearchCV setup
    print(f"\nHyperparameter Grid:")
    for param, values in model_config['params'].items():
        print(f"  {param}: {values}")

    grid_search = GridSearchCV(
        model_config['model'],
        model_config['params'],
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Training
    print(f"\nStarting GridSearchCV with 5-fold cross-validation...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    start_time = time.time()
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

    # Calculate metrics
    metrics = {
        'train': {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred),
            'mape': calculate_mape(y_train.values, y_train_pred)
        },
        'test': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred),
            'mape': calculate_mape(y_test.values, y_test_pred)
        },
        'cv': {
            'mean_rmse': -grid_search.best_score_,
            'std_rmse': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        },
        'timing': {
            'train_time_seconds': train_time,
            'inference_time_ms': inference_time
        },
        'best_params': grid_search.best_params_
    }

    # Print results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print("\nTraining Set:")
    print(f"  RMSE:  {metrics['train']['rmse']:.2f}")
    print(f"  MAE:   {metrics['train']['mae']:.2f}")
    print(f"  R2:    {metrics['train']['r2']:.4f}")
    print(f"  MAPE:  {metrics['train']['mape']:.2f}%")

    print("\nTest Set:")
    print(f"  RMSE:  {metrics['test']['rmse']:.2f}")
    print(f"  MAE:   {metrics['test']['mae']:.2f}")
    print(f"  R2:    {metrics['test']['r2']:.4f}")
    print(f"  MAPE:  {metrics['test']['mape']:.2f}%")

    print("\nCross-Validation:")
    print(f"  Mean CV RMSE: {metrics['cv']['mean_rmse']:.2f} (+/- {metrics['cv']['std_rmse']:.2f})")

    print("\nTiming:")
    print(f"  Training time:   {metrics['timing']['train_time_seconds']:.2f} seconds")
    print(f"  Inference time:  {metrics['timing']['inference_time_ms']:.4f} ms/sample")

    # Check overfitting
    r2_diff = metrics['train']['r2'] - metrics['test']['r2']
    if r2_diff > 0.1:
        print(f"\nWarning: Possible overfitting (R2 difference: {r2_diff:.4f})")
    else:
        print(f"\nModel generalizes well (R2 difference: {r2_diff:.4f})")

    results = {
        'model': best_model,
        'metrics': metrics,
        'predictions': {
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred
        }
    }

    return results


def create_comparison_table(all_results):
    """
    Create a comparison table of all models.

    Args:
        all_results (dict): Results from all models

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []

    for model_name, results in all_results.items():
        metrics = results['metrics']
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Test RMSE': metrics['test']['rmse'],
            'Test MAE': metrics['test']['mae'],
            'Test R2': metrics['test']['r2'],
            'Test MAPE': metrics['test']['mape'],
            'CV RMSE': metrics['cv']['mean_rmse'],
            'Train Time (s)': metrics['timing']['train_time_seconds'],
            'Inference (ms)': metrics['timing']['inference_time_ms'],
            'Overfitting': metrics['train']['r2'] - metrics['test']['r2']
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Test RMSE')

    return df_comparison


def plot_model_comparison(df_comparison, output_dir):
    """
    Create comparison visualizations.

    Args:
        df_comparison (pd.DataFrame): Comparison table
        output_dir (Path): Directory to save plots
    """
    print("\nGenerating comparison visualizations...")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # 1. RMSE Comparison
    ax = axes[0, 0]
    x = np.arange(len(df_comparison))
    width = 0.35
    ax.bar(x - width/2, df_comparison['Test RMSE'], width, label='Test RMSE', alpha=0.8)
    ax.bar(x + width/2, df_comparison['CV RMSE'], width, label='CV RMSE', alpha=0.8)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. R2 Score Comparison
    ax = axes[0, 1]
    ax.barh(df_comparison['Model'], df_comparison['Test R2'], alpha=0.8, color='green')
    ax.set_xlabel('R2 Score', fontsize=12)
    ax.set_title('R2 Score Comparison (Higher is Better)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 3. MAE and MAPE
    ax = axes[1, 0]
    x = np.arange(len(df_comparison))
    ax2 = ax.twinx()
    ax.bar(x - width/2, df_comparison['Test MAE'], width, label='MAE', alpha=0.8, color='orange')
    ax2.bar(x + width/2, df_comparison['Test MAPE'], width, label='MAPE (%)', alpha=0.8, color='red')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12, color='orange')
    ax2.set_ylabel('MAPE (%)', fontsize=12, color='red')
    ax.set_title('MAE and MAPE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=15, ha='right')
    ax.tick_params(axis='y', labelcolor='orange')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # 4. Training Time vs Performance
    ax = axes[1, 1]
    scatter = ax.scatter(df_comparison['Train Time (s)'], df_comparison['Test R2'],
                        s=200, alpha=0.6, c=df_comparison.index, cmap='viridis')
    for idx, row in df_comparison.iterrows():
        ax.annotate(row['Model'], (row['Train Time (s)'], row['Test R2']),
                   fontsize=9, ha='center', va='bottom')
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Test R2 Score', fontsize=12)
    ax.set_title('Training Time vs Performance', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def plot_predictions_comparison(all_results, output_dir):
    """
    Plot actual vs predicted for all models.

    Args:
        all_results (dict): Results from all models
        output_dir (Path): Directory to save plots
    """
    print("Generating predictions comparison plots...")

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Actual vs Predicted Values (Test Set)', fontsize=16, fontweight='bold')

    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        y_test = results['predictions']['y_test']
        y_pred = results['predictions']['y_test_pred']

        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
               'r--', lw=2, label='Perfect prediction')

        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f"{model_name.replace('_', ' ').title()}\nR2={results['metrics']['test']['r2']:.4f}",
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'predictions_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Predictions comparison plot saved to: {output_path}")
    plt.close()


def plot_feature_importance_comparison(all_results, feature_names, output_dir):
    """
    Plot feature importance for tree-based models.

    Args:
        all_results (dict): Results from all models
        feature_names (list): List of feature names
        output_dir (Path): Directory to save plots
    """
    print("Generating feature importance plots...")

    # Filter tree-based models
    tree_models = {name: results for name, results in all_results.items()
                  if hasattr(results['model'], 'feature_importances_')}

    if not tree_models:
        print("No tree-based models found, skipping feature importance plots")
        return

    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')

    for idx, (model_name, results) in enumerate(tree_models.items()):
        ax = axes[idx]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': results['model'].feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        ax.barh(range(len(importance_df)), importance_df['importance'], alpha=0.8)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f"{model_name.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'feature_importance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {output_path}")
    plt.close()


def save_results(all_results, df_comparison, scaler, feature_names, model_dir, reports_dir):
    """
    Save all models, metrics, and results.

    Args:
        all_results (dict): Results from all models
        df_comparison (pd.DataFrame): Comparison table
        scaler: Fitted scaler
        feature_names (list): Feature names
        model_dir (Path): Directory to save models
        reports_dir (Path): Directory to save reports
    """
    print("\nSaving results...")

    # Save comparison table
    comparison_path = reports_dir / 'model_comparison_results.csv'
    df_comparison.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")

    # Save all models
    for model_name, results in all_results.items():
        model_path = model_dir / f'{model_name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(results['model'], f)
        print(f"{model_name} model saved to: {model_path}")

    # Save best model
    best_model_name = df_comparison.iloc[0]['Model'].lower().replace(' ', '_')
    best_model = all_results[best_model_name]['model']
    best_model_path = model_dir / 'best_model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model ({best_model_name}) saved to: {best_model_path}")

    # Save scaler
    scaler_path = model_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save feature names
    features_path = model_dir / 'feature_names.json'
    with open(features_path, 'w') as f:
        json.dump({'features': feature_names}, f, indent=2)
    print(f"Feature names saved to: {features_path}")

    # Save all metrics
    all_metrics = {}
    for model_name, results in all_results.items():
        all_metrics[model_name] = results['metrics']

    metrics_path = model_dir / 'all_models_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"All metrics saved to: {metrics_path}")

    # Save best hyperparameters
    best_params = {}
    for model_name, results in all_results.items():
        best_params[model_name] = results['metrics']['best_params']

    params_path = model_dir / 'best_hyperparameters.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Best hyperparameters saved to: {params_path}")


def main():
    """Main pipeline for multi-model training and evaluation."""

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'raw non-dvc' / 'bike_sharing_cleaned_v1.csv'
    model_dir = project_root / 'models'
    reports_dir = project_root / 'reports' / 'figures'

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MULTI-MODEL TRAINING AND EVALUATION")
    print("Bike Sharing Demand Prediction")
    print("="*70)
    print(f"\nProject root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"Models directory: {model_dir}")
    print(f"Reports directory: {reports_dir}")

    # Load and preprocess
    df = load_and_preprocess_data(data_path)

    # Engineer features
    df_engineered = engineer_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_engineered)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Get model configurations
    model_configs = get_model_configs()

    print(f"\n{'='*70}")
    print(f"MODELS TO TRAIN: {len(model_configs)}")
    print(f"{'='*70}")
    for name, config in model_configs.items():
        print(f"\n{name.upper().replace('_', ' ')}")
        print(f"  Description: {config['description']}")
        print(f"  Hyperparameters to tune: {list(config['params'].keys())}")

    # Train all models
    all_results = {}
    for model_name, model_config in model_configs.items():
        results = train_and_evaluate_model(
            model_name, model_config,
            X_train_scaled, y_train,
            X_test_scaled, y_test
        )
        all_results[model_name] = results

    # Create comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    df_comparison = create_comparison_table(all_results)
    print("\n" + df_comparison.to_string(index=False))

    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    best_model = df_comparison.iloc[0]
    print(f"\nModel: {best_model['Model']}")
    print(f"Test RMSE: {best_model['Test RMSE']:.2f}")
    print(f"Test MAE: {best_model['Test MAE']:.2f}")
    print(f"Test R2: {best_model['Test R2']:.4f}")
    print(f"Test MAPE: {best_model['Test MAPE']:.2f}%")

    # Generate visualizations
    plot_model_comparison(df_comparison, reports_dir)
    plot_predictions_comparison(all_results, reports_dir)
    plot_feature_importance_comparison(all_results, X_train_scaled.columns.tolist(), reports_dir)

    # Save results
    save_results(all_results, df_comparison, scaler, X_train_scaled.columns.tolist(),
                model_dir, reports_dir)

    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the comparison visualizations in reports/figures/")
    print("2. Check the detailed metrics in models/all_models_metrics.json")
    print("3. Use the best model from models/best_model.pkl for predictions")


if __name__ == '__main__':
    main()
