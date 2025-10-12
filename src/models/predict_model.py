"""
Prediction script for bike sharing demand.

This script loads a trained Random Forest model and makes predictions
on new data.

Usage:
    python src/models/predict_model.py

Author: MLOps Team 4
Date: 2025-10-12
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path


def load_model_artifacts(model_dir):
    """
    Load the trained model, scaler, and feature names.

    Args:
        model_dir (Path): Directory containing model artifacts

    Returns:
        tuple: (model, scaler, feature_names)
    """
    print("Loading model artifacts...")

    # Load best model (Gradient Boosting from multi-model training)
    model_path = model_dir / 'best_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {model_path}")

    # Load scaler
    scaler_path = model_dir / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from: {scaler_path}")

    # Load feature names
    features_path = model_dir / 'feature_names.json'
    with open(features_path, 'r') as f:
        feature_data = json.load(f)
        feature_names = feature_data['features']
    print(f"Feature names loaded from: {features_path}")
    print(f"Total features: {len(feature_names)}")

    return model, scaler, feature_names


def preprocess_input(input_data):
    """
    Preprocess input data to match training format.

    Args:
        input_data (pd.DataFrame or dict): Raw input data

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for prediction
    """
    # Convert dict to dataframe if needed
    if isinstance(input_data, dict):
        if not isinstance(next(iter(input_data.values())), list):
            # Single prediction - wrap values in lists
            input_data = {k: [v] for k, v in input_data.items()}
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()

    # Drop columns that shouldn't be in input
    columns_to_drop = ['instant', 'dteday', 'casual', 'registered',
                      'atemp', 'mixed_type_col', 'cnt']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def engineer_features(df):
    """
    Apply the same feature engineering as during training.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
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

    # Convert to string
    for col in categorical_cols:
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].astype(str)

    # Get dummies
    df_encoded = pd.get_dummies(df_engineered, columns=categorical_cols, drop_first=True)

    return df_encoded


def align_features(df, expected_features):
    """
    Ensure input dataframe has all expected features in the correct order.

    Args:
        df (pd.DataFrame): Processed input dataframe
        expected_features (list): List of feature names from training

    Returns:
        pd.DataFrame: Aligned dataframe with all expected features
    """
    # Add missing columns with 0s
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[expected_features]

    return df


def scale_features(df, scaler):
    """
    Scale numerical features using the fitted scaler.

    Args:
        df (pd.DataFrame): Input dataframe
        scaler: Fitted StandardScaler from training

    Returns:
        pd.DataFrame: Dataframe with scaled numerical features
    """
    df_scaled = df.copy()

    # Identify numerical columns
    numerical_cols = ['yr', 'mnth', 'hr', 'temp', 'hum', 'windspeed']
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    if numerical_cols:
        df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])

    return df_scaled


def predict(model, scaler, feature_names, input_data):
    """
    Make predictions on new data.

    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names (list): Expected feature names
        input_data: Raw input data (dict or DataFrame)

    Returns:
        np.array: Predictions
    """
    # Preprocess
    df = preprocess_input(input_data)

    # Engineer features
    df_engineered = engineer_features(df)

    # Align features
    df_aligned = align_features(df_engineered, feature_names)

    # Scale features
    df_scaled = scale_features(df_aligned, scaler)

    # Make predictions
    predictions = model.predict(df_scaled)

    return predictions


def format_prediction_output(predictions, input_data):
    """
    Format prediction results for display.

    Args:
        predictions (np.array): Model predictions
        input_data: Original input data

    Returns:
        pd.DataFrame: Formatted results
    """
    if isinstance(input_data, dict):
        if not isinstance(next(iter(input_data.values())), list):
            input_data = {k: [v] for k, v in input_data.items()}
        df_input = pd.DataFrame(input_data)
    else:
        df_input = input_data.copy()

    # Create results dataframe
    results = df_input.copy()
    results['predicted_cnt'] = predictions.round(0).astype(int)

    return results


def main():
    """Main prediction pipeline with examples."""

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / 'models'

    print("="*60)
    print("BIKE SHARING DEMAND PREDICTION - INFERENCE")
    print("="*60)

    # Load model artifacts
    model, scaler, feature_names = load_model_artifacts(model_dir)

    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)

    # Example 1: Single prediction - Morning rush hour on a warm day
    print("\n1. Morning rush hour (8 AM, warm Monday, working day):")
    example1 = {
        'season': 3,        # Summer
        'yr': 1,            # 2012
        'mnth': 7,          # July
        'hr': 8,            # 8 AM
        'holiday': 0,       # Not a holiday
        'weekday': 1,       # Monday
        'workingday': 1,    # Working day
        'weathersit': 1,    # Clear weather
        'temp': 0.68,       # Warm temperature
        'hum': 0.65,        # Moderate humidity
        'windspeed': 0.15   # Light wind
    }

    pred1 = predict(model, scaler, feature_names, example1)
    print(f"Input: {example1}")
    print(f"Predicted bike rentals: {int(pred1[0])}")

    # Example 2: Evening on a cold winter night
    print("\n2. Evening (6 PM, cold Friday, working day):")
    example2 = {
        'season': 1,        # Winter
        'yr': 1,            # 2012
        'mnth': 1,          # January
        'hr': 18,           # 6 PM
        'holiday': 0,       # Not a holiday
        'weekday': 5,       # Friday
        'workingday': 1,    # Working day
        'weathersit': 2,    # Misty
        'temp': 0.22,       # Cold
        'hum': 0.75,        # High humidity
        'windspeed': 0.25   # Moderate wind
    }

    pred2 = predict(model, scaler, feature_names, example2)
    print(f"Input: {example2}")
    print(f"Predicted bike rentals: {int(pred2[0])}")

    # Example 3: Weekend afternoon
    print("\n3. Weekend afternoon (2 PM, nice spring Saturday):")
    example3 = {
        'season': 2,        # Spring
        'yr': 1,            # 2012
        'mnth': 4,          # April
        'hr': 14,           # 2 PM
        'holiday': 0,       # Not a holiday
        'weekday': 6,       # Saturday
        'workingday': 0,    # Weekend
        'weathersit': 1,    # Clear
        'temp': 0.55,       # Mild
        'hum': 0.60,        # Moderate humidity
        'windspeed': 0.18   # Light wind
    }

    pred3 = predict(model, scaler, feature_names, example3)
    print(f"Input: {example3}")
    print(f"Predicted bike rentals: {int(pred3[0])}")

    # Example 4: Batch prediction
    print("\n4. Batch prediction (multiple time slots):")
    batch_data = {
        'season': [3, 3, 3, 3],
        'yr': [1, 1, 1, 1],
        'mnth': [7, 7, 7, 7],
        'hr': [7, 12, 17, 22],  # Different hours
        'holiday': [0, 0, 0, 0],
        'weekday': [3, 3, 3, 3],  # Wednesday
        'workingday': [1, 1, 1, 1],
        'weathersit': [1, 1, 1, 1],
        'temp': [0.60, 0.75, 0.70, 0.55],
        'hum': [0.65, 0.60, 0.65, 0.70],
        'windspeed': [0.15, 0.12, 0.18, 0.20]
    }

    predictions_batch = predict(model, scaler, feature_names, batch_data)
    results = format_prediction_output(predictions_batch, batch_data)

    print("\nBatch Results:")
    print(results[['hr', 'temp', 'predicted_cnt']])

    print("\n" + "="*60)
    print("PREDICTION COMPLETED!")
    print("="*60)
    print("\nTo use this script in your code:")
    print("  from src.models.predict_model import load_model_artifacts, predict")
    print("  model, scaler, features = load_model_artifacts(model_dir)")
    print("  predictions = predict(model, scaler, features, your_data)")


if __name__ == '__main__':
    main()
