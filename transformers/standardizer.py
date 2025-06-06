import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
import joblib

def learn_standardizer(df):
    """
    Learn standardization parameters with predefined scaling methods for specific columns.
    Scaling configuration is manually defined inside the function.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data to fit the scalers on

    Returns:
    --------
    dict
        Dictionary containing:
        - 'scalers': Dictionary of fitted scaler objects for each column
        - 'column_mapping': Mapping of columns to their scaling methods
        - 'scale_config': The full scaling configuration used
    """
    # MANUAL CONFIGURATION - Modify this section as needed
    # Define which columns should use which scaling method
    scale_config = {
        'log1p_standard': ["Rotational speed rpm"],
        'standard': ["Air temperature K", "Process temperature K", "Torque Nm", "Tool wear min"],
        'minmax': [],
    }

    # Filter columns that exist in the DataFrame
    for method, cols in scale_config.items():
        scale_config[method] = [col for col in cols if col in df.columns]

    stand_info = {
        'scalers': {},
        'column_mapping': {},
        'scale_config': scale_config
    }

    for method, cols in scale_config.items():
        for col in cols:
            if method == 'log1p_standard':
                # Pipeline: log1p -> StandardScaler
                scaler = Pipeline([
                    ('log1p', FunctionTransformer(np.log1p)),
                    ('standard', StandardScaler())
                ])
                stand_info['column_mapping'][col] = 'log1p_standard'

            elif method == 'standard':
                scaler = StandardScaler()
                stand_info['column_mapping'][col] = 'standard'

            elif method == 'minmax':
                scaler = MinMaxScaler()
                stand_info['column_mapping'][col] = 'minmax'

            scaler.fit(df[[col]])
            stand_info['scalers'][col] = scaler

    return stand_info

def apply_standardizer(df, stand_info, columns_to_drop=None):
    """
    Apply the learned scaling transformations to new data.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to transform
    stand_info : dict
        Scaling information dictionary returned by learn_standardizer

    Returns:
    --------
    pd.DataFrame
        Transformed data with scaled columns
    """
    # Drop specified columns
    if columns_to_drop:
        dropped_df = df[columns_to_drop].copy()
        df_scaled = df.drop(columns=columns_to_drop).copy()
    else:
        dropped_df = None
        df_scaled = df.copy()

    for col, scaler in stand_info['scalers'].items():
        if col in df.columns:
            # Apply the appropriate transformation
            # Handle FunctionTransformer separately if needed
            df_scaled[col] = scaler.transform(df[[col]]).flatten()

    # Reattach dropped columns if any
    if dropped_df is not None:
        df_scaled = pd.concat([df_scaled, dropped_df], axis=1)

    return df_scaled

def save_standardizer(stand_info, filepath):
    """
    Save standardizer info (scalers and config) to disk.

    Parameters:
    -----------
    stand_info : dict
        Output of learn_standardizer containing scalers and config
    filepath : str
        Path to save the serialized file (.joblib recommended)
    """
    joblib.dump(stand_info, filepath)
    print(f"Standardizer saved to {filepath}")

def load_standardizer(filepath):
    """
    Load standardizer info from disk.

    Parameters:
    -----------
    filepath : str
        Path to the serialized standardizer file

    Returns:
    --------
    dict
        Deserialized standardizer info (same structure as returned by learn_standardizer)
    """
    stand_info = joblib.load(filepath)
    print(f"Standardizer loaded from {filepath}")
    return stand_info
