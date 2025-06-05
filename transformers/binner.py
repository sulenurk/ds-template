import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def learn_binner(df):
    """
    Learn custom binning thresholds with predefined rules for specific columns.
    No imputation - assumes no missing values in columns to be binned.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data to fit the binners on (must not contain missing values in binned columns)

    Returns:
    --------
    dict
        Dictionary containing fitted binning transformers:
        {
            'binners': {
                'col1': {'binner': binner, 'bin_labels': [...]},
                ...
            },
            'column_config': {
                'col1': {'bin_edges': [...], 'labels': [...]},
                ...
            }
        }
    """
    # MANUAL CUSTOM CONFIGURATION - Edit this for each column
    column_config = {
        'age': {
            'bin_edges': [0, 18, 35, 50, 65, np.inf],
            'labels': ['Child', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior']
        },
        'income': {
            'bin_edges': [-np.inf, 30000, 60000, 100000, np.inf],
            'labels': ['Low', 'Medium', 'High', 'Very High']
        },
        'score': {
            'bin_edges': [0, 4, 7, 10],
            'labels': ['Low', 'Medium', 'High']
        }
    }

    # Initialize binning info storage
    bin_info = {
        'binners': {},
        'column_config': column_config
    }

    for col_name, config in column_config.items():
        if col_name not in df.columns:
            continue

        # Create and fit binner with custom edges
        binner = KBinsDiscretizer(
            n_bins=len(config['labels']),
            encode='ordinal',
            strategy='uniform',
            bins=config['bin_edges']
        )
        binner.fit(df[[col_name]])

        # Store the fitted binner
        bin_info['binners'][col_name] = {
            'binner': binner,
            'bin_labels': config['labels'],
            'bin_edges': config['bin_edges']
        }

    return bin_info

def apply_binner(df, bin_info):
    """
    Apply learned custom binning to new data

    Parameters:
    -----------
    df : pd.DataFrame
        Data to transform (must not contain missing values in binned columns)
    bin_info : dict
        Binning information from learn_binner

    Returns:
    --------
    pd.DataFrame
        Transformed data with binned columns
    """
    df_binned = df.copy()

    for col_name, binner_info in bin_info['binners'].items():
        if col_name not in df.columns:
            continue

        # Apply binning directly (no imputation)
        binned_col = binner_info['binner'].transform(df[[col_name]])

        # Convert to categorical with labels
        df_binned[col_name] = pd.Categorical.from_codes(
            codes=binned_col.astype(int).flatten(),
            categories=binner_info['bin_labels']
        )

    return df_binned