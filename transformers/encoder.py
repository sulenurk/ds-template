import pandas as pd
from utils.data_prep_utils import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def learn_encoder(df, min_frequency=5):
    """
    Learns the encoding configuration from the training DataFrame.

    Parameters:
    df (pd.DataFrame): The input training DataFrame containing categorical variables.
    min_frequency (int): Minimum frequency for a category to avoid being grouped under 'others'.

    Returns:
    dict: A dictionary containing the fitted encoder and relevant metadata.
    """
    # Clean column names (e.g., remove whitespace or special characters)
    df = clean_column_names_encoder(df)

    # Identify all categorical columns (object or category dtype)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # If there are no categorical columns, return None (nothing to encode)
    if not categorical_cols:
        return None

    # Define a OneHotEncoder that drops the first category and groups infrequent values into 'infrequent'
    encoder = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore',
        min_frequency=min_frequency
    )
    
    # Create a column transformer to apply the encoder only on categorical columns
    ct = ColumnTransformer(
        transformers=[('cat', encoder, categorical_cols)],
        remainder='passthrough'  # Keep other columns unchanged
    )

    # Fit the column transformer to the data
    ct.fit(df)
    
    # Extract the OneHotEncoder part of the transformer
    onehot_encoder = ct.named_transformers_['cat']

    # Store which categories were considered infrequent during fitting
    infrequent_categories = {
    col: inf.tolist() if inf is not None else []
    for col, inf in zip(categorical_cols, onehot_encoder.infrequent_categories_)
    }

    # Bundle all relevant information into a dictionary
    encoder_info = {
        'encoder': ct,
        'categorical_cols': categorical_cols,
        'feature_names': ct.get_feature_names_out(),
        'infrequent_categories': infrequent_categories,
        'min_frequency': min_frequency
    }

    return encoder_info

def apply_encoder(df, encoder_info, columns_to_drop=None):
    """
    Applies the previously learned encoder to a new DataFrame.
    New categories not seen during training are encoded as 'others' if they meet the frequency threshold.

    Parameters:
    df (pd.DataFrame): The new DataFrame to encode.
    encoder_info (dict): Output dictionary from the learn_encoder function.

    Returns:
    pd.DataFrame: Transformed DataFrame with one-hot encoded features and '_others' columns.
    """
    # Clean column names to match the training format
    df = clean_column_names_encoder(df)

    # If no encoder was learned (no categorical columns), return the original DataFrame
    if encoder_info is None:
        return df.copy()
    
        # Drop specified columns
    if columns_to_drop:
        dropped_df = df[columns_to_drop].copy()
        df_input = df.drop(columns=columns_to_drop)
    else:
        dropped_df = None
        df_input = df

    # Unpack encoder information
    ct = encoder_info['encoder']
    categorical_cols = encoder_info['categorical_cols']
    feature_names = encoder_info['feature_names']
    min_frequency = encoder_info['min_frequency']
    infrequent_categories = encoder_info['infrequent_categories']

    # Apply the transformation to the new data
    encoded_array = ct.transform(df_input)

    # Extract the OneHotEncoder used in the transformer
    onehot_encoder = ct.named_transformers_['cat']

    # Clean up feature names (remove 'cat__' prefix)
    new_feature_names = []
    for name in feature_names:
        if name.startswith('cat__'):
            orig_col = name.split('__')[1]
            new_name = name.replace(f'cat__{orig_col}_', f'{orig_col}_')
            new_feature_names.append(new_name)
        else:
            new_feature_names.append(name)

    # Convert encoded array into a DataFrame
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=new_feature_names,
        index=df_input.index
    )

    # Process each categorical column to ensure '_others' column is correctly handled
    for col in categorical_cols:
        # Identify all encoded columns that belong to this feature
        cols_for_feature = [c for c in encoded_df.columns if c.startswith(f'{col}_')]

        if cols_for_feature:
            # Assume the last column in the group is the 'others' column
            others_col = f'{col}_others'

            # Rename the last dummy column to explicitly indicate it is for infrequent/unknown categories
            encoded_df[others_col] = encoded_df[cols_for_feature[-1]]

            # Drop the original unnamed 'others' column
            encoded_df.drop(cols_for_feature[-1], axis=1, inplace=True)

            # Convert the column to integer (0 or 1)
            encoded_df[others_col] = encoded_df[others_col].astype(int)

            # Get all training categories for this column
            train_categories = onehot_encoder.categories_[categorical_cols.index(col)]

            # Check values in the new data that were not seen in training
            test_values = df[col].dropna().astype(str)
            unseen = test_values[~test_values.isin(train_categories)]

            # Among unseen values, keep only those that occur frequently enough
            valid_unseen = unseen.value_counts()[lambda x: x >= min_frequency].index.tolist()

            if valid_unseen:
                # For valid unseen values, set 'others' column to 1 in the corresponding rows
                encoded_df[others_col] |= df[col].isin(valid_unseen).astype(int)

    encoded_df = clean_column_names_remainder(encoded_df)

    # Reattach dropped columns if any
    if dropped_df is not None:
        encoded_df = pd.concat([encoded_df, dropped_df], axis=1)

    return encoded_df

def save_encoder(encoder_info, filepath):
    """
    Save standardizer info (scalers and config) to disk.

    Parameters:
    -----------
    stand_info : dict
        Output of learn_standardizer containing scalers and config
    filepath : str
        Path to save the serialized file (.joblib recommended)
    """
    joblib.dump(encoder_info, filepath)
    print(f"Encoder saved to {filepath}")

def load_encoder(filepath):
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
    encoder_info = joblib.load(filepath)
    print(f"Encoder loaded from {filepath}")
    return encoder_info
