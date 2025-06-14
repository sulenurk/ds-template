import re
from sklearn.model_selection import StratifiedShuffleSplit

"""##Data Preprocessing"""

def clean_column_names_encoder(df):
    """
    Clean column names by removing special characters that can cause issues with models like XGBoost.

    :param df: DataFrame whose columns need to be cleaned
    :return: DataFrame with cleaned column names
    """
    # Define a regex pattern to match unwanted characters
    pattern = r'[\[\]<>(),:{}]'  # Add any other problematic characters as needed

    # Clean column names
    df.columns = [re.sub(pattern, '', col) for col in df.columns]

    return df
    
def clean_column_names_remainder(df):
    """
    Clean column names by removing special characters that can cause issues with models like XGBoost.

    :param df: DataFrame whose columns need to be cleaned
    :return: DataFrame with cleaned column names
    """
    # Clean column names
    df.columns = df.columns.str.replace("remainder__", "", regex=False)

    return df

""" def data_split(X, y, test_size):

  sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

  for train_index, test_index in sss.split(X, y):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  return X_train, X_test, y_train, y_test """

def data_split_stratified(df, target_col, test_size):
    """
    Split a DataFrame into stratified train-test sets and label them using a 'dataset' column.

    Performs a single stratified split, ensuring that the class distribution
    in the `target_col` is preserved across train and test sets. The result is
    returned as a single DataFrame with a new column:

    - `'dataset' == 1` → training set  
    - `'dataset' == 0` → test set

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset to be split.
    target_col : str
        The name of the column containing the class labels.
    test_size : float
        The proportion of the dataset to include in the test split
        (e.g., 0.2 for 80/20 split).

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an added `'dataset'` column
        indicating the split assignment.

    Notes
    -----
    - The split is stratified using `StratifiedShuffleSplit` from scikit-learn.
    - Only one split is performed (`n_splits=1`).
    - The input DataFrame is not modified in place.
    """    
    df = df.copy() 
    df['dataset'] = -1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_idx, test_idx in sss.split(df, df[target_col]):
        df.loc[train_idx, 'dataset'] = 1  # train set
        df.loc[test_idx, 'dataset'] = 0   # test set

    return df