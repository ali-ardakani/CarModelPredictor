import pandas as pd
import random

import pandas as pd
import random

def balance(df: pd.DataFrame, group_by: str, range_samples: tuple) -> pd.DataFrame:
    """
    Balance the number of samples in a DataFrame by randomly sampling from each group.

    Args:
        df (pd.DataFrame): The input DataFrame to balance.
        group_by (str): The column name to group the DataFrame by.
        range_samples (tuple): A tuple of two integers representing the minimum and maximum number of samples to keep for each group.

    Returns:
        pd.DataFrame: A balanced DataFrame with the same columns as the input DataFrame.
    """
    return df.groupby(group_by).apply(lambda x: x.sample(random.randint(*range_samples), replace=True)).reset_index(drop=True)