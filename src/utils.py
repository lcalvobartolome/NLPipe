import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from langdetect import detect


def det(x: str) -> str:
    """
    Detects the language of a given text
    """
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang


def max_column_length(df: dd.DataFrame, col_name: str) -> int:
    """Returns the maximum length of values in a Dask DataFrame column."""
    lengths = df[col_name].str.len()
    with ProgressBar():
        max_length = lengths.max().compute(scheduler='processes')
    return max_length
