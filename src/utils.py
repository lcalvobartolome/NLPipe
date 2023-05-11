import pathlib
import shutil

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from langdetect import detect


def det(x: str) -> str:
    """
    Detects the language of a given text

    Parameters
    ----------
    x : str
        Text whose language is to be detected

    Returns
    -------
    lang : str
        Language of the text
    """

    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang


def max_column_length(df: dd.DataFrame,
                      col_name: str) -> int:
    """
    Returns the maximum length of values in a Dask DataFrame column.

    Parameters
    ----------
    df : dd.DataFrame
        Dask DataFrame
    col_name : str
        Name of the column whose length is to be calculated

    Returns
    -------
    max_length: int
        Maximum length of the values in the column
    """

    lengths = df[col_name].str.len()
    with ProgressBar():
        max_length = lengths.max().compute(scheduler='processes')
    return max_length


def save_parquet(outFile: pathlib.Path,
                 df: dd.DataFrame, nw=0) -> None:
    """
    Saves a Dask DataFrame in a parquet file.

    Parameters
    ----------
    outFile : pathlib.Path
        Path to the parquet file to be saved
    df : dd.DataFrame
        Dask DataFrame to be saved
    nw : int, optional
        Number of workers to use with Dask
    """
    if outFile.is_file():
        outFile.unlink()
    elif outFile.is_dir():
        shutil.rmtree(outFile)

    with ProgressBar():
        if nw > 0:
            df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                'scheduler': 'processes', 'num_workers': nw})
        else:
            # Use Dask default number of workers (i.e., number of cores)
            df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                'scheduler': 'processes'})

    return
