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


def max_column_length(df, col_name, use_dask=False):
    """
    Returns the maximum length of values in a DataFrame column.

    Parameters
    ----------
    df : DataFrame
        DataFrame (pandas or Dask)
    col_name : str
        Name of the column whose length is to be calculated
    use_dask : bool, optional
        Flag to indicate whether the DataFrame is Dask or not

    Returns
    -------
    max_length: int
        Maximum length of the values in the column
    """

    if use_dask:
        lengths = df[col_name].str.len()
        with ProgressBar():
            max_length = lengths.max().compute(scheduler='processes')
    else:
        lengths = df[col_name].str.len()
        max_length = lengths.max()

    return max_length


def save_parquet(outFile: pathlib.Path,
                 df: dd.DataFrame,
                 use_dask=False,
                 nw=0) -> None:
    """
    Saves a Dask DataFrame in a parquet file.

    Parameters
    ----------
    outFile : pathlib.Path
        Path to the parquet file to be saved
    df : DataFrame
        DataFrame (pandas or Dask)
    use_dask : bool, optional
        Flag to indicate whether the DataFrame is Dask or not
    nw : int, optional
        Number of workers to use with Dask
    """
    if outFile.is_file():
        outFile.unlink()
    elif outFile.is_dir():
        shutil.rmtree(outFile)

    if use_dask:
        with ProgressBar():
            if nw > 0:
                df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                    'scheduler': 'processes', 'num_workers': nw})
            else:
                # Use Dask default number of workers (i.e., number of cores)
                df.to_parquet(outFile, write_index=False, schema="infer", compute_kwargs={
                    'scheduler': 'processes'})
    else:
        df.to_parquet(outFile)
        #df.to_parquet(outFile, write_index=False)

    return
