import argparse
import json
import logging
import pathlib
import shutil
import sys
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from pyfiglet import figlet_format
from termcolor import cprint

from src.embeddings_manager import EmbeddingsManager
from src.pipe import Pipe
from src.utils import det, max_column_length, save_parquet

# ########################
# Main body of application


def main():

    # Read input arguments
    parser = argparse.ArgumentParser(
        description="NLPipe")
    parser.add_argument("--source_path", type=str, default=None,
                        required=True, help="Path to the source file")
    parser.add_argument("--source_type", type=str, default='parquet',
                        required=False, help="Source file's format")
    parser.add_argument("--source", type=str, default=None,
                        required=True, help="Name of the dataset to be preprocessed (e.g., cordis, scholar, etc.)")
    parser.add_argument("--destination_path", type=str, default=None,
                        required=True, help="Path to save the preprocessed data")
    parser.add_argument("--stw_path", type=str, default="data/stw_lists",
                        required=False, help="Folder path for stopwords")
    parser.add_argument("--nw", type=int, default=0,
                        required=False, help="Number of workers to use with Dask")
    parser.add_argument("--lang", type=str, default="en",
                        required=False, help="Language of the text to be preprocessed (en/es)")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm",
                        required=False, help="Spacy model to be used for preprocessing")
    parser.add_argument('--no_ngrams', default=False, required=False,
                        action='store_true', help="Flag to disable ngrams detection")
    parser.add_argument('--do_embeddings', default=False, required=False,
                        action='store_true', help="Flag to acticate embeddings calculation")
    parser.add_argument("--embeddings_model", type=str,
                        default="all-mpnet-base-v2", required=False,
                        help="Model to be used for calculating the embeddings")
    parser.add_argument("--max_sequence_length", type=int, default=384,
                        required=False, help="Context of the model to be used for calculating the embeddings.")

    args = parser.parse_args()
    print(args)

    # Create logger object
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('nlpPipeline')

    if args.lang not in ['en', 'es']:
        logger.error(
            f"-- The language {args.lang} is not supported by the tool. Exiting... ")
        sys.exit()

    if pathlib.Path(args.source_path).exists():
        source_path = pathlib.Path(args.source_path)
    else:
        logger.error(
            f"-- Source path {args.source_path} does not exist. Exiting... ")
        sys.exit()

    destination_path = pathlib.Path(args.destination_path)

    # Read config file to get the id, title and abstract fields associated with the dataset under preprocessing
    with open('config.json') as f:
        field_mappings = json.load(f)

    if args.source in field_mappings:
        mapping = field_mappings[args.source]
        logger.info(f"Reading from {args.source}...")
        id_fld = mapping["id"]
        raw_text_fld = mapping["raw_text"]
        title_fld = mapping["title"]
    else:
        logger.error(f"Unknown source: {args.source}. Exiting...")
        sys.exit()

    readers = {
        "xlsx": lambda path: dd.from_pandas(pd.read_excel(path), npartitions=3).fillna(""),
        "parquet": lambda path: dd.read_parquet(path).fillna("")
    }

    # Get reader according to file format
    if args.source_type in readers:
        reader = readers[args.source_type]
        df = reader(source_path)
    else:
        logger.error(
            f"-- Unsupported source type: {args.source_type}. Exiting...")
        sys.exit()

    # Detect abstracts' language and filter out those that are not in the language specified in args.lang
    logger.info(f"-- Detecting language...")
    df = \
        df[df[raw_text_fld].apply(
            det,
            meta=('langue', 'str')) == args.lang]

    # Concatenate title + abstract/summary if title is given
    if title_fld != "":
        df["raw_text"] = \
            df[[title_fld, raw_text_fld]].apply(
                " ".join, axis=1, meta=('raw_text', 'str'))
    else:
        # Rename text field to raw_text
        df = df.rename(columns={raw_text_fld: 'raw_text'})
    
    # Keep only necessary columns
    corpus_df = df[[id_fld, 'raw_text']]

    # Filter out rows with no raw_text
    corpus_df = corpus_df.replace("nan", np.nan)
    corpus_df = corpus_df.dropna(subset=["raw_text"], how="any")

    # Check max length of raw_text column to pass to the Pipe class
    logger.info(f"-- Checking max length of column 'raw_text'...")
    max_len = max_column_length(corpus_df, 'raw_text')
    logger.info(f"-- Max length of column 'raw_text' is {max_len}.")

    # Get stopword lists
    stw_lsts = []
    for entry in pathlib.Path(args.stw_path).joinpath(args.lang).iterdir():
        # check if it is a file
        if entry.as_posix().endswith("txt"):
            stw_lsts.append(entry)

    # Create pipeline
    nlpPipeline = Pipe(stw_files=stw_lsts,
                       spaCy_model=args.spacy_model,
                       language=args.lang,
                       max_length=max_len,
                       logger=logger)

    logger.info(f'-- -- NLP preprocessing starts...')

    start_time = time.time()
    corpus_df = nlpPipeline.preproc(corpus_df, args.nw, args.no_ngrams)
    logger.info(
        f'-- -- NLP preprocessing finished in {(time.time() - start_time)}')

    # Calculate embeddings if flag is activated
    if args.do_embeddings:
        
        # Save new df in parquet file
        logger.info(
            f'-- -- Saving preprocessed data without embeddings in {destination_path.as_posix()}...')
        save_parquet(outFile=destination_path, df=corpus_df, nw=args.nw)
        
        logger.info(f'-- -- Embeddings calculation starts...')
        start_time = time.time()
        em = EmbeddingsManager(logger=logger)
        corpus_df = em.generate_embeddings(corpus_df=corpus_df,
                                           embeddings_model=args.embeddings_model,
                                           max_seq_length=args.max_sequence_length,
                                           nw=args.nw)
        logger.info(
            f'-- -- Embeddings calculation finished in {(time.time() - start_time)}')
        
        if destination_path.as_posix().endswith("parquet"):
            bare_name = destination_path.as_posix().split(".parquet")[0]
            destination_path = pathlib.Path(bare_name + "_embeddings.parquet")
        else:
            destination_path = destination_path.joinpath("_embeddings")

    # Save new df in parquet file
    logger.info(
        f'-- -- Saving final preprocessed data in {destination_path.as_posix()}...')
    save_parquet(outFile=destination_path, df=corpus_df, nw=args.nw)

    return


# ############
# Execute main
if __name__ == '__main__':

    cprint(figlet_format("NLPipe",
           font='big'), 'blue', attrs=['bold'])
    print('\n')

    main()
