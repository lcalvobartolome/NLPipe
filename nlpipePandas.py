import argparse
import json
import logging
import pathlib
import sys
import time

# import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyfiglet import figlet_format
from termcolor import cprint

# from src.embeddings_manager import EmbeddingsManager
from src.pipePandas import Pipe
from src.utils import det #, max_column_length, save_parquet

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
    parser.add_argument("--lang", type=str, default="en",
                        required=False, help="Language of the text to be preprocessed (en/es)")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm",
                        required=False, help="Spacy model to be used for preprocessing")
    parser.add_argument('--no_ngrams', default=False, required=False,
                        action='store_true', help="Flag to disable ngrams detection")

    args = parser.parse_args()

    # Create logger object
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('nlpPipeline')
    
    # Check that the language is valid
    if args.lang not in ['en', 'es']:
        logger.error(
            f"-- The language {args.lang} is not supported by the tool. Exiting... ")
        sys.exit()

    # Check that the source file exists
    if pathlib.Path(args.source_path).exists():
        source_path = pathlib.Path(args.source_path)
    else:
        logger.error(
            f"-- Source path {args.source_path} does not exist. Exiting... ")
        sys.exit()

    destination_path = pathlib.Path(args.destination_path)

    if True: #Hay que desindentar lo siguiente
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

        df = pd.read_parquet(source_path, columns=[id_fld, title_fld, raw_text_fld]).fillna("") #.sample(frac=.01)
        
        print(len(df))
        start_time = time.time()
        # Detect abstracts' language and filter out those that are not in the language specified in args.lang
        logger.info(f"-- Detecting language...")
        df = df[df[raw_text_fld].apply(det) == args.lang]
        logger.info(
            f'-- -- Language detection finished in {(time.time() - start_time)}')
        print(len(df))

         # Concatenate title + abstract/summary if title is given
        df["raw_text"] = df[title_fld] + " " + df[raw_text_fld]

        # Keep only necessary columns
        corpus_df = df[[id_fld, 'raw_text']]

        # Filter out rows with no raw_text
        corpus_df = corpus_df.replace("nan", np.nan)
        corpus_df = corpus_df.dropna(subset=["raw_text"], how="any")

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
                           #max_length=max_len,
                           logger=logger)

        logger.info(f'-- -- NLP preprocessing starts...')

        start_time = time.time()
        corpus_df = nlpPipeline.preproc(corpus_df, args.no_ngrams)
        logger.info(
            f'-- -- NLP preprocessing finished in {(time.time() - start_time)}')

        # Save new df in parquet file
        logger.info(
            f'-- -- Saving preprocessed data without embeddings in {destination_path.as_posix()}...')
        corpus_df.to_parquet(destination_path, index=False)

    return


# ############
# Execute main
if __name__ == '__main__':

    cprint(figlet_format("NLPipe",
           font='big'), 'blue', attrs=['bold'])
    print('\n')

    main()
