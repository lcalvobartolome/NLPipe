import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import List

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingsManager(object):
    """Class to manage embeddings generation"""

    def __init__(self,
                 logger=None):
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        """

        # Create logger object
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('EmbeddingsManager')

    def _check_max_local_length(self,
                                max_seq_length: int,
                                texts: List[str]) -> None:
        """
        Checks if the longest document in the collection is longer than the maximum sequence length allowed by the model

        Parameters
        ----------
        max_seq_length: int
            Context of the transformer model used for the embeddings generation
        texts: list[str]
            The sentences to embed
        """

        max_local_length = np.max([len(t.split()) for t in texts])
        if max_local_length > max_seq_length:
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                          f"truncates to {max_seq_length} tokens.")
        return

    def bert_embeddings_from_list(self,
                                  texts: List[str],
                                  sbert_model_to_load: str,
                                  batch_size=32,
                                  max_seq_length=None) -> np.ndarray:
        """
        Creates SBERT Embeddings from a list

        Parameters
        ----------
        texts : list[str]
            The sentences to embed
        sbert_model_to_load: str
            Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
        batch_size: int (default=32)
            The batch size used for the computation
        max_seq_length: int
            Context of the transformer model used for the embeddings generation

        Returns
        -------
        embeddings: list
            List with the embeddings for each document
        """

        model = SentenceTransformer(sbert_model_to_load)

        if max_seq_length is not None:
            model.max_seq_length = max_seq_length

        self._check_max_local_length(max_seq_length, texts)
        embeddings = model.encode(
            texts, show_progress_bar=True, batch_size=batch_size).tolist()

        return embeddings

    def add_embeddins_to_parquet(self,
                                 parquet_file: Path,
                                 parquet_new: Path,
                                 embeddins_model: str,
                                 max_seq_length: int) -> Path:
        """Generates the embeddings for a set of files given in parquet format, and saves them in a new parquet file that containing the original data plus an additional column named 'embeddings'.

        Parameters
        ----------
        parquet_file : Path
            Path from which the source parquet files are read
        parquet_new: Path
            Path in which the new parquet files will be saved
        embeddins_model: str
            Model to be used for generating the embeddings
        max_seq_length: int
            Context of the transformer model used for the embeddings generation

        Returns
        -------
        parquet_new: Path
            Path in which the new parquet files have been saved
        """

        path_parquet = Path(parquet_file)

        res = []
        if path_parquet.is_file():
            res.append(path_parquet)
        elif path_parquet.is_dir():
            for entry in path_parquet.iterdir():
                # check if it is a file
                if entry.as_posix().endswith("parquet"):
                    res.append(entry)

        process_entries = []
        for entry in parquet_new.iterdir():
            # check if it is a file
            if entry.as_posix().endswith("parquet"):
                s = entry.as_posix().split("/")[7].split("_")[3].split(".")[0]
                process_entries.append(int(s))

        self._logger.info(
            f"-- -- Number of parquet to process: {str(len(res))}")

        for i, f in enumerate(tqdm(res)):
            if i not in process_entries:
                self._logger.info(f"-- -- Processing now: {str(i)}")
                df = pd.read_parquet(f)  # .fillna("")

                raw = df['raw_text'].values.tolist()
                embeddings = self.bert_embeddings_from_list(
                    texts=raw,
                    sbert_model_to_load=embeddins_model,
                    max_seq_length=max_seq_length)
                df['embeddings'] = embeddings

                # Save new df in parquet file
                new_name = "parquet_embeddings_part_" + str(i) + ".parquet"
                outFile = parquet_new.joinpath(new_name)
                df.to_parquet(outFile)

        return

    def generate_embeddings(self,
                            corpus_df: dd.DataFrame,
                            embeddings_model: str,
                            max_seq_length: int,
                            nw: int) -> Path:
        """Generates the embeddings for the field 'raw_text' associated to a corpus given as dask dataframe and saves such the obtained embeddings in the latter.

        Parameters
        ----------
        corpus_df : dd.DataFrame
            Dataframe containing the corpus whose embeddings are going to be generated
        embeddins_model: str
            Model to be used for generating the embeddings
        max_seq_length: int
            Context of the transformer model used for the embeddings generation

        Returns
        -------
        corpus_df: dd.DataFrame
            Original dataframe with the new embeddings column
        """

        self._logger.info(f"-- -- Computing raw text from corpus...")
        # Get raw text from corpus datafrmae
        with ProgressBar():
            if nw > 0:
                raw = corpus_df["raw_text"].compute(
                    scheduler='processes', num_workers=nw)
            else:
                # Use Dask default number of workers (i.e., number of cores)
                raw = corpus_df["raw_text"].compute(
                    scheduler='processes')

        self._logger.info(f"-- -- Calculating embeddings...")
        # Calculate embeddings from raw
        embeddings = self.bert_embeddings_from_list(
            texts=raw.values.tolist(),
            sbert_model_to_load=embeddings_model,
            max_seq_length=max_seq_length)

        # Convert to string
        embeddings = [str(emb) for emb in embeddings]

        def get_embeddings(doc):
            return embeddings[doc]

        self._logger.info(f"-- -- Saving embeddings in dataframe...")
        # Work out the lengths of each partition
        chunks = corpus_df.map_partitions(
            lambda x: len(x)).compute().to_numpy()

        # Build a Dask array with the same partition sizes
        emdarray = da.from_array(embeddings, chunks=tuple(chunks))

        # Assign the list of embeddings to the dataframe
        corpus_df['embeddings'] = emdarray

        return corpus_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scripts for Embeddings Service")
    parser.add_argument("--path_parquet", type=str, default=None,
                        required=True, metavar=("path_to_parquet"),
                        help="path to parquet file to caclulate embeddings of")
    parser.add_argument("--path_new", type=str, default=None,
                        required=True, metavar=("path_new"),
                        help="path to parquet folder to locate embeddings")
    parser.add_argument("--embeddings_model", type=str,
                        default="all-mpnet-base-v2", required=False,
                        metavar=("embeddings_model"),
                        help="Model to be used for calculating the embeddings")
    parser.add_argument("--max_sequence_length", type=int, default=384,
                        required=False, metavar=("max_sequence_length"), help="Model's context")

    args = parser.parse_args()

    # Create logger object
    logging.basicConfig(level='INFO')
    logger = logging.getLogger('EmbeddingsManager')

    em = EmbeddingsManager(logger=logger)

    parquet_path = Path(args.path_parquet)
    parquet_new = Path(args.path_new)
    em.add_embeddins_to_parquet(
        parquet_path, parquet_new, args.embeddings_model, args.max_sequence_length)
