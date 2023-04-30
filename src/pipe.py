import logging
import pathlib
import re
from typing import List
from spacy_download import load_spacy

import contractions
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from gensim.models.phrases import Phrases

import src.acronyms as acronyms


class Pipe():
    """
    Class to carry out text preprocessing tasks that are needed by topic modeling
    - Basic stopword removal
    - Acronyms substitution
    - NLP preprocessing
    - Ngrams detection
    """

    def __init__(self,
                 stw_files: List[pathlib.Path],
                 spaCy_model: str,
                 language: str,
                 max_length: int,
                 logger=None):
        """
        Initilization Method
        Stopwords files will be loaded during initialization

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        spaCy_model: str
            Name of the spaCy model to be used for preprocessing
        language: str
            Language of the text to be preprocessed (en/es)
        max_length: int
            Maximum length of the text to be processed
        logger: Logger object
            To log object activity
        """

        # Create logger object
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('nlpPipeline')

        # Load stopwords and acronyms
        self._loadSTW(stw_files)
        self._loadACR(language)

        # Download spaCy model if not already downloaded and load
        self._nlp = load_spacy(spaCy_model, exclude=['parser', 'ner'])
        self._nlp.max_length = max_length + round(0.1 * max_length)

        return

    def _loadSTW(self, stw_files: List[pathlib.Path]) -> None:
        """
        Loads stopwords as list from files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        """

        stw_list = \
            [pd.read_csv(stw_file, names=['stopwords'], header=None,
                         skiprows=3) for stw_file in stw_files]
        stw_list = \
            [stopword for stw_df in stw_list for stopword in stw_df['stopwords']]
        self._stw_list = list(dict.fromkeys(stw_list))  # remove duplicates
        self._logger.info(
            f"Stopwords list created with {len(stw_list)} items.")

        return

    def _loadACR(self, lang: str) -> None:
        """
        Loads list of acronyms
        """

        self._acr_list = acronyms.en_acronyms_list if lang == 'en' else acronyms.es_acronyms_list

        return

    def _replace(self, text, patterns) -> str:
        """
        Replaces patterns in strings.

        Parameters
        ----------
        text: str
            Text in which the patterns are going to be replaced
        patterns: List of tuples
            Replacement to be carried out

        Returns
        -------
        text: str
            Replaced text
        """

        for (raw, rep) in patterns:
            regex = re.compile(raw, flags=re.IGNORECASE)
            text = regex.sub(rep, text)
        return text

    def do_pipeline(self, rawtext) -> str:
        """
        Implements the preprocessing pipeline, by carrying out:
        - Lemmatization according to POS
        - Removal of non-alphanumerical tokens
        - Removal of basic English stopwords and additional ones provided       
          within stw_files
        - Acronyms replacement
        - Expansion of English contractions
        - Word tokenization
        - Lowercase conversion

        Parameters
        ----------
        rawtext: str
            Text to preprocess

        Returns
        -------
        final_tokenized: List[str]
            List of tokens (strings) with the preprocessed text
        """

        # Change acronyms by their meaning
        text = self._replace(rawtext, self._acr_list)

        # Expand contractions
        try:
            text = contractions.fix(text)
        except:
            text = text  # this is only for SS

        valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])

        doc = self._nlp(text)
        lemmatized = [token.lemma_ for token in doc
                      if token.is_alpha
                      and token.pos_ in valid_POS
                      and not token.is_stop
                      and token.lemma_ not in self._stw_list]

        # Convert to lowercase
        final_tokenized = [token.lower() for token in lemmatized]

        return final_tokenized

    def preproc(self, corpus_df: dd.DataFrame) -> dd.DataFrame:
        """
        Invokes NLP pipeline and carries out, in addition, n-gram detection.

        Parameters
        ----------
        corpus_df: dd.DataFrame
            Dataframe representation of the corpus to be preprocessed. 
            It needs to contain (at least) the following columns:
            - raw_text

        Returns
        -------
        corpus_df: dd.DataFrame
            Preprocessed DataFrame
            It needs to contain (at least) the following columns:
            - raw_text
            - lemmas_with_grams
        """

        # Lemmatize text
        self._logger.info("-- Lemmatizing text")
        lemmas = corpus_df["raw_text"].apply(self.do_pipeline,
                                             meta=('lemmas', 'object'))
        # this does the same but with batch preprocessing
        # def apply_pipeline_to_partition(partition):
        #    return partition['raw_text'].apply(self.do_pipeline)
        # lemmas = corpus_df.map_partitions(apply_pipeline_to_partition, meta=('lemmas', 'object'))

        # Create corpus from tokenized lemmas
        self._logger.info(
            "-- Creating corpus from lemmas for n-grams detection")
        with ProgressBar():
            lemmas = lemmas.compute(scheduler='processes')
        corpus = lemmas.values.tolist()

        # Create Phrase model for n-grams detection
        self._logger.info("-- Creating Phrase model")
        phrase_model = Phrases(corpus, min_count=2, threshold=20)

        # Carry out n-grams substitution
        self._logger.info("-- Carrying out n-grams substitution")
        corpus = (phrase_model[doc] for doc in corpus)
        corpus = list((" ".join(doc) for doc in corpus))

        # Save n-grams in new column in the dataFrame
        self._logger.info(
            "-- Saving n-grams in new column in the dataFrame")

        def get_ngram(row):
            return corpus.pop(0)
        corpus_df["lemmas_with_grams"] = corpus_df.apply(
            get_ngram, meta=('lemmas_with_grams', 'object'), axis=1)
        # corpus_df["lemmas_with_grams"] =  corpus_df.apply(lambda row: corpus[row.name], meta=('lemmas_with_grams', 'object'), axis=1)

        return corpus_df
