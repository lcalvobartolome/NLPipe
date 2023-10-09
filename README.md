# NLPipe

[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Nemesis1303/NLPipe/blob/main/LICENSE)

This GitHub repository hosts an NLP pipeline based on Spacy and Gensim for topic modeling preprocessing in English and Spanish and calculation of Transformer-based embeddings. The repository contains the necessary code and files for the pipeline implementation.

## Installation

To install this project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create and activate a virtual environment.
4. Install the required dependencies using `pip install -r requirements.txt`.

## Usage

To use this project, follow these steps:

1. Add your dataset's information in the `config.json`.  The library offers support for two distinct modes, depending on your specific requirements:

    * *Mode 1: Concatenation of `'title'` and `'raw_text'` columns*

        If your preferred approach involves lemmatizing or calculating embeddings based on the concatenation of a `'title'` and `'raw_text'` column (commonly used in tasks involving topic modeling of research papers), you should specify the following details for your dataset:

        ```json
        "my_dataset": {
            "id": "id_field_name",
            "raw_text": "raw_text_field_name",
            "title": "title_field_name"
        }
        ```

    * *Model 2: Mode 2: Independent Columns (N Columns)*

        If you intend to lemmatize or calculate embeddings for N independent columns (e.g., `'col1'`, `'col2'`, `'col3'`), these columns should be provided as a list within the 'raw_text' field, while leaving the `'title'` field empty

        ```json
        "my_dataset": {
            "id": "id_field_name",
            "raw_text": ["col1", "col2", "col3"],
            "title": ""
        }
        ```

2. Run the main script using the following command:

    ```bash
    python nlpipe.py [--source_path SOURCE_PATH] [--source_type SOURCE_TYPE] [--source SOURCE] [--destination_path DESTINATION_PATH] [--stw_path STW_PATH] [--lang LANG] [--spacy_model SPACY_MODEL] [--no_ngrams NO_NGRAMS] [--no_preproc NO_PREPROC] [--do_embeddings DO_EMBEDDINGS] [--embeddings_model EMBEDDINGS_MODEL] [--max_sequence_length MAX_SEQUENCE] [--use_dask USE_DASK] [--nw NW]
    ```

    where:
    * `--source_path`: Path to the source data.
    * `--source_type`: File format of the source data. The default value is parquet.
    * `--source`: Name of the dataset to be preprocessed (e.g., cordis, scholar, etc.).
    * `--destination_path`: Path to save the preprocessed data.
    * `--stw_path`: Folder path for stopwords. The default value is `data/stw_lists`. There you can find specific stopword lists in the languages supported by the tool.
    * `--lang`: Language of the text to be preprocessed. At the time being, only English (`en`) and Spanish (`es`) are supported. The default value is `en`.
    * `--spacy_model`: Spacy model to be used for the preprocessing. The default value is `"en_core_web_md"`.
    * `--no_ngrams`: Flag to disable n-gram detection. The default is False, meaning that n-gram detection will be carried out if not specified otherwise.
    * `--no_preproc`:  Flag to disable NLP preprocessing. The default is False, meaning that NLP preprocessing will be carried out if not specified otherwise. If the --do_embeddings flag is disabled, this flag must also be disabled.
    * `--do_embeddings`: Flag to activate the calculation of embeddings for raw text. The default is False, meaning that embeddings will only be calculated if this flag is set. If the --no_preproc flag is disabled, this flag must be set.
    * `--embeddings_model`: Transformer model to be used for the calculation of the embeddings.
    * `--max_sequence_length`: Context of the model to be used for calculating the embeddings.
    * `--use_dask`: Flag to activate Dask usage. By default, pandas is used.
    * `--nw`: Number of workers to use with Dask. The default value is `0`.

> *Note that you need to choose the Spacy model according to the language of the text to be preprocessed. For example, if the text is in `English`, you can choose one out of `en_core_web_sm` | `en_core_web_md` | `en_core_web_lg` | `en_core_web_trf`. In case the language of the text is Spanish, the following are available: `es_core_news_sm` | `es_core_news_md` | `es_core_news_lg` | `es_core_news_trf`. In general, if you have enough computational resources and need advanced text processing capabilities, `xx_core_xx_lg` or `xx_core_xx_trf` are the best choices. However, if you have limited resources or need to process text quickly, `xx_core_xx_sm` might be a better option. `xx_core_xx_md` provides a balance between the latter options.*
>> **If you are using transformer models, you still need to install spacy-transformers yourself!**

## Directory Structure

The repository is organized as follows:

```bash
NLPipe/
├── data/
│   ├── stw_lists/
│   │   ├── en/
│   │   │   ├── stopwords_atire_ncbi.txt
│   │   │   ├── stopwords_ebscohost_medline_cinahl.txt
│   │   │   ├── stopwords_ovid.txt
│   │   │   ├── stopwords_pubmed.txt
│   │   │   ├── stopwords_technical.txt
│   │   │   ├── stopwords_UPSTO.txt
│   │   ├── es/
│   │   │   ├── stw_academic.txt   
│   │   │   ├── stw_generic.txt
│   │   │   └── stw_science.txt
├── src/
│   ├── acronyms.py
│   ├── embeddings_manager.py
│   ├── pipe.py
│   └── utils.py
├── .devcontainer/
│   ├── devcontainer.json
├── nlpipe.py
├── README.md
├── requirements.txt
├── LICENSE
└── Dockerfile
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
