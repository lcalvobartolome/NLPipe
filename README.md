# NLPipe

[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Nemesis1303/NLPipe/blob/main/LICENSE)

This GitHub repository hosts an NLP pipeline based on Spacy and Gensim for topic modeling preprocessing in English and Spanish, utilizing Dask for distributed computing. The repository contains the necessary code and files for the pipeline implementation.

## Installation

To install this project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create and activate a virtual environment.
4. Install the required dependencies using `pip install -r requirements.txt`.

## Usage

To use this project, follow these steps:

1. Add your dataset's information about the ID, raw text and title names in the `config.json`, as follows:

   ```json
    "my_dataset": {
        "id": "id_field_name",
        "raw_text": "raw_text_field_name",
        "title": "title_field_name"
    }
    ```

2. Run the main script using the following command:

    ```bash
    python nlpipe.py [--source_path SOURCE_PATH] [--source_type SOURCE_TYPE] [--source SOURCE] [--destination_path DESTINATION_PATH] [--stw_path STW_PATH] [--nw NW] [--lang LANG] [--spacy_model SPACY_MODEL] [--no_ngrams NO_NGRAMS] [--do_embeddings DO_EMBEDDINGS] [--embeddings_model EMBEDDINGS_MODEL] [--max_sequence_length MAX_SEQUENCE]
    ```

    where:
    * `--source_path` is the path to the source data.
    * `--source_type` is the file format of the source data. The default value is parquet.
    * `--source` is the name of the dataset to be preprocessed (e.g., cordis, scholar, etc.).
    * `--destination_path` is the path to save the preprocessed data.
    * `--stw_path` is the folder path for stopwords. The default value is `data/stw_lists`. There you can find specific stopword lists in the languages supported by the tool.
    * `--nw` is the number of workers to use with Dask. The default value is `0`.
    * `--lang` is the language of the text to be preprocessed. At the time being, only English (`en`) and Spanish (`es`) are supported. The default value is `en`.
    * `--spacy_model`is the Spacy model to be used for the preprocessing. The default value is `"en_core_web_md"`.
    * `--no_ngrams` is a flag to disable n-grams dectection.
    * `--do_embeddings` is a flag to activate the calculation of the embeddings for the raw text.
    * `--embeddings_model` is the Transformer model to be used for the calculation of the embeddings.
    * `--max_sequence_length` is the context of the model to be used for calculating the embeddings.

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
