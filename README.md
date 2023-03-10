# NLPipe
This GitHub repository hosts an NLP pipeline based on Spacy and Gensim for topic modeling preprocessing, utilizing Dask for distributed computing. The repository contains the necessary code and files for the pipeline implementation.

## Installation

To install this project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create and activate a virtual environment.
4. Install the required dependencies using `pip install -r requirements.txt`.
5. Install the spacy model to be used for the preprocessing via `!python3 -m spacy download spacy_model` where `spacy_model` is one of `en_core_web_sm` | `en_core_web_md` | `en_core_web_lg`. 
In general, if you have enough computational resources and need advanced text processing capabilities, `en_core_web_lg` is the best choice. However, if you have limited resources or need to process text quickly, `en_core_web_sm` might be a better option. `en_core_web_md` provides a balance between the two.

## Usage
To use this project, follow these steps:

1. Add your dataset's information about the ID, raw text and title names in the `config.json`, as follows:

   ```
    "my_dataset": {
        "id": "id_field_name",
        "raw_text": "raw_text_field_name",
        "title": "title_field_name"
    }
    ```
2. Run the main script using the following command:

    ```
    python nlpipe.py [--source_path SOURCE_PATH] [--source_type SOURCE_TYPE] [--source SOURCE] [--destination_path DESTINATION_PATH] [--stw_path STW_PATH] [--nw NW]
    ```
    where 
    * `--source_path` is the path to the source data.
    * `--source_type` is the file format of the source data. The default value is parquet.
    * `--source` is the name of the dataset to be preprocessed (e.g., cordis, scholar, etc.).
    * `--destination_path` is the path to save the preprocessed data.
    * `--stw_path` is the folder path for stopwords. The default value is `data/stw_lists`.
    * `--nw` is the number of workers to use with Dask. The default value is 0.

## Directory Structure

The repository is organized as follows:

```bash
NLPipe/
├── data/
│   ├── stw_lists/
│   │   ├── stopwords_atire_ncbi.txt
│   │   ├── stopwords_ebscohost_medline_cinahl.txt
│   │   ├── stopwords_ovid.txt   
│   │   └── stopwords_pubmed.txt
├── src/
│   ├── acronyms.py
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
