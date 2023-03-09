import nltk
from langdetect import detect


def check_nltk_packages():
    packages = ['punkt','stopwords','omw-1.4','wordnet']

    for package in packages:
        try:
            nltk.data.find('tokenizers/' + package)
        except LookupError:
            nltk.download(package)
            
def det(x: str) -> str:
    """
    Detects the language of a given text
    """
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang