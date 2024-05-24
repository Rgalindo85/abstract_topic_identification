import os
import glob
import logging
import hydra
import xml.etree.ElementTree as ET
import nltk
import re

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

DIR_PATH = Path(__file__).resolve().parents[1]
# include into stopwords characteres that are typically found in xml files
stopwords = set(stopwords.words('english') + ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'] + ['lt', 'br', 'gt'])


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    logger = logging.getLogger('main')

    logger.info("Get abtract from xml files")
    dict_data = get_data(config)

    logger.info("Preprocess data")
    dict_data = preprocess_data(dict_data)

    # save dict_data as yaml
    save_path = os.path.join(DIR_PATH, config.data.processed.path)
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'data.json')

    logger.info(f"Save data to {save_file}")    
    with open(save_file, 'w') as f:
        f.write(OmegaConf.to_yaml(dict_data))
    
    logger.info("Done!")


def preprocess_data(data: dict) -> dict:
    
    dict_processed = {}
    for key in tqdm(data.keys()):
        text = data[key]
        if text is None or text == '':
            continue
        if len(text) == 0:
            continue
        dict_processed[key] = preprocess_text(data[key])
    return dict_processed

def preprocess_text(text: str) -> str:
    # Remove special characters typically found in xml files

    text = re.sub(r'&lt;', '<', text)

    # Remove non-alphabetic characters and tokenize
    tokens = word_tokenize(re.sub(r'[^a-zA-Z]', ' ', text.lower()))
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return ' '.join(processed)


def get_data(config: DictConfig) -> dict:
    """Get data from xml files

    Args:
        config (DictConfig): configuration file

    Returns:
        dict: dictionary with paper id as key and abstract as value
    """
    logger = logging.getLogger('get_data')

    # Get list of files
    data_path = os.path.join(DIR_PATH, config.data.raw.path, '*.xml')
    list_of_files = glob.glob(data_path)

    logger.info(f"Found {len(list_of_files)} files")
    
    # Get abstracts
    dict_data = {}
    for file in tqdm(list_of_files[:1000]):
        try:
            abstract = get_abstract(file)
            paper = file.split('/')[-1].split('.')[0]
            dict_data[paper] = abstract
        except Exception as e:
            logger.error(f"Error: {e}")
            continue

    return dict_data


def get_abstract(filename):
    """Read file and extract abstract

    Args:
        filename (str): filepath to the xml file

    Returns:
        _type_: _description_
    """
    xml_files = open(filename, 'r').read()

    root = ET.fromstring(xml_files)

    # Iterate through elements
    abstract = ''
    for child in root:
        for subchild in child:
            if 'abstract' in str(subchild.tag).lower():
                abstract = subchild.text
    return abstract
    


if __name__ == "__main__":
    main()