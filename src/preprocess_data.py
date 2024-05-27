import os
import hydra
import logging
import umap

import numpy as np

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DIR_PATH = Path(__file__).resolve().parents[1]

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    logger = logging.getLogger('main')
    logger.info("Preprocess data")

    dict_data = load_data(config)
    processed_data = preprocess_data(dict_data)

    # save processed data
    output_file = os.path.join(DIR_PATH, config.data.model_input.path)
    os.makedirs(os.path.join(DIR_PATH, config.data.model_input.path), exist_ok=True)
    print(output_file)
    np.save(os.path.join(output_file, 'data.npy'), processed_data)

    logger.info("Done!")

def preprocess_data(data: dict) -> dict:
    logger = logging.getLogger('preprocess_data')
    logger.info("Preprocess data")

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data.values())

    # reduce dimensionality
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(tfidf_matrix.toarray())

    return embedding



def load_data(config: DictConfig) -> dict:
    logger = logging.getLogger('load_data')

    filename = os.path.join(DIR_PATH, config.data.processed.path, 'data.json')
    logger.info(f"Load data from {filename}")

    # use omegaconf to load file
    dict_data = OmegaConf.load(filename)
    return dict_data

if __name__ == '__main__':
    main()