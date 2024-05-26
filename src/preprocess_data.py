import os
import hydra
import logging

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

DIR_PATH = Path(__file__).resolve().parents[1]

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    dict_data = load_data(config)


def load_data(config: DictConfig) -> dict:
    logger = logging.getLogger('load_data')

    filename = os.path.join(DIR_PATH, config.data.processed.path, 'data.json')
    logger.info(f"Load data from {filename}")

    # use omegaconf to load file
    dict_data = OmegaConf.load(filename)
    return dict_data

if __name__ == '__main__':
    main()