import os
import hydra
import logging

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

DIR_PATH = Path(__file__).resolve().parents[1]

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    pass

if __name__ == '__main__':
    main()