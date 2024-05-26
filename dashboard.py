import streamlit as st
import xml.etree.ElementTree as ET
import hydra
import os
import glob

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

DIR_PATH = Path(__file__).resolve().parents[0]

@hydra.main(config_path="config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    st.title("Abstracts Topic Identification")

    # load data
    data_path = os.path.join(DIR_PATH, config.data.raw.path)
    list_files = glob.glob(data_path + '/*.xml')

    print(data_path)
    filename = list_files[0]
    tree = ET.parse(filename)
    root = tree.getroot()

    # show xml file in the dashboard
    st.write(root)


if __name__ == '__main__':
    main()