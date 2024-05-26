import streamlit as st
import numpy as np

import hydra
import os
import logging
import joblib

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import matplotlib.pyplot as plt

DIR_PATH = Path(__file__).resolve().parents[0]

@hydra.main(config_path="config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    logger = logging.getLogger('main')

    st.title("Abstracts Topic Identification")
    st.header("Cluster and Topic Visualization")

    dict_data = load_data(config)
    model = load_model(config)

    st.subheader("Clustering Analysis")
    clustering_analsyis(dict_data, model, config)

    st.subheader("Topic Visualization")


def clustering_analsyis(dict_data: dict, model, config: DictConfig):
    logger = logging.getLogger('clustering_analysis')
    logger.info("Clustering Analysis")

    cols = st.columns(3)

    # show clustering results
    show_cluster_scatter(dict_data, cols[0])
    show_cluster_distribution(dict_data, cols[1])
    show_cluster_dendogram(model, cols[2])
    
    # plot clusters
    st.write("Plotting clusters")
    plot_clusters(dict_data, model, config)


def show_cluster_dendogram(model, col: st.columns):
    col.write("Show cluster dendogram")

    num_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)  # number of clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))  # Create a list of colors

    fig, ax = plt.subplots()
    model.condensed_tree_.plot(select_clusters=True, selection_palette=colors)
    col.pyplot(fig)


def show_cluster_distribution(dict_data: dict, col: st.columns):
    col.subheader("Cluster Distribution")

    # show top ten of most frequent clusters
    unique, counts = np.unique(dict_data['labels'], return_counts=True)

    fig, ax = plt.subplots()
    bar_plot = ax.bar(unique[:10], counts[:10])
    ax.set(
        xlabel='Clusters',
        ylabel='Frequency',
        title='Top 10 - Cluster Distribution'
    )
    
    # add annotation to bar plot
    for i, v in enumerate(counts[:10]):
        ax.text(i-1, v + 5, str(v), color='black', ha='center')

    col.pyplot(fig)


def show_cluster_scatter(dict_data: dict, col: st.columns):
    col.subheader("UMAP with HDBSCAN Clustering")
    col.write(
        "Number of clusters: {}".format(
            len(np.unique(dict_data['labels'])
            )
        )
    )

    fig, ax = plt.subplots()
    scatter = ax.scatter(dict_data['embeddings'][:, 0], dict_data['embeddings'][:, 1], c=dict_data['labels'], cmap='viridis', s=5, alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.set(
        xlabel='UMAP 1',
        ylabel='UMAP 2',
    )
    ax.add_artist(legend)
    col.pyplot(fig)



def plot_clusters(dict_data: dict, model, config: DictConfig):

    # plot clusters with a word cloud
    st.write("Plotting clusters with word cloud")
    from wordcloud import WordCloud
    from collections import Counter

    fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(20, 20))
    ax = ax.flatten()

    i = 0

    list_abstracts = list(dict_data['abstracts'].values())

    for cluster in np.unique(dict_data['labels'])[:10]:
        st.write(f"Cluster {cluster}")

        idx = np.where(dict_data['labels'] == cluster)[0]
        abstracts = [list_abstracts[i] for i in idx]

        text = ' '.join(abstracts)
        wordcloud = WordCloud(width=800, height=400).generate(text)

        ax[i].imshow(wordcloud, interpolation='bilinear')
        ax[i].axis('off')
        i = i + 1
        # plt.show()
    
    st.pyplot(fig)

    
    # for cluster in np.unique(dict_data['labels']
    #     st.write(f"Cluster {cluster}")
    #     idx = np.where(dict_data['labels'] == cluster)
    #     for i in idx:
    #         st.write(dict_data['abstracts'][i])



def load_model(config: DictConfig):
    logger = logging.getLogger('load_model')
    logger.info("Loading model")

    model_name = list(config.models.clustering[0].keys())[0]
    model = joblib.load(os.path.join(DIR_PATH, 'models', model_name, 'model.joblib'))

    return model


def load_data(config: DictConfig) -> dict:
    logger = logging.getLogger('load_data')
    logger.info("Loading data")

    dict_data = {
        'embeddings': np.load(os.path.join(DIR_PATH, config.data.model_input.path, 'data.npy')),
        'abstracts': OmegaConf.load(os.path.join(DIR_PATH, config.data.processed.path, 'data.json')),
        'labels': np.load(os.path.join(DIR_PATH, config.data.model_output.path, 'hdbscan', 'labels.npy'))
    }

    return dict_data


if __name__ == '__main__':
    # configure streamlit dashboard
    st.set_page_config(layout="wide")
    main()