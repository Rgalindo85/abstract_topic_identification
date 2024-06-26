import os
import hydra
import logging
import joblib
import hdbscan

import numpy as np

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from sklearn.cluster import DBSCAN

DIR_PATH = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    logger = logging.getLogger('main')

    # load data
    data = load_data(config)

    # apply hierarchical clustering

    list_models = config.models.clustering
    for m in list_models:
        model_name = list(m.keys())[0]

        logger.info(f"Apply hierarchical clustering with {model_name}")
        model, labels = apply_hierarchical_clustering(data, model=m, model_name=model_name)

        model_path = os.path.join(DIR_PATH, 'models', model_name)
        model_pred_path = os.path.join(DIR_PATH, config.data.model_output.path, model_name)

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(model_pred_path, exist_ok=True)

        model_file = os.path.join(model_path, 'model.joblib')
        labels_file = os.path.join(model_pred_path, 'labels.npy')

        # save model
        logger.info(f"Save model to {model_file}")
        joblib.dump(model, model_file)
        
        # save labels
        logger.info(f"Save labels to {labels_file}")
        np.save(labels_file, labels)

    logger.info("Done!")

def load_data(config: DictConfig) -> dict:
    logger = logging.getLogger('load_data')
    
    filename = os.path.join(DIR_PATH, config.data.model_input.path, 'data.npy')
    
    logger.info(f"Load data from {filename}")
    
    data = np.load(filename)

    return data


def apply_hierarchical_clustering(X, model=None, model_name='hdbscan'):
    logger = logging.getLogger('apply_hierarchical_clustering')

    if model is None:
        logger.info("No model provided.")
        return

    method = model_name
    params = model[model_name].params

    if method == 'hdbscan':
        logger.info("Apply HDBSCAN")

        min_cluster_size = params.min_cluster_size
        min_samples = params.min_samples
        cluster_selection_epsilon = params.cluster_selection_epsilon
        metric = params.metric

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            gen_min_span_tree=True, 
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric
        )
        labels = clusterer.fit_predict(X)

        # plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_, cmap='viridis', alpha=0.3)
        # plt.legend()
        # plt.show()

        return clusterer, labels
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.3, min_samples=10)
        labels = clusterer.fit_predict(X)

        # plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_, cmap='viridis', alpha=0.3)
        # plt.show()

        return clusterer, labels
    else:
        raise ValueError(f"Method {method} not supported")


def show_dim_reduction(tfidf_matrix, vectorizer, method='tnse'):
    if method == 'tnse':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=300, n_jobs=-1)
        X_tsne = tsne.fit_transform(tfidf_matrix.toarray())

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.3)
        plt.show()

        return X_tsne

    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(tfidf_matrix.toarray())

        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
        plt.show()

        return X_pca
    elif method == 'umap':
        import umap
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(tfidf_matrix.toarray())
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.3)
        plt.show()

        return embedding

    else:
        raise ValueError(f"Method {method} not supported")


def find_topics(tfidf_matrix, algo='LDA', n_topics=3):
    if algo == 'LDA':
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online', n_jobs=-1)
        model.fit(tfidf_matrix)

        # plot components
        print(model.exp_dirichlet_component_)
        print(model.components_.shape)
        plt.scatter(model.components_[0], model.components_[1])
        plt.show()

    elif algo == 'SVD':
        model = TruncatedSVD(n_components=n_topics, random_state=42)
        model.fit(tfidf_matrix)

        # plot components
        print(model.explained_variance_ratio_)
        print(model.components_.shape)
        plt.scatter(model.components_[0], model.components_[1])
        plt.show()
    else:
        raise ValueError(f"Algorithm {algo} not supported")

    model.fit(tfidf_matrix)
    return model


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def vectorization(dict_data):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dict_data.values())

    return vectorizer, tfidf_matrix

if __name__ == "__main__":
    main()