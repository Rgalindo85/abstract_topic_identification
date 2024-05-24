import os
import hydra
import numpy as np
import matplotlib.pyplot as plt

import pyLDAvis


from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import pyLDAvis.lda_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD

DIR_PATH = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    
    # load data
    filename = os.path.join(DIR_PATH, config.data.processed.path, 'data.json')
    print(f"Load data from {filename}")

    # use omegaconf to load file
    dict_data = OmegaConf.load(filename)

    # vectorization
    vectorizer, tfidf_matrix = vectorization(dict_data)
    
    # dim reduction visualization
    X_tsne = show_dim_reduction(tfidf_matrix, vectorizer, method='tnse')
    X_pca = show_dim_reduction(tfidf_matrix, vectorizer, method='pca')
    X_umap = show_dim_reduction(tfidf_matrix, vectorizer, method='umap')

    # Apply Hierarchical Clustering - HDBSCAN
    clus_tsne = apply_hierarchical_clustering(X_tsne, method='hdbscan')
    clus_pca = apply_hierarchical_clustering(X_pca, method='hdbscan')
    clus_umap = apply_hierarchical_clustering(X_umap, method='hdbscan')

    print('TSNE + HDSCAN clusters: ', np.unique(clus_tsne.labels_, return_counts=True))
    print('PCA + HDSCAN clusters: ', np.unique(clus_pca.labels_, return_counts=True))
    print('UMAP + HDSCAN clusters: ', np.unique(clus_umap.labels_, return_counts=True))


    model_lda = find_topics(tfidf_matrix, algo='LDA', n_topics=10)
    model_svd = find_topics(tfidf_matrix, algo='SVD', n_topics=10)

    # display topics
    tfidf_feature_names = vectorizer.get_feature_names_out()
    print("\nTopics from LDA:")
    display_topics(model_lda, tfidf_feature_names, 10)
    print("\nTopics from TruncatedSVD:")
    display_topics(model_svd, tfidf_feature_names, 10)

    # evaluate model

    

    # evaluate model
    # pyLDAvis.enable_notebook()
    # panel = pyLDAvis.lda_model.prepare(lda, tfidf_matrix, vectorizer, mds='tsne')
    # pyLDAvis.save_html(panel, os.path.join('reports', 'HTML','lda.html'))
    # pyLDAvis.display(panel)

def apply_hierarchical_clustering(X, method='hdbscan'):

    if method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
        clusterer.fit(X)
        plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_, cmap='viridis', alpha=0.3)
        plt.legend()
        plt.show()

        return clusterer
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.3, min_samples=10)
        clusterer.fit(X)
        plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_, cmap='viridis', alpha=0.3)
        plt.show()

        return clusterer
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