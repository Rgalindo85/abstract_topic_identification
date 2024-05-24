import os
import hydra
import numpy as np

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
    svd = TruncatedSVD(n_components=4, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)

    print("\nTopics from TruncatedSVD:")
    terms = vectorizer.get_feature_names_out()
    for i, component in enumerate(svd.components_):
        top_terms_indices = component.argsort()[:-11:-1]
        top_terms = [terms[index] for index in top_terms_indices]
        print(f"Topic {i+1}: {', '.join(top_terms)}")
        
    # LDA
    n_topics = 6
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online', n_jobs=-1)
    lda.fit(tfidf_matrix)

    # display topics
    tfidf_feature_names = vectorizer.get_feature_names_out()
    display_topics(lda, tfidf_feature_names, 10)

    # evaluate model
    # pyLDAvis.enable_notebook()
    panel = pyLDAvis.lda_model.prepare(lda, tfidf_matrix, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, os.path.join('reports', 'HTML','lda.html'))
    # pyLDAvis.display(panel)
        

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