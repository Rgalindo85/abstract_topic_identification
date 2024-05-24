import os
import hydra

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

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
    svd = TruncatedSVD(n_components=10, random_state=42)
    svd.fit_transform(tfidf_matrix)

    feature_scores = dict(zip(vectorizer.get_feature_names_out(), svd.components_[0]))

    topic_out = sorted(
        feature_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(topic_out)


    # DBSCAN
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    dbscan.fit(tfidf_matrix)

    labels = dbscan.labels_
    #print(labels)    

    # LDA
    n_topics = 10
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online', n_jobs=-1)
    lda.fit(tfidf_matrix)

    # display topics
    tfidf_feature_names = vectorizer.get_feature_names_out()
    display_topics(lda, tfidf_feature_names, 10)

    # evaluate model
        

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