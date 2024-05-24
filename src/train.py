import os
import hydra
import json

from omegaconf import DictConfig

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation 

@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(config: DictConfig):
    
    # load data
    filename = os.path.join(config.data.processed.path, 'data.json')
    dict_data = json.load(open(filename))

    vectorizer, tfidf_matrix = vectorization(dict_data)

    n_topics = 10
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    tfidf_feature_names = vectorizer.get_feature_names_out()
    display_topics(lda, tfidf_feature_names, 10)


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