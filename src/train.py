import pandas as pd
import yaml
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline

def train_model():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    df = pd.read_csv(config['data']['raw_path'])
    texts = df['review']
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')),
        ('nmf', NMF(n_components=config['model']['n_topics'], random_state=config['model']['random_state']))
    ])
    
    print("Fitting NLP pipeline (TF-IDF + NMF)...")
    pipeline.fit(texts)
    
    # Save the pipeline
    model_path = config['data']['model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Display top words per topic
    tfidf_feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    nmf_model = pipeline.named_steps['nmf']
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [tfidf_feature_names[i] for i in top_features_ind]
        print(f"Topic {topic_idx}: {' '.join(top_features)}")

if __name__ == "__main__":
    train_model()
