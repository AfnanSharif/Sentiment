import streamlit as st
import joblib
import yaml

st.set_page_config(page_title="Sentiment | Topic Modeling", layout="wide")

@st.cache_resource
def load_model():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config['data']['model_path'])
    return pipeline

try:
    pipeline = load_model()
except Exception as e:
    st.error("Model not found. Please run `src/train.py` first.")
    st.stop()

st.title("🗣️ Sentiment: Topic Modeling Analyzer")
st.markdown("Analyze customer reviews to automatically discover underlying topics using NLP (NMF).")

text_input = st.text_area("Enter a customer review:", value="The product is amazing but the delivery was late.")

if st.button("Analyze Topic", type="primary"):
    if text_input:
        # Predict topic distribution
        topic_dist = pipeline.transform([text_input])[0]
        dominant_topic = topic_dist.argmax()
        
        st.success(f"**Dominant Topic ID:** {dominant_topic}")
        
        st.subheader("Topic Distribution")
        st.bar_chart(topic_dist)
        
        # Display top words for the dominant topic
        tfidf_feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
        nmf_model = pipeline.named_steps['nmf']
        topic = nmf_model.components_[dominant_topic]
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [tfidf_feature_names[i] for i in top_features_ind]
        
        st.info(f"**Keywords for Topic {dominant_topic}:** {', '.join(top_features)}")
    else:
        st.warning("Please enter some text.")
