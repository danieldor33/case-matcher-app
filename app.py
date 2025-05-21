import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Define your static list of How-To articles
HOW_TO_ARTICLES = [
    "Reset Salesforce password",
    "Create a new Salesforce report",
    "Update Opportunity stage",
    "Add a new user to Salesforce",
    "Delete a Salesforce contact",
    "Fix duplicate leads",
    "Assign leads to users automatically",
    "Export opportunities to Excel",
    "Create dashboards for sales team",
    "Track stage duration in pipeline",
    "Log calls in Salesforce",
    "Send mass emails using templates",
    "Schedule reports for weekly delivery",
    "Create and manage campaigns",
    "Reassign accounts when users leave",
    "Integrate Salesforce with Gmail",
    "Enable two-factor authentication",
    "Customize page layouts",
    "Manage record types",
    "Archive old reports"
]

# Precompute embeddings
@st.cache_resource
def compute_embeddings(articles):
    return model.encode(articles, convert_to_tensor=True)

ARTICLE_EMBEDDINGS = compute_embeddings(HOW_TO_ARTICLES)

# UI layout
st.set_page_config(page_title="Case-to-How-To Matcher", layout="centered")
st.title("🔍 Sales Ops Help Matcher")

st.markdown("""
Enter your question or case description below, and we'll suggest the most relevant "How-To" article from our knowledge base.
""")

user_query = st.text_input("Enter your case/question")

if st.button("Find Matching Article"):
    if not user_query.strip():
        st.warning("Please enter a valid question or case description.")
    else:
        with st.spinner("Finding best match..."):
            query_embedding = model.encode(user_query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, ARTICLE_EMBEDDINGS)[0]
            best_idx = torch.argmax(scores).item()
            top_score = scores[best_idx].item()
            best_match = HOW_TO_ARTICLES[best_idx]

            st.success(f"✅ Best Match: **{best_match}**")
            st.caption(f"Similarity Score: {top_score:.4f}")

            # Optional: Show top 3 matches
            top_k = torch.topk(scores, k=3)
            st.markdown("**Other suggestions:**")
            for score, idx in zip(top_k[0], top_k[1]):
                if idx.item() != best_idx:
                    st.write(f"- {HOW_TO_ARTICLES[idx]} (Score: {score:.4f})")
