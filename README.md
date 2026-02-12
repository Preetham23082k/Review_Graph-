# Review_Graph-
# 🏨 Review Graph Embedding for Hotel Rating Prediction

## 📌 Project Overview

Online hotel reviews contain rich semantic and structural information.
Instead of treating reviews purely as flat text, this project models them
as a **Knowledge Graph** and leverages **Node2Vec graph embeddings**
to predict hotel ratings.

The system integrates:

- NLP-based triple extraction
- Sentiment analysis
- Graph construction
- Graph embedding (Node2Vec)
- Supervised Machine Learning

The goal is to evaluate whether structural graph features combined with
sentiment improve rating prediction.

---

# 🧠 Motivation

Traditional approaches for rating prediction rely on:

- Bag-of-Words
- TF-IDF
- Word embeddings

However, reviews contain structured relational information:

> “staff was helpful”  
> “room had great view”  
> “breakfast was disappointing”

These relational structures can be represented as:

(subject → relation → object)

By modeling reviews as graphs, we capture:

- Structural context
- Co-occurrence patterns
- Relational semantics
- Connectivity information

Graph embeddings allow us to encode this structure numerically.

---

# ⚙️ Complete Pipeline

