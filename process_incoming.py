import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed",
                      json={
                          "model": "bge-m3",
                          "input": text_list
                      })
    embedding = r.json()["embeddings"]
    return embedding

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a question: ")
query_embedding = create_embedding([incoming_query])[0]

#Find the similarity of question embedding with others embedding
similarities = cosine_similarity(np.vstack(df["embedding"]), [query_embedding]).flatten()
print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][0::top_results]
print(max_indx)
new_df = df.loc[max_indx]
print(new_df[["text", "chunk_id"]])