import requests
import os
import json
import pandas as pd
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


jsons = os.listdir("jsons")   # List all the jsons 
my_dicts = []
chunk_id = 0 
 
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Processing {json_file} with {len(content['chunks'])} chunks")
    embedddings = create_embedding([c["text"] for c in content["chunks"]])
    
    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
df = pd.DataFrame.from_records(my_dicts)

# Save the dataframe to joblib file
joblib.dump(df, "embeddings.joblib")
