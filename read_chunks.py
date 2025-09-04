import requests

def create_embedding(prompt):
    r = requests.post("http://localhost:11434/api/embeddings",
                      json={
                          "model": "bge-m3",
                          "prompt": prompt
                      })
    embedding = r.json()["embedding"]
    return embedding

a = create_embedding("Hello, world!")
print(a)

