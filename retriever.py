from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class VectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self.model.encode(documents)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
    
    def retrieve(self, query, top_k=3, return_indices=False):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding), top_k
    )

        if return_indices:
            return indices[0]

        return [self.documents[i] for i in indices[0]]