import numpy as np

class VectorDB:
    def __init__(self):
        self.vector_map = {}

    def add_vector(self, vector, text):
        self.vector_map[text] = vector

    def search_similar_vectors(self, query_vector, k, distance_func):
        distances = [distance_func(query_vector, vector) for vector in self.vector_map.values()]
        indices = np.argsort(distances)[:k]
        return [(list(self.vector_map.values())[i], list(self.vector_map.keys())[i]) for i in indices]

# Example usage
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))


db = VectorDB()
db.add_vector(np.array([1, 2, 3]), "Text 1")
db.add_vector(np.array([4, 5, 6]), "Text 2")
db.add_vector(np.array([7, 8, 9]), "Text 3")

query_vector = np.array([2, 3, 4])
similar_vectors = db.search_similar_vectors(query_vector, k=2, distance_func=euclidean_distance)

for vector, text in similar_vectors:
    print(f"Similar vector: {vector}, Text: {text}")