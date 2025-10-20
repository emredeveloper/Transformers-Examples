import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Global configuration and model loading ---
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initial pool of reference texts
reference_texts = [
    "The cat lay down on the rug.",
    "It was a sunny day.",
    "A loud noise suddenly came from the kitchen.",
    "It was raining.",
    "The phone rang."
]
reference_embeddings = encoder.encode(reference_texts)

# Nearest-neighbour configuration
n_neighbors = 3
nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(reference_embeddings)

# Logging helper
def log_query(query, results):
    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()} - Query: {query} - Results: {results}\n")

# Function to update the reference pool
def update_reference_pool(new_texts):
    global reference_texts, reference_embeddings, nbrs
    reference_texts.extend(new_texts)
    reference_embeddings = encoder.encode(reference_texts)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(reference_embeddings)
    print("Reference pool updated.")

# K-means clustering helper
def perform_clustering(n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reference_embeddings)
    # Pair the cluster assignments with the reference texts
    cluster_info = {text: int(cluster) for text, cluster in zip(reference_texts, clusters)}
    return cluster_info

# Query processing function (single or batched queries)
def process_queries(queries, similarity_threshold=0.7):
    if isinstance(queries, str):
        queries = [queries]

    results_all = []
    start_total = time.time()
    for query_text in queries:
        start = time.time()
        pred_embedding = encoder.encode([query_text])
        distances, indices = nbrs.kneighbors(pred_embedding)
        result = []
        for dist, idx in zip(distances[0], indices[0]):
            sim = 1 - dist  # cosine similarity
            if sim < similarity_threshold:
                result.append({
                    "text": reference_texts[idx],
                    "similarity": round(sim, 2),
                    "status": "Similarity too low, no match"
                })
            else:
                result.append({
                    "text": reference_texts[idx],
                    "similarity": round(sim, 2),
                    "status": "Match found"
                })
        elapsed = time.time() - start
        results_all.append({
            "query": query_text,
            "results": result,
            "processing_time_sec": round(elapsed, 4)
        })
        # Write to the log file
        log_query(query_text, result)
    total_time = time.time() - start_total
    print(f"Total processing time: {total_time:.4f} seconds")
    return results_all

# Interactive visualisation (Plotly) function
def visualize_embeddings(query_text=None):
    # PCA could be applied for real dimensionality reduction.
    # Random 2D coordinates are generated for demonstration purposes.
    np.random.seed(42)
    coords = np.random.rand(len(reference_texts), 2)
    fig = px.scatter(x=coords[:,0], y=coords[:,1], text=reference_texts,
                     title="Reference text visualisation")
    if query_text:
        query_embedding = encoder.encode([query_text])
        query_coord = np.random.rand(1, 2)  # sample coordinate
        fig.add_trace(go.Scatter(x=query_coord[:,0], y=query_coord[:,1],
                                 mode='markers+text', marker=dict(color='red', size=12),
                                 text=[query_text], name="Query Text"))
    fig.show()

# --- Main program block ---
if __name__ == "__main__":
    # Optionally update the reference pool
    update_choice = input("Would you like to add new reference texts? (y/n): ").strip().lower()
    if update_choice == 'y':
        new_texts_input = input("Enter the texts you want to add, separated by commas: ")
        new_texts = [txt.strip() for txt in new_texts_input.split(",") if txt.strip()]
        if new_texts:
            update_reference_pool(new_texts)

    # Show the K-means clustering results?
    cluster_choice = input("Would you like to see the clustering results? (y/n): ").strip().lower()
    if cluster_choice == 'y':
        clusters = perform_clustering(n_clusters=2)
        print("Clustering Results:")
        for text, cluster in clusters.items():
            print(f"'{text}' -> Cluster {cluster}")

    # Batch query support: supply multiple queries separated by commas
    queries_input = input("Enter query sentences (separated by commas): ")
    queries = [q.strip() for q in queries_input.split(",") if q.strip()]
    results = process_queries(queries)
    
    for res in results:
        print("\nQuery:", res["query"])
        for item in res["results"]:
            print(f"Text: '{item['text']}' - Cosine Similarity: {item['similarity']} - Status: {item['status']}")

    # Interactive visualisation: display an example for the first query
    if queries:
        visualize_embeddings(query_text=queries[0])