import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Global Ayarlar ve Model Yüklemesi ---
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Başlangıç referans metin havuzu
reference_texts = [
    "Kedi halının üzerine yattı.",
    "Güneşli bir gündü.",
    "Birden mutfaktan gürültülü bir ses geldi.",
    "Yağmur yağıyordu.",
    "Telefon çaldı."
]
reference_embeddings = encoder.encode(reference_texts)

# En yakın komşu ayarları
n_neighbors = 3
nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(reference_embeddings)

# Loglama fonksiyonu
def log_query(query, results):
    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()} - Sorgu: {query} - Sonuçlar: {results}\n")

# Referans havuzunu güncelleyen fonksiyon
def update_reference_pool(new_texts):
    global reference_texts, reference_embeddings, nbrs
    reference_texts.extend(new_texts)
    reference_embeddings = encoder.encode(reference_texts)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(reference_embeddings)
    print("Referans havuzu güncellendi.")

# K-means kümeleme fonksiyonu
def perform_clustering(n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(reference_embeddings)
    # Küme sonuçlarını referans metinlerle eşleştiriyoruz
    cluster_info = {text: int(cluster) for text, cluster in zip(reference_texts, clusters)}
    return cluster_info

# Sorgu işleme fonksiyonu (tek veya toplu sorgu desteği)
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
                    "status": "Benzerlik düşük, eşleşme yapılmadı"
                })
            else:
                result.append({
                    "text": reference_texts[idx],
                    "similarity": round(sim, 2),
                    "status": "Eşleşme yapıldı"
                })
        elapsed = time.time() - start
        results_all.append({
            "query": query_text,
            "results": result,
            "processing_time_sec": round(elapsed, 4)
        })
        # Loglama işlemi
        log_query(query_text, result)
    total_time = time.time() - start_total
    print(f"Toplam işleme süresi: {total_time:.4f} saniye")
    return results_all

# İnteraktif görselleştirme (Plotly) fonksiyonu
def visualize_embeddings(query_text=None):
    # Gerçek bir boyut indirgeme için PCA kullanılabilir.
    # Örnek amaçlı rastgele 2D koordinatlar üretilmiştir.
    np.random.seed(42)
    coords = np.random.rand(len(reference_texts), 2)
    fig = px.scatter(x=coords[:,0], y=coords[:,1], text=reference_texts,
                     title="Referans Metinlerin Görselleştirmesi")
    if query_text:
        query_embedding = encoder.encode([query_text])
        query_coord = np.random.rand(1, 2)  # örnek koordinat
        fig.add_trace(go.Scatter(x=query_coord[:,0], y=query_coord[:,1],
                                 mode='markers+text', marker=dict(color='red', size=12),
                                 text=[query_text], name="Sorgu Metni"))
    fig.show()

# --- Ana Program Bölümü ---
if __name__ == "__main__":
    # Referans havuzunu güncelleme (isteğe bağlı)
    update_choice = input("Yeni referans metin eklemek ister misiniz? (E/H): ").strip().lower()
    if update_choice == 'e':
        new_texts_input = input("Eklemek istediğiniz metinleri virgülle ayırarak giriniz: ")
        new_texts = [txt.strip() for txt in new_texts_input.split(",") if txt.strip()]
        if new_texts:
            update_reference_pool(new_texts)

    # K-means kümeleme sonucu gösterilsin mi?
    cluster_choice = input("Kümeleme sonuçlarını görmek ister misiniz? (E/H): ").strip().lower()
    if cluster_choice == 'e':
        clusters = perform_clustering(n_clusters=2)
        print("Kümeleme Sonuçları:")
        for text, cluster in clusters.items():
            print(f"'{text}' -> Küme {cluster}")

    # Toplu sorgu desteği: virgülle ayrılmış birden fazla sorgu girişi
    queries_input = input("Sorgu cümlelerini giriniz (virgülle ayırınız): ")
    queries = [q.strip() for q in queries_input.split(",") if q.strip()]
    results = process_queries(queries)
    
    for res in results:
        print("\nSorgu:", res["query"])
        for item in res["results"]:
            print(f"Metin: '{item['text']}' - Cosine Benzerliği: {item['similarity']} - Durum: {item['status']}")

    # İnteraktif görselleştirme: ilk sorgu için örnek
    if queries:
        visualize_embeddings(query_text=queries[0])