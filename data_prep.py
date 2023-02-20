import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DIM = 768

encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
dataset = load_dataset("IlyaGusev/gazeta", revision="v1.0")

data = pd.DataFrame(dataset["train"])
embeddings = encoder.encode(data["summary"].values, show_progress_bar=True)
np.save("embeddings.npy", embeddings)

index = faiss.IndexFlatIP(DIM)
faiss.normalize_L2(embeddings)
index.add(embeddings)
faiss.write_index(index, "index_summaries")

sample_queries = ["МЧС прилагает все усилия для тушения лесных пожаров в Западной Сибири", "Президент России Владимир Путин провёл встречу с министром МВД Колокольцевым", \
    "Британские учёные обнаружили новый вид рыб в Тихом океане", "Астронавты НАСА провели выход в открытый космос на МКС"]

with open("sample_results.txt", "w", encoding="utf-8") as f:
    for query in sample_queries:
        f.write(f"Запрос: {query}\n")
        query_vector = encoder.encode([query])
        faiss.normalize_L2(query_vector)
        _, indices = index.search_indices(query_vector, 10)
                        
        for i, idx in enumerate(indices[0]):
            news = data["summary"].iloc[idx]
            url = data["url"].iloc[idx]
            lines = [f"{i + 1} Новость: {news}", f"Ссылка: {url}\n"]
            f.write("\n".join(lines))           
        f.write("\n")

