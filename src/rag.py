import os
import math
from datetime import datetime
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_SAMPLE_PATH = os.path.join(BASE_DIR, "data", "raw_sample.csv")

class RAGRetriever:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, collection_name: str = "amazon_support_history", decay_lambda: float = 0.001):
        print(f"Initializing ChromaDB at '{db_path}' and local embedding model...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.decay_lambda = decay_lambda  # Exponential time decay rate per day
        
        # Local open-source embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def populate_index(self, sample_csv_path: str = DEFAULT_SAMPLE_PATH):
        """
        Loads preprocessed dataset into ChromaDB if the index is empty.
        """
        if self.collection.count() > 0:
            print(f"Collection already populated with {self.collection.count()} records.")
            return

        print(f"Indexing historical responses from {sample_csv_path}...")
        df = pd.read_csv(sample_csv_path)

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            documents.append(str(row["customer_text"]))
            metadatas.append({
                "brand_response": str(row["brand_response"]),
                "customer_tweet_id": str(row["customer_tweet_id"]),
                "created_at": str(row.get("created_at", ""))
            })
            ids.append(f"doc_{idx}")

        # Add records to vector store in safe batches
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
        print(f"Successfully indexed {len(documents)} resolution pairs into ChromaDB.")

    def _parse_created_at(self, date_str: str) -> datetime | None:
        if not date_str:
            return None
        # Example format: "Wed Nov 22 09:23:01 +0000 2017"
        try:
            return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            try:
                return pd.to_datetime(date_str)
            except Exception:
                return None

    def retrieve_context(self, query: str, intent: str = None, top_k: int = 3) -> list:
        """
        Retrieves top_k similar historical resolution pairs for grounding.
        Applies exponential age-decay scoring to suppress outdated policy information.
        """
        fetch_k = max(top_k * 3, 10)
        query_kwargs = {
            "query_texts": [query],
            "n_results": fetch_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if intent:
            query_kwargs["where"] = {"intent": intent}

        try:
            results = self.collection.query(**query_kwargs)
        except Exception:
            # Fallback to query without metadata filter if intent filter isn't present
            query_kwargs.pop("where", None)
            results = self.collection.query(**query_kwargs)

        scored_candidates = []
        now = datetime.now().astimezone()

        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0] if "distances" in results else [0.5] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Convert L2 / Cosine distance to similarity (1 / (1 + dist))
                base_similarity = 1.0 / (1.0 + float(dist)) if dist is not None else 0.5
                
                # Calculate time decay
                created_at_str = meta.get("created_at", "")
                doc_dt = self._parse_created_at(created_at_str)

                if doc_dt:
                    try:
                        age_days = (now - doc_dt).total_seconds() / 86400.0
                        time_weight = math.exp(-self.decay_lambda * max(age_days, 0.0))
                    except Exception:
                        time_weight = 1.0
                else:
                    time_weight = 1.0

                final_score = base_similarity * time_weight

                scored_candidates.append({
                    "historical_customer_query": doc,
                    "historical_brand_response": meta.get("brand_response", ""),
                    "score": final_score,
                    "base_similarity": base_similarity,
                    "time_weight": time_weight
                })

        # Rank by time-decay adjusted score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

if __name__ == "__main__":
    retriever = RAGRetriever()
    retriever.populate_index()

    test_query = "My package was supposed to arrive yesterday but the tracking number shows delayed."
    contexts = retriever.retrieve_context(test_query, top_k=2)

    print("\n--- Retrieved RAG Contexts ---")
    for i, ctx in enumerate(contexts, 1):
        print(f"\nResult #{i}:")
        print(f"Past Customer Query: {ctx['historical_customer_query']}")
        print(f"Past Brand Answer:   {ctx['historical_brand_response']}")