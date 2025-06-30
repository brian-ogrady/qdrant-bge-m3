import os
from typing import Union, List, Dict

from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


class BGEM3:
    model = BGEM3FlagModel("BAAI/bge-m3")
    dense_vector_config = VectorParams(
        size=1024,
        distance=Distance.COSINE,
    )
    sparse_vector_config = SparseVectorParams()

    def encode(self, text: Union[str, List[str]]) -> Dict[str, List[float]]:
        if isinstance(text, str):
            text = [text]
        return self.model.encode(
            text,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )


class bm25:
    model = SparseTextEmbedding("Qdrant/bm25")
    
    sparse_vector_config = SparseVectorParams()

    def encode(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        return list(self.model.embed(text))


class gte:
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    dense_vector_config = VectorParams(
        size=768,
        distance=Distance.COSINE,
    )

    def encode(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text)


gte = gte()
bm25 = bm25()
bgem3 = BGEM3()


def _setup_qdrant() -> QdrantClient:
    """
    Initialize and configure a Qdrant client instance.
    
    Uses environment variables to determine host and port, with defaults if not specified:
    - QDRANT_HOST: Defaults to 'localhost'
    - QDRANT_PORT: Defaults to '6333'
    
    Returns:
        QdrantClient: Configured Qdrant client with a 180-second timeout
    """
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=180)
    return client


def _initialize_collection(client: QdrantClient, collection_name: str) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "bge_dense": bgem3.dense_vector_config,
                "gte": gte.dense_vector_config,
            },
            sparse_vectors_config={"bm25": bm25.sparse_vector_config}
        )


if __name__ == "__main__":
    client = _setup_qdrant()
    _initialize_collection(client, "test")
    texts = ["Hello, world!", "This is a test."]
    insertion = {
        "bge_dense": bgem3.encode(texts)["dense_vecs"],
        "bm25": [
            types.SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist()
            ) for emb in bm25.encode(texts)
        ],
        "gte": gte.encode(texts),
    }

    print(insertion)
    client.upsert(
        collection_name="test",
        points=[
            types.PointStruct(
                id=1,
                vector={model_name: vector[0] for model_name, vector in insertion.items()},
                payload={"text": texts[0]},
            ),
            types.PointStruct(
                id=2,
                vector={model_name: vector[1] for model_name, vector in insertion.items()},
                payload={"text": texts[1]},
            ),
        ],
    )
        
    