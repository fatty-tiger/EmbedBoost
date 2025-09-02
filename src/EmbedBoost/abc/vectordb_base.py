import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

from EmbedBoost.abc.embedder import BaseEmbedder


class BaseVectorStore(ABC):
    """
    Base class for vector database operations.
    Supports building indices and performing different types of retrieval.
    """
    
    @abstractmethod
    def insert_docs(self, documents: List[Dict[str, Any]], embedder: BaseEmbedder, max_length: int) -> Any:
        """
        Build vector index from texts using the provided embedder.
        
        Args:
            texts (List[str]): List of texts to index
            embedder (BaseEmbedder): Embedder model to use for encoding texts
            max_length (int): sequence max_length for encoding
            
        Returns:
            Any: Vector index (implementation-specific)
        """
        pass
    
    @abstractmethod
    def dense_retrieval(self, dense_vectors: np.ndarray, topk: int) -> List[Dict[str, Union[str, float]]]:
        """
        Perform dense retrieval using dense vectors.
        
        Args:
            dense_vectors: Query dense embeddings
            topk (int): Number of top results to return
            
        Returns:
            List of retrieved documents with ids and scores:
            [{"id": "id1", "text": "text1", "score": 0.9}, ...]
        """
        pass
    
    @abstractmethod
    def sparse_retrieval(self, sparse_vectors: Any, topk: int) -> List[Dict[str, Union[str, float]]]:
        """
        Perform sparse retrieval using sparse vectors.
        
        Args:
            sparse_vectors: Query sparse embeddings
            topk (int): Number of top results to return
            
        Returns:
            List of retrieved documents with ids and scores:
            [{"id": "id1", "text": "text1", "score": 0.9}, ...]
        """
        pass