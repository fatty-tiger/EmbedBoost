from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np


class BaseEmbedder(ABC):
    """
    Base class for embedder models.
    Defines the interface for any embedder that can be used in the evaluation framework.
    """
    
    @abstractmethod
    def encode(self, texts: List[str], max_length: int) -> Dict[str, Union[np.ndarray, None]]:
        """
        Encode texts into embeddings.
        
        Args:
            texts (List[str]): List of texts to encode
            
        Returns:
            Dict containing:
                - dense_vectors: numpy array of dense embeddings (required)
                - sparse_vectors: sparse representations (optional)
                - colbert_vectors: colbert-style embeddings (optional)
        """
        pass