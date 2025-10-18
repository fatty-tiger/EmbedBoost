import numpy as np

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors.
    
    Args:
        vectors: Numpy array of vectors
        
    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms