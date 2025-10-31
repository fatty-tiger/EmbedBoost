import numpy as np

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors.
    
    Args:
        vectors: Numpy array of vectors
        
    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


if __name__ == '__main__':
    x = np.arange(24).reshape(2, 3, 4)
    norms = normalize_vectors(x)
    print(x)
    print("")
    print(norms)

    lst = norms[0][0].tolist()
    print(lst)
    sum = 0
    for x in lst:
        sum += x ** 2
    print(sum)

