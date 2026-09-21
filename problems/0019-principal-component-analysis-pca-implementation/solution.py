import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    data = standarize(data)
    X_centered = data - data.mean(axis=0)

    data = standarize(data)
    cov = covariance_matrix(data)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvectors = fix_eigenvector_signs(eigenvectors)
   
    idx = np.argsort(eigenvalues)[::-1]      
    top_k_eigenvalues = eigenvalues[idx][:k]
    top_k_eigenvectors = eigenvectors[:, idx][:, :k] 

    X_reduced = X_centered @ top_k_eigenvectors
   
    return np.round(top_k_eigenvectors, 4)

def standarize(X):
    mean = X.mean(axis=0) # axis=0 po kolumnach(po cechach)
    std = X.std(axis=0)

    Z = (X - mean) / std
    return Z

def covariance_matrix(X):
    n = X.shape[0]
    X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / (n - 1)


def fix_eigenvector_signs(eigenvectors):
    for i in range(eigenvectors.shape[1]):
        max_abs_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[:, i][max_abs_idx] < 0:
            eigenvectors[:, i] *= -1
    return eigenvectors
