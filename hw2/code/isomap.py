import numpy as np
from sklearn.neighbors import KNeighborsTransformer 
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eig

class IsoMap:
    def __init__(self, X):
        self.n = X.shape[0] # num data points
        self.X = X
        
    def mds(self):
        n = self.n # write explicitly for readability
        H = np.eye(n) - (np.ones((n. n)) / n) # centering matrix
        B = -0.5 * (H @ self.D @ H) # gram matrix
        eig_values, eig_vec = eig(B) # spectral decomposition
        print(eig_values)
        eig_values = np.maximum(eig_values, 0) # remove negative eigenvalues
        E = eig_vec @ np.diag(np.sqrt(eig_values)) # compute embedding
        return E[:, :self.d]
    
    def fit(self, k, d):
        self.d = d # embedding dimension d
        
        self.W = KNeighborsTransformer(n_neighbors=k, mode='distance').fit_transform(self.X)
        self.D = np.power(shortest_path(self.W, directed=False), 2) + 1e-7
        self.Y = self.mds() # lower dimensional embedding
        return self.Y