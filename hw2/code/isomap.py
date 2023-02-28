import numpy as np
from sklearn.neighbors import KNeighborsTransformer 
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigs

class IsoMap:
    def __init__(self, X):
        self.n = X.shape[0] # num data points
        self.X = X
        
    def mds(self):
        H = np.eye(self.n) - np.ones((self.n, self.n)) / self.n # centering matrix
        B = -0.5 * H @ self.D @ H # gram matrix
        eig_values, eig_vec = eigs(B, k=self.d) # spectral decomposition, compute d largest eigenvalues
        eig_values = np.maximum(eig_values, 0) # remove negative eigenvalues
        E = eig_vec @ np.diag(np.sqrt(eig_values)) # compute embedding
        return E
    
    def fit(self, k, d):
        self.d = d
        
        self.W = KNeighborsTransformer(n_neighbors=k, mode='distance').fit_transform(self.X)
        self.D = np.power(shortest_path(self.W, directed=False), 2)
        self.Y = self.mds()
        return self.Y
        