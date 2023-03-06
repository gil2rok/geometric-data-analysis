import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

class LocallyLinearEmbeddings:
    def __init__(self, n_neighbors, embedding_dim):
        self.n_neighbors = n_neighbors
        self.embedding_dim = embedding_dim
        
    def _numerical_stability(self, G):
        reg = 1e-3
        trace = np.trace(G)
        
        if trace > 0:
            R = reg * trace
        else:
            R = reg
            
        G.flat[:: self.n_neighbors + 1] += R # add regularization
        return G
        
    def _reconstruction_weights(self):
        n = self.X.shape[0] # num data pts
        W = lil_matrix(np.zeros((n,n))) # weight matrix [n x n]
        
        # nearest neighbors estimator
        nbrs_est = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)
        
        for i in range(n):
            xi = self.X[i:i+1] # i-th data pt, xi [1 x D]
            nbrs_idx = nbrs_est.kneighbors(xi, return_distance=False).squeeze() # nearest neighbors idx
            nbrs_idx = nbrs_idx[1:] # remove self
            nbrs = self.X[nbrs_idx] # nearest neighbors [n_neighbors x D]
        
            G = (xi - nbrs) @ (xi - nbrs).T # gram matrix [n_neighbors x n_neighbors]
            ones = np.ones((self.n_neighbors, 1)) # ones vec [n_neighbors x 1]
            
            G = self._numerical_stability(G) # increase numerical stability
            W_tilde = solve(G, ones, assume_a='sym') # reconstruction weights [n_neighbors x 1]
            
            W_tilde /= np.sum(W_tilde) # normalize
            W[i, nbrs_idx] = W_tilde # input into sparse matrix
        return W
    
    def _embedding(self):
        n = self.X.shape[0] # num data pts
        I = lil_matrix(np.eye(n)) # sparse identity matrix [n x n]
        M = (I - self.W).T @ (I - self.W) # [n x n]
        _, eig_vec = eigsh(M, k=self.embedding_dim + 1, which='SM')
        return eig_vec[:, 1:] # remove first eigenvector [n x embedding_dim]
        
    def fit_transform(self, X):
        self.X = X # data matrix [n x D]
        
        self.W = self._reconstruction_weights() # sparse weight matrix [n x n]
        return self._embedding() # embedding [n x embedding_dim]

    def fit(self, X, d):
        self.fit_transform(X)
        return self
        
        
        