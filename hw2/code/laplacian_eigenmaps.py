import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

from util import rbf_kernel

class LaplacianEigenmaps:
    def __init__(self, embedding_dim, affinity, n_neighbors, sigma=None):
        self.embedding_dim = embedding_dim # embedding dimension
        self.affinity = affinity # 'knn' or 'rbf'
        self.n_neighbors = n_neighbors # num neighbors
        
        # check valid number of neighbors
        try:
            assert(self.n_neighbors > 0)
        except:
            raise ValueError('n_neighbors must be positive integer')
        
         # check valid affinity
        if self.affinity == 'rbf':
            try:
                assert(sigma is not None)
                self.sigma = sigma
            except:
                raise ValueError('sigma must be specified for rbf affinity')
        elif self.affinity == 'knn':
            pass
        else:
            raise ValueError('affinity must be knn or rbf')
        
    def _get_weight_matrix(self):
        n_samples = self.X.shape[0]
        self.W= lil_matrix(np.zeros((n_samples,n_samples))) # weight matrix [n_samples x n_samples]
        
        # iterate over each data point
        for i in range(n_samples):
            xi = self.X[i: i+1] # ith data pt, xi [1 x D]
            
            # find nearest neighbors
            nbrs_est = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(self.X)
            nbrs_idx = nbrs_est.kneighbors(xi, return_distance=False).squeeze() 
            nbrs_idx = nbrs_idx[1:] # remove self
            
            # iterate over each neighbor
            for j in nbrs_idx:
                xj = self.X[j: j+1] # jth data pt, xj [1 x D]
                
                # compute symmetric weight btwn xi, xj per rbf or knn affinity
                if self.affinity == 'rbf':
                    self.W[i, j] = rbf_kernel(xi, xj, self.sigma) 
                    self.W[j, i] = self.W[i, j]
                elif self.affinity == 'knn':
                    self.W[i, j] = 1.
                    self.W[j, i] = self.W[i, j]
        return self.W
    
    def _get_degree_matrix(self):
        self.D = np.diagflat(np.sum(self.W, axis=1)) # degree matrix [n_samples x n_samples]
        return self.D
     
    def _get_laplacian(self):
        W = self._get_weight_matrix() # weight matrix [n_samples x n_samples]
        D = self._get_degree_matrix() # degree matrix [n_samples x n_samples]
        
        L = D - W # Laplacian [n_samples x n_samples]
        return L
                
    def fit_transform(self, X):
        self.X = X
        L = self._get_laplacian()

        _, eig_vec = eigsh(L, self.embedding_dim + 1, self.D, which='SM') # factorize
        Y = eig_vec[:, 1:] # drop trival zero eigvector 
        return Y # embedding [n_samples x embedding_dim]
        
    def fit(self, X):
        self.X = X
        self.fit_transform(X)
        