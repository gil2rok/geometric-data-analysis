import numpy as np
from scipy.sparse.linalg import eigs

class SpectralClustering:
    def __init__(self, X, metric, similarity_method='rbf'):
        self.X = X
        self.num_data_pts = self.X.shape[0]
        self.similarity_method = similarity_method
        self.metric = metric
        
        # compute similarity graph and laplacian
        self.similarity_graph()
        self.laplacian()
                
    def rbf_kernel(self, u, v, sigma=5):
        return np.exp(-self.metric(u, v)**2 / (2 * sigma**2))

    def similarity_graph(self):
        self.S = np.empty((self.num_data_pts, self.num_data_pts))
        if self.similarity_method == 'rbf':
            
            for i in range(self.num_data_pts):
                for j in range(self.num_data_pts):
                    self.S[i, j] = self.rbf_kernel(self.X[i], self.X[j])
                    
        else:
            raise ValueError('Invalid similarity method')
            
    def laplacian(self):
        self.D = np.diag(np.sum(self.S, axis=1))
        self.L = self.D - self.S
 
    def fit(self, k):
        self.k = k
        eig_values, eig_vectors = eigs(self.L, self.k) # columns of eig_vectors are eigenvectors
        return eig_vectors
        