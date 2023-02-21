import numpy as np
from scipy.sparse.linalg import eigsh

class SpectralClustering:
    def __init__(self, X, metric, similarity_method='rbf', similarity_param=1):
        self.X = X
        self.num_data_pts = self.X.shape[0]
        self.similarity_method = similarity_method
        self.similarity_param = similarity_param
        self.metric = metric
        
        # compute similarity graph and laplacian
        self.similarity_graph()
        self.laplacian()
                
    def rbf_kernel(self, u, v, sigma):
        return np.exp(-self.metric(u, v)**2 / (2 * sigma**2))

    def similarity_graph(self):
        self.W = np.zeros((self.num_data_pts, self.num_data_pts)) # init weight matrix
        
        # construct weight matrix with rbf kernel
        if self.similarity_method == 'rbf':
            
            # iterate pairwise over data points to compute rbf weight
            for i in range(self.num_data_pts):
                for j in range(self.num_data_pts):
                    self.W[i, j] = self.rbf_kernel(self.X[i], self.X[j], 
                                                   sigma=self.similarity_param)
                   
        # construct weight matrix with k nearest neighbors 
        elif self.similarity_method == 'knn':
            
            # compute pairwise distances
            D = np.empty((self.num_data_pts, self.num_data_pts))
            for i in range(self.num_data_pts):
                D[i,i] = np.inf # exclude self from nearest neighbors 
                for j in range(i+1, self.num_data_pts):
                    D[i,j] = self.metric(self.X[i], self.X[j])
                    D[j,i] = D[i,j]
                    
            # find k nearest neighbors for each data point
            for i in range(self.num_data_pts):
                partition = np.argpartition(D[i], self.similarity_param)
                idx = partition[:self.similarity_param]
                
                self.W[i, idx] = 1
                self.W[idx, i] = 1
             
        # else raise error   
        else:
            raise ValueError('Invalid similarity method')
            
    def laplacian(self):
        self.D = np.diag(np.sum(self.W, axis=0))
        self.L = self.D - self.W
 
    def fit(self, k):
        self.k = k
        eig_values, eig_vectors = eigsh(self.L, self.k, which='SM') # columns of eig_vectors are eigenvectors
        return eig_vectors
        