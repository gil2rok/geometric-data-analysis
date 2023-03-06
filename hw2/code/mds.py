import numpy as np

from numpy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

class MultiDimensionalScaling:
    """ Perform Classical Multi-Dimensional Scaling (MDS)

        Parameters:
            dissimilarity (str): indicate of dissimilarity matrix is provided or
                                    if euclidean distance should be used
            d (int): dimension of embedding
        """
        
    def __init__(self, d, dissimilarity):
        self.dissimilarity = dissimilarity # dissimilarity type
        self.d = d # embedding dimension   
        
        # validate parameters
        assert(d > 0)
        assert(type(d) == int)
        try:
            dissimilarity in ['euclidean', 'precomputed']
        except:
            raise ValueError('Invalid dissimilarity type')
    
    def fit(self, X):
        self.X = X
        self.n = X.shape[0] # num data points
        
        # obtain [n x n] dissimilarity matrix
        if self.dissimilarity == 'euclidean':
            self.D = squareform(pdist(self.X, metric='euclidean'))
            assert(self.d <= self.n)
        else:
            self.D = self.X
            
        H = np.eye(self.n) - (1 / self.n) * np.ones((self.n, self.n)) # centering matrix [n x n]
        B = -0.5 * H @ (self.D ** 2) @ H  # gram matrix [n x n]
        
        eig_val, eig_vec = eigh(B) # eig_val in ascending order
        eig_val = np.maximum(eig_val, 0) # when D is NOT pos def, take max with 0
        
        Y = eig_vec @ np.diag(np.sqrt(eig_val)) # compute embedding [n x n]
        self.Y = Y[:, -self.d:] # reduce to d dimensions [n x d]
        
    def fit_transform(self, X):
        """ Fit and transform X into lower, d-dimensional embedding

        Args:
            X: [n x n] dissimilarity matrix or [n x D] data matrix

        Returns:
            Y ([n x d]): d dimensional embedding of X
        """
        
        self.fit(X)
        return self.Y
        