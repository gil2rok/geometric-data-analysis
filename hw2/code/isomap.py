from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path

from mds import MultiDimensionalScaling

class IsoMap:
    def __init__(self, n_neighbors, d):
        self.n_neighbors = n_neighbors
        self.d = d # embedding dimension
        
    def fit(self, X):
        # k-nearest neighbors graph with euclidean weights for neighbors xi, xj
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        nbg = kneighbors_graph(nbrs, self.n_neighbors, mode='distance') # [n x num_neighbors]
        print(nbg.shape)
        
        # symmetrical shortest-path dist matrix
        dist_matrix = shortest_path(nbg, directed=False) # [n x n]
        
        # multi-dimensional scaling
        mds = MultiDimensionalScaling(self.d, 'precomputed')
        self.embedding = mds.fit_transform(dist_matrix) # [n x d]
        
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding
        
        