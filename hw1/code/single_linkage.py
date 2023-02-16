import numpy as np

class SingleLinkage:
    def __init__(self, X, metric):
        self.X = X
        self.metric = metric
        
        self.clusters = [[i] for i in self.X] # list of clusters
        self.dendrogram = [self.clusters]
        
    def cluster_distance(self, cluster1, cluster2):
        min_dist = np.inf
        for c1 in cluster1:
            for c2 in cluster2:
                min_dist = min(min_dist, self.metric(c1, c2))
        return min_dist
    
    def clusters_equal(self, cluster, nearest_clusters):
        nearest1_equal, nearest2_equal = True, True
        
        # check if all points in cluster are in nearest_clusters[0]
        for c, nearest1 in zip(cluster, nearest_clusters[0]):
            if np.any(c != nearest1):
                nearest1_equal = False
                break
            
        # check if all points in cluster are in nearest_clusters[1]
        for c, nearest2 in zip(cluster, nearest_clusters[1]):
            if np.any(c != nearest2):
                nearest2_equal = False
                break
            
        return nearest1_equal and nearest2_equal
        
    def fit(self):
        # D = pairwise_distances(self.X, metric=self.metric)
        
        while len(self.clusters) > 1:
            
            # find nearest clusters
            nearest_cluster_dist, nearest_clusters = np.inf, [None, None]
            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    
                    c1, c2 = self.clusters[i], self.clusters[j]
                    nearest_cluster_dist = min(nearest_cluster_dist, self.cluster_distance(c1, c2))
                    nearest_clusters = [c1, c2]
 
            # update clusters
            self.clusters = [c for c in self.clusters if not self.clusters_equal(c, nearest_clusters)]
            self.clusters.append(nearest_clusters[0] + nearest_clusters[1])
            self.dendrogram.append(self.clusters)
            print(len(self.clusters))
        
        
        
        
            