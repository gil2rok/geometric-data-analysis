import numpy as np

seed = 1
rng = np.random.default_rng(seed)

class KMedians:
    def __init__(self, X, metric):
        """ Class for K medians clustering

        Args:
            X ([m x n]): m data points in n dimensional space
            metric (callable): function to compute distance
        """
        
        self.X = X # data to cluster
        self.metric = metric # callable metric
        self.m = self.X.shape[0] # num data points
        
    def init_means(self):
        """ Select k data points to initialize k means (Forgy method)
        """
        
        idx = rng.integers(low=0, high=self.m, size=self.k) 
        self.means = self.X[idx]
        
    def assign_clusters(self):
        self.assignments = [None] * self.m
        # iterate over all m datapoints
        for i in range(self.m):
            x_i = self.X[i,:] # data point
            min_dist = np.array([np.inf]) # init min dist
            min_assignement = None # init min assignment
            
            # iterate over all means
            for j in range(self.k):
                x_j = self.means[j,:] # cluster mean
                cur_dist = self.metric(x_i, x_j) # distance btwn data point and cluster mean
                
                # update min distance and cluster assignement
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_assignement = j
                    
            self.assignments[i] = (min_assignement, min_dist) # save cluster assignement         

    def update_clusters(self):
        # iterate over all k means
        for i in range(self.k):
            cluster_data = []
            
            # iterate over all data points in k-th mean
            for j, (assignment, dist) in enumerate(self.assignments):
                if assignment == i:
                    cluster_data.append(self.X[j])
            
            # update k-th mean
            self.means[i, :] = np.median(cluster_data, axis=0)
      
    def step(self):
        self.update_clusters()
        self.assign_clusters()
        return [cluster[0] for cluster in self.assignments]
        
    def fit(self, k):
        # init k means
        self.k = k
        self.init_means()
        self.assign_clusters()
        
        # init prev and cur cluster assignements
        prev_assignements = self.step() 
        cur_assignements = self.step()
        
        # while prev and cur assignements are different, update clusters
        while prev_assignements != cur_assignements:
            prev_assignements = cur_assignements
            cur_assignements = self.step()