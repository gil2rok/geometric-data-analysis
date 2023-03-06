# Geometric Data Analysis

The work of Gilad Turok for [COMS 4995] Geometric Data Analysis with Professor Andrew Blumberg in Spring of 2023.

1. **HW1:** implement and compare clustering algorithms: **$k$-means**, **$k$-medians**, **single-linkage clustering**, **spectral clustering** with graphs constructed by $k$ nearest-neighbors and the Gaussian/RBF kernel.
2. **HW2:** implement and compare manifold dimensionality reduction algorithms: **IsoMap**, Multi-Dimensional Scaling (**MDS**), Locally Linear Embedding (**LLE**), and **Laplacian Eigenmaps**.

    **Prove multiple properties of manifolds**: enumerate a space that is not a manifold; show that for any point $m \in \mathcal{M}$ on a manifold, a chart $\theta$ can be constructed such that $\theta(m)=0$; explicate coordinate patches and transition functions for the circle using charts defined by the angle and projections to the axes; prove that solving the MDS matrix factorization problem $A=U \Lambda U^T$ for $m << n$ points approximates the true solution.

    Helpful Resources: [Github tutorial on manifold learning](https://github.com/drewwilimitis/Manifold-Learning) and [paper summarizing manifold learning algorithms](https://www.cs.columbia.edu/~verma/classes/ml/ref/lec8_cayton_manifolds.pdf).

3. **HW3:**